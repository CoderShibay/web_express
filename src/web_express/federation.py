"""Ask several sources at once and merge their answers into one ranked list."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from .net import USER_AGENT
from .sources import CATEGORIES, DuckDuckGo, Hit, Source, SourceError, default_sources

FRESHNESS_DAYS = {"day": 1, "week": 7, "month": 31, "year": 366}
QUEUE_LIMIT = 25.0  # longest a search waits for one source, including its rate-limit queue
RRF_K = 60
_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|mc_[ce]id|ref|ref_src|igshid)$", re.IGNORECASE)

_CODE_HINT = re.compile(
    r"\w\.\w+\(|\w::\w|`|\b(error|exception|traceback|stack ?trace|python|javascript|typescript|kotlin|java|rust|"
    r"golang|c\+\+|npm|pip|sdk|api|library|function|compile|compiler|github|android|docker|kubernetes|sql|regex|"
    r"segfault|undefined|null pointer|import)\b",
    re.IGNORECASE,
)
_PAPER_HINT = re.compile(
    r"\b(papers?|stud(y|ies)|arxiv|pubmed|doi|journal|trial|cohort|meta-analysis|systematic review|et al|"
    r"benchmark|state of the art|survey|literature|preprint|clinical|patients?|tumou?rs?|radiomics|"
    r"segmentation|neural network|transformer|dataset)\b",
    re.IGNORECASE,
)
_NEWS_HINT = re.compile(r"\b(news|latest|today|yesterday|this week|announced|announcement|breaking|headline)\b", re.IGNORECASE)


def guess_categories(query: str, freshness: str = "") -> list[str]:
    """For category 'auto': always the web, plus any specialist area the query hints at."""
    categories = ["web"]
    if _PAPER_HINT.search(query):
        categories.append("papers")
    if _CODE_HINT.search(query):
        categories.append("code")
    if _NEWS_HINT.search(query) or freshness in ("day", "week"):
        categories.append("news")
    return categories


def canonical_url(url: str) -> str:
    """A form of the URL that is the same for trivially different links to one page.

    Never raises: a URL that cannot be parsed is returned unchanged.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower().rstrip(".").removeprefix("www.")
        path = parts.path.rstrip("/")
        if host == "arxiv.org":
            path = re.sub(r"v\d+$", "", re.sub(r"^/(pdf|html)/", "/abs/", path).removesuffix(".pdf"))
        query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING.match(k)])
        port = f":{parts.port}" if parts.port and parts.port not in (80, 443) else ""
        return urlunsplit(("https", host + port, path, query, ""))
    except ValueError:
        return url.strip()


def valid_hit(hit) -> bool:
    """Whether a source's row is safe to rank and return: right types, a real http(s) link."""
    if not isinstance(hit, Hit) or not all(isinstance(v, str) for v in (hit.title, hit.url, hit.snippet, hit.source)):
        return False
    if not isinstance(hit.extra, dict) or not isinstance(hit.published, (str, type(None))):
        return False
    if not isinstance(hit.content, (str, type(None))):
        return False
    try:
        parts = urlsplit(hit.url)
        _ = parts.port
    except ValueError:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    if "doi" in hit.extra and not isinstance(hit.extra["doi"], str):
        del hit.extra["doi"]
    return True


def _identity(hit: Hit) -> str:
    doi = hit.extra.get("doi")
    return f"doi:{doi.lower()}" if doi else canonical_url(hit.url)


@dataclass
class Outcome:
    hits: list[Hit]
    used: list[str]  # sources that answered
    categories: list[str] = field(default_factory=list)  # what the search covered
    errors: dict[str, str] = field(default_factory=dict)  # source -> why it did not answer


class Federation:
    def __init__(self, sources: list[Source] | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.sources = default_sources() if sources is None else sources
        self._transport = transport
        self._paused: dict[str, tuple[float, int]] = {}  # source -> (retry after, failures so far)

    def _select(self, categories: list[str]) -> list[Source]:
        return [s for s in self.sources if s.available() and any(c in s.categories for c in categories)]

    def _is_paused(self, source: Source) -> bool:
        until, _ = self._paused.get(source.name, (0.0, 0))
        return time.monotonic() < until

    def _record(self, source: Source, ok: bool) -> None:
        if ok:
            self._paused.pop(source.name, None)
            return
        failures = self._paused.get(source.name, (0.0, 0))[1] + 1
        pause = min(60 * 2 ** (failures - 1), 900)  # 1, 2, 4 ... up to 15 minutes
        self._paused[source.name] = (time.monotonic() + pause, failures)

    async def _ask(self, source, client, query, limit, since, freshness, region, web_suffix) -> list[Hit]:
        if isinstance(source, DuckDuckGo):  # the only source that understands site: operators
            return await source.search(client, query + web_suffix, limit, since, freshness, region)
        return await source.search(client, query, limit, since)

    async def search(
        self, query: str, category: str = "auto", limit: int = 10, freshness: str = "", region: str = "",
        web_suffix: str = "",
    ) -> Outcome:
        if category != "auto" and category not in CATEGORIES:
            raise ValueError(f"category must be one of: auto, {', '.join(CATEGORIES)}.")
        if category == "news" and not freshness:
            freshness = "month"  # news means recent
        categories = guess_categories(query, freshness) if category == "auto" else [category]
        since = date.today() - timedelta(days=FRESHNESS_DAYS[freshness]) if freshness else None
        chosen = self._select(categories)
        errors = {s.name: "paused after repeated failures" for s in chosen if self._is_paused(s)}
        active = [s for s in chosen if not self._is_paused(s)]
        per_source = max(limit, 8)

        async def run(source: Source) -> list[Hit]:
            # Each source times its own requests. This outer limit only stops a search
            # from queueing behind a source's rate limit for ever.
            answer = await asyncio.wait_for(
                self._ask(source, client, query, per_source, since, freshness, region, web_suffix), QUEUE_LIMIT
            )
            if not isinstance(answer, list):
                raise SourceError(f"{source.name}: sent an unexpected answer.")
            return [hit for hit in answer if valid_hit(hit)]

        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=10, follow_redirects=True, transport=self._transport,
        ) as client:
            answers = await asyncio.gather(*(run(s) for s in active), return_exceptions=True)

        ranked: list[tuple[Source, list[Hit]]] = []
        for source, answer in zip(active, answers):
            if isinstance(answer, SourceError):
                errors[source.name] = str(answer)
            elif isinstance(answer, TimeoutError):  # our own queue, not the source: do not pause it
                errors[source.name] = f"{source.name}: busy, skipped this time."
                continue
            elif isinstance(answer, BaseException):
                errors[source.name] = f"{source.name}: failed ({type(answer).__name__})."
            else:
                ranked.append((source, answer))
            self._record(source, not isinstance(answer, BaseException))
        hits = fuse([hits for _, hits in ranked], since)
        return Outcome(hits[:limit], [s.name for s, _ in ranked], errors=errors, categories=categories)


def fuse(result_lists: list[list[Hit]], since: date | None = None) -> list[Hit]:
    """Reciprocal rank fusion: a page ranked well by several sources rises to the top."""
    merged: dict[str, Hit] = {}
    scores: dict[str, float] = {}
    for hits in result_lists:
        seen_here: set[str] = set()
        for rank, hit in enumerate(hits):
            if since and hit.published and hit.published < since.isoformat():
                continue
            key = _identity(hit)
            if key in seen_here:
                continue
            seen_here.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            kept = merged.get(key)
            if kept is None:
                merged[key] = hit
                hit.extra.setdefault("sources", [hit.source])
            else:  # the same page from another source: keep the richer details
                kept.extra["sources"].append(hit.source)
                kept.content = kept.content or hit.content
                kept.published = kept.published or hit.published
                kept.snippet = kept.snippet or hit.snippet
    return [merged[key] for key in sorted(scores, key=lambda k: -scores[k])]
