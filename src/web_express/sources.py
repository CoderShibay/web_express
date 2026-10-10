"""Where web_express looks for results.

Every source is free and works without an account. A few get better with a
free key or a contact address, read from environment variables:

    WEB_EXPRESS_CONTACT        URL or email; enables Wikipedia, polite pool on OpenAlex
    WEB_EXPRESS_GITHUB_TOKEN   raises GitHub's search limit
    WEB_EXPRESS_OPENALEX_KEY   raises OpenAlex's daily allowance

Each source returns Hit rows. A Hit may carry `content` (an abstract, an
answer) taken straight from the source's API, which saves fetching the page.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html import unescape

import httpx

from .engines import DuckDuckGo as _DuckDuckGoEngine
from .engines import EngineError
from .extract import _plain_html

CATEGORIES = ("web", "papers", "code", "news")


class SourceError(Exception):
    """A source could not answer; the message is safe to show."""


@dataclass
class Hit:
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    published: str | None = None  # ISO date, when the source knows it
    content: str | None = None  # text supplied by the source's API
    extra: dict = field(default_factory=dict)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _html_to_text(html: str) -> str:
    return _plain_html(html or "", "")[1]


def _iso_date(value) -> str | None:
    """Best-effort ISO date from a timestamp, an epoch number or a date string."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).date().isoformat()
    match = re.match(r"\d{4}-\d{2}-\d{2}", str(value))
    return match.group(0) if match else None


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Source:
    name = ""
    categories: tuple[str, ...] = ()
    min_interval = 0.0  # seconds between requests to this source
    timeout = 8.0

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last = 0.0

    def available(self) -> bool:
        return True

    async def search(self, client: httpx.AsyncClient, query: str, limit: int, since: date | None) -> list[Hit]:
        raise NotImplementedError

    async def _get(self, client: httpx.AsyncClient, url: str, params: dict, headers: dict | None = None) -> httpx.Response:
        async with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                raise SourceError(f"{self.name}: could not connect ({type(exc).__name__}).") from exc
            finally:
                self._last = time.monotonic()
        if response.status_code in (403, 429):
            raise SourceError(f"{self.name}: rate limit reached (HTTP {response.status_code}).")
        if response.status_code != 200:
            raise SourceError(f"{self.name}: answered HTTP {response.status_code}.")
        return response

    async def _json(self, client: httpx.AsyncClient, url: str, params: dict, headers: dict | None = None) -> dict:
        response = await self._get(client, url, params, headers)
        try:
            data = response.json()
        except ValueError as exc:
            raise SourceError(f"{self.name}: sent an unreadable answer.") from exc
        if not isinstance(data, dict):
            raise SourceError(f"{self.name}: sent an unexpected answer.")
        return data


# --- web ---------------------------------------------------------------------

class DuckDuckGo(Source):
    name = "duckduckgo"
    categories = ("web", "news")
    timeout = 12.0

    def __init__(self) -> None:
        super().__init__()
        self._engine = _DuckDuckGoEngine()

    async def search(self, client, query, limit, since, freshness: str = "", region: str = ""):
        try:
            hits = await self._engine.search(query, freshness, region)
        except EngineError as exc:
            raise SourceError(str(exc)) from exc
        return [Hit(h.title, h.url, h.snippet, self.name) for h in hits]


class Wikipedia(Source):
    """Needs WEB_EXPRESS_CONTACT: Wikimedia refuses clients that give no contact."""

    name = "wikipedia"
    categories = ("web",)

    def available(self) -> bool:
        return bool(_env("WEB_EXPRESS_CONTACT"))

    async def search(self, client, query, limit, since):
        data = await self._json(client, "https://en.wikipedia.org/w/api.php", {
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": min(limit, 5),
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": "max", "inprop": "url", "format": "json",
        })
        pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        return [
            Hit(p.get("title", ""), p["fullurl"], _clip(p.get("extract", "")), self.name,
                _iso_date(p.get("touched")), p.get("extract") or None)
            for p in pages if p.get("fullurl")
        ]


# --- papers ------------------------------------------------------------------

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


class Arxiv(Source):
    name = "arxiv"
    categories = ("papers",)
    min_interval = 3.0  # arXiv asks for one request every three seconds
    timeout = 12.0

    async def search(self, client, query, limit, since):
        terms = re.sub(r'[":()\[\]]', " ", query).split()
        if not terms:
            return []
        response = await self._get(client, "https://export.arxiv.org/api/query", {
            "search_query": "all:" + " ".join(terms[:12]), "max_results": limit, "sortBy": "relevance",
        })
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            raise SourceError("arxiv: sent an unreadable answer.") from exc
        hits = []
        for entry in root.findall("a:entry", _ATOM):
            link = re.sub(r"v\d+$", "", (entry.findtext("a:id", "", _ATOM) or "").replace("http://", "https://"))
            if not link:
                continue
            summary = " ".join((entry.findtext("a:summary", "", _ATOM) or "").split())
            authors = [a.findtext("a:name", "", _ATOM) for a in entry.findall("a:author", _ATOM)]
            hits.append(Hit(
                " ".join((entry.findtext("a:title", "", _ATOM) or "").split()), link, _clip(summary), self.name,
                _iso_date(entry.findtext("a:published", "", _ATOM)), summary or None,
                {"authors": authors[:6], "pdf_url": link.replace("/abs/", "/pdf/")},
            ))
        return hits


class EuropePMC(Source):
    """Life-science and medical literature, including everything in PubMed."""

    name = "europepmc"
    categories = ("papers",)
    min_interval = 0.2

    async def search(self, client, query, limit, since):
        if since:
            query = f"({query}) AND FIRST_PDATE:[{since.isoformat()} TO {date.today().isoformat()}]"
        data = await self._json(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search", {
            "query": query, "format": "json", "pageSize": limit, "resultType": "core",
        })
        hits = []
        for r in (data.get("resultList") or {}).get("result", []):
            if not r.get("id") or not r.get("source"):
                continue
            abstract = _html_to_text(r.get("abstractText", ""))
            extra = {"authors": r.get("authorString", ""), "open_access": r.get("isOpenAccess") == "Y"}
            if r.get("doi"):
                extra["doi"] = r["doi"]
            hits.append(Hit(
                unescape(r.get("title", "")).rstrip("."), f"https://europepmc.org/article/{r['source']}/{r['id']}",
                _clip(abstract), self.name, _iso_date(r.get("firstPublicationDate")), abstract or None, extra,
            ))
        return hits


class OpenAlex(Source):
    name = "openalex"
    categories = ("papers",)
    min_interval = 1.0

    async def search(self, client, query, limit, since):
        params = {"search": query, "per-page": limit}
        if since:
            params["filter"] = f"from_publication_date:{since.isoformat()}"
        if _env("WEB_EXPRESS_OPENALEX_KEY"):
            params["api_key"] = _env("WEB_EXPRESS_OPENALEX_KEY")
        if "@" in _env("WEB_EXPRESS_CONTACT"):
            params["mailto"] = _env("WEB_EXPRESS_CONTACT")
        data = await self._json(client, "https://api.openalex.org/works", params)
        hits = []
        for work in data.get("results", []):
            location = work.get("primary_location") or {}
            oa_url = (work.get("open_access") or {}).get("oa_url")
            url = oa_url or location.get("landing_page_url") or work.get("doi") or work.get("id")
            if not url or not work.get("title"):
                continue
            words: dict[int, str] = {}
            for word, positions in (work.get("abstract_inverted_index") or {}).items():
                for position in positions:
                    words[position] = word
            abstract = " ".join(words[i] for i in sorted(words))
            extra = {"cited_by": work.get("cited_by_count", 0)}
            if work.get("doi"):
                extra["doi"] = work["doi"].removeprefix("https://doi.org/")
            hits.append(Hit(work["title"], url, _clip(abstract), self.name, _iso_date(work.get("publication_date")),
                            abstract or None, extra))
        return hits


# --- code --------------------------------------------------------------------

class GitHub(Source):
    name = "github"
    categories = ("code",)
    min_interval = 2.0  # the keyless search limit is ten requests a minute

    async def search(self, client, query, limit, since):
        headers = {"Accept": "application/vnd.github+json"}
        if _env("WEB_EXPRESS_GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {_env('WEB_EXPRESS_GITHUB_TOKEN')}"
        pushed = f" pushed:>{since.isoformat()}" if since else ""
        data = await self._json(client, "https://api.github.com/search/repositories",
                                {"q": query + pushed, "per_page": limit}, headers)
        hits = []
        for repo in data.get("items", []):
            description = repo.get("description") or ""
            hits.append(Hit(
                repo.get("full_name", ""), repo.get("html_url", ""), _clip(description), self.name,
                _iso_date(repo.get("pushed_at")), None,
                {"stars": repo.get("stargazers_count", 0), "language": repo.get("language"),
                 "license": (repo.get("license") or {}).get("spdx_id")},
            ))
        return [h for h in hits if h.url]


class StackOverflow(Source):
    """Questions with their top answer. Content is CC BY-SA; the link is the attribution."""

    name = "stackoverflow"
    categories = ("code",)
    min_interval = 0.5
    api = "https://api.stackexchange.com/2.3"

    async def search(self, client, query, limit, since):
        params = {"q": query, "site": "stackoverflow", "pagesize": min(limit, 8), "order": "desc",
                  "sort": "relevance", "filter": "withbody"}
        if since:
            params["fromdate"] = int(datetime(since.year, since.month, since.day, tzinfo=timezone.utc).timestamp())
        data = await self._json(client, f"{self.api}/search/advanced", params)
        questions = [q for q in data.get("items", []) if q.get("question_id") and q.get("link")]
        answers: dict[int, dict] = {}
        answered = [str(q["question_id"]) for q in questions if q.get("is_answered")]
        if answered and not data.get("backoff"):
            try:
                found = await self._json(client, f"{self.api}/questions/{';'.join(answered)}/answers", {
                    "site": "stackoverflow", "order": "desc", "sort": "votes", "filter": "withbody", "pagesize": 30,
                })
            except SourceError:
                found = {}
            for answer in found.get("items", []):  # sorted by votes: keep the first per question
                answers.setdefault(answer.get("question_id"), answer)
        hits = []
        for q in questions:
            question = _html_to_text(q.get("body", ""))
            content = "## Question\n\n" + question
            answer = answers.get(q["question_id"])
            if answer:
                label = "Accepted answer" if answer.get("is_accepted") else "Top answer"
                content += f"\n\n## {label} (score {answer.get('score', 0)})\n\n" + _html_to_text(answer.get("body", ""))
            hits.append(Hit(
                unescape(q.get("title", "")), q["link"], _clip(question), self.name, _iso_date(q.get("creation_date")),
                content, {"score": q.get("score", 0), "answered": bool(q.get("is_answered"))},
            ))
        return hits


# --- news and discussion -----------------------------------------------------

class HackerNews(Source):
    name = "hackernews"
    categories = ("news", "code")

    async def search(self, client, query, limit, since):
        params = {"query": query, "tags": "story", "hitsPerPage": limit}
        if since:
            stamp = int(datetime(since.year, since.month, since.day, tzinfo=timezone.utc).timestamp())
            params["numericFilters"] = f"created_at_i>{stamp}"
        data = await self._json(client, "https://hn.algolia.com/api/v1/search", params)
        hits = []
        for story in data.get("hits", []):
            discussion = f"https://news.ycombinator.com/item?id={story.get('objectID')}"
            text = _html_to_text(story.get("story_text") or "")
            hits.append(Hit(
                story.get("title") or "", story.get("url") or discussion, _clip(text), self.name,
                _iso_date(story.get("created_at")), text or None,
                {"points": story.get("points", 0), "discussion": discussion},
            ))
        return [h for h in hits if h.title]


class Gdelt(Source):
    """Worldwide news index. Strict: one request every five seconds."""

    name = "gdelt"
    categories = ("news",)
    min_interval = 5.5
    timeout = 6.0  # it is often slow; do not let it hold up the other sources

    async def search(self, client, query, limit, since):
        days = max(1, (date.today() - since).days) if since else 30
        response = await self._get(client, "https://api.gdeltproject.org/api/v2/doc/doc", {
            "query": f"{query} sourcelang:english", "mode": "artlist", "format": "json", "maxrecords": limit,
            "sort": "hybridrel",
            "timespan": f"{days}d",
        })
        try:
            data = response.json()
        except ValueError as exc:  # GDELT explains bad queries in plain text
            raise SourceError(f"gdelt: {_clip(response.text, 120)}") from exc
        hits = []
        for article in data.get("articles", []) if isinstance(data, dict) else []:
            seen = article.get("seendate", "")
            published = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 and seen[:8].isdigit() else None
            if article.get("url") and article.get("title"):
                hits.append(Hit(article["title"], article["url"], "", self.name, published, None,
                                {"domain": article.get("domain"), "language": article.get("language")}))
        return hits


def default_sources() -> list[Source]:
    return [DuckDuckGo(), Wikipedia(), Arxiv(), EuropePMC(), OpenAlex(), GitHub(), StackOverflow(), HackerNews(), Gdelt()]
