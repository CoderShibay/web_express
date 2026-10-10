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


def _text(value) -> str:
    """A string from a field that should be one; anything else becomes empty."""
    return value if isinstance(value, str) else ""


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _literal(query: str, drop: str = '()[]{}:') -> str:
    """The query as plain words: no field qualifiers, grouping or boolean operators.

    Sources have their own query languages. Mandatory filters (dates, language)
    are appended to the user's words, so those words must not be able to rewrite them.
    """
    cleaned = query.translate({ord(c): " " for c in drop})
    if cleaned.count('"') % 2:
        cleaned = cleaned.replace('"', " ")
    words = [w.lower() if w in ("AND", "OR", "NOT") else w for w in cleaned.split()]
    return " ".join(words)


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Source:
    name = ""
    categories: tuple[str, ...] = ()
    min_interval = 0.0  # seconds between requests to this source
    timeout = 8.0  # for one request, not counting the wait for our own rate limit

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last = 0.0

    def available(self) -> bool:
        return True

    def _rows(self, value) -> list[dict]:
        """The list of result rows from an API answer; a different shape is a source error."""
        if not isinstance(value, list):
            raise SourceError(f"{self.name}: sent an unexpected answer.")
        return [row for row in value if isinstance(row, dict)]

    def _collect(self, rows: list[dict], build) -> list[Hit]:
        """Build a Hit from each row, skipping rows whose fields are not what the API promises."""
        hits = []
        for row in rows:
            try:
                hit = build(row)
            except (TypeError, AttributeError, KeyError, ValueError, IndexError):
                continue
            if hit is not None:
                hits.append(hit)
        return hits

    async def search(self, client: httpx.AsyncClient, query: str, limit: int, since: date | None) -> list[Hit]:
        raise NotImplementedError

    async def _get(self, client: httpx.AsyncClient, url: str, params: dict, headers: dict | None = None) -> httpx.Response:
        async with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            try:  # the deadline covers the request only, not the wait for our own rate limit
                response = await asyncio.wait_for(client.get(url, params=params, headers=headers), self.timeout)
            except TimeoutError as exc:
                raise SourceError(f"{self.name}: took too long to answer.") from exc
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
        pages = self._rows(list(_dict(_dict(data.get("query")).get("pages")).values()))
        pages.sort(key=lambda p: p.get("index") if isinstance(p.get("index"), int) else 0)

        def build(p):
            extract = _text(p.get("extract"))
            return Hit(_text(p.get("title")), p["fullurl"], _clip(extract), self.name, _iso_date(p.get("touched")),
                       extract or None)

        return self._collect(pages, build)


# --- papers ------------------------------------------------------------------

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


class Arxiv(Source):
    name = "arxiv"
    categories = ("papers",)
    min_interval = 3.0  # arXiv asks for one request every three seconds
    timeout = 12.0

    async def search(self, client, query, limit, since):
        terms = _literal(query, drop='()[]{}:"').split()
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
        query = _literal(query)
        if not query:
            return []
        if since:
            query = f"({query}) AND FIRST_PDATE:[{since.isoformat()} TO {date.today().isoformat()}]"
        data = await self._json(client, "https://www.ebi.ac.uk/europepmc/webservices/rest/search", {
            "query": query, "format": "json", "pageSize": limit, "resultType": "core",
        })

        def build(r):
            source, number = _text(r.get("source")), _text(r.get("id"))
            if not re.fullmatch(r"[A-Za-z]+", source) or not re.fullmatch(r"[\w.-]+", number):
                return None
            abstract = _html_to_text(_text(r.get("abstractText")))
            extra = {"authors": _text(r.get("authorString")), "open_access": r.get("isOpenAccess") == "Y"}
            if _text(r.get("doi")):
                extra["doi"] = r["doi"]
            return Hit(
                unescape(_text(r.get("title"))).rstrip("."), f"https://europepmc.org/article/{source}/{number}",
                _clip(abstract), self.name, _iso_date(r.get("firstPublicationDate")), abstract or None, extra,
            )

        return self._collect(self._rows(_dict(data.get("resultList")).get("result", [])), build)


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

        def build(work):
            title = _text(work.get("title"))
            doi = _text(work.get("doi"))
            url = (_text(_dict(work.get("open_access")).get("oa_url"))
                   or _text(_dict(work.get("primary_location")).get("landing_page_url")) or doi or _text(work.get("id")))
            if not url or not title:
                return None
            words: dict[int, str] = {}
            for word, positions in _dict(work.get("abstract_inverted_index")).items():
                for position in positions:
                    if isinstance(position, int):
                        words[position] = word
            abstract = " ".join(words[i] for i in sorted(words))
            extra: dict = {"cited_by": work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else 0}
            if doi:
                extra["doi"] = doi.removeprefix("https://doi.org/")
            return Hit(title, url, _clip(abstract), self.name, _iso_date(work.get("publication_date")), abstract or None, extra)

        return self._collect(self._rows(data.get("results")), build)


# --- code --------------------------------------------------------------------

class GitHub(Source):
    name = "github"
    categories = ("code",)
    min_interval = 2.0  # the keyless search limit is ten requests a minute

    async def search(self, client, query, limit, since):
        headers = {"Accept": "application/vnd.github+json"}
        if _env("WEB_EXPRESS_GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {_env('WEB_EXPRESS_GITHUB_TOKEN')}"
        query = _literal(query)
        if not query:
            return []
        pushed = f" pushed:>{since.isoformat()}" if since else ""
        data = await self._json(client, "https://api.github.com/search/repositories",
                                {"q": query + pushed, "per_page": limit}, headers)

        def build(repo):
            if not _text(repo.get("html_url")) or not _text(repo.get("full_name")):
                return None
            return Hit(
                repo["full_name"], repo["html_url"], _clip(_text(repo.get("description"))), self.name,
                _iso_date(repo.get("pushed_at")), None,
                {"stars": repo.get("stargazers_count") if isinstance(repo.get("stargazers_count"), int) else 0,
                 "language": _text(repo.get("language")) or None,
                 "license": _text(_dict(repo.get("license")).get("spdx_id")) or None},
            )

        return self._collect(self._rows(data.get("items")), build)


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
        questions = [q for q in self._rows(data.get("items"))
                     if isinstance(q.get("question_id"), int) and _text(q.get("link"))]
        answers: dict[int, dict] = {}
        answered = [str(q["question_id"]) for q in questions if q.get("is_answered")]
        if answered and not data.get("backoff"):
            try:
                found = await self._json(client, f"{self.api}/questions/{';'.join(answered)}/answers", {
                    "site": "stackoverflow", "order": "desc", "sort": "votes", "filter": "withbody", "pagesize": 30,
                })
                rows = self._rows(found.get("items"))
            except SourceError:
                rows = []
            for answer in rows:  # sorted by votes: keep the first per question
                answers.setdefault(answer.get("question_id"), answer)

        def build(q):
            question = _html_to_text(_text(q.get("body")))
            content = "## Question\n\n" + question
            answer = answers.get(q["question_id"])
            if answer:
                label = "Accepted answer" if answer.get("is_accepted") else "Top answer"
                score = answer.get("score") if isinstance(answer.get("score"), int) else 0
                content += f"\n\n## {label} (score {score})\n\n" + _html_to_text(_text(answer.get("body")))
            return Hit(
                unescape(_text(q.get("title"))), q["link"], _clip(question), self.name, _iso_date(q.get("creation_date")),
                content, {"score": q.get("score") if isinstance(q.get("score"), int) else 0,
                          "answered": bool(q.get("is_answered"))},
            )

        return self._collect(questions, build)


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

        def build(story):
            title = _text(story.get("title"))
            if not title:
                return None
            discussion = f"https://news.ycombinator.com/item?id={story['objectID']}"
            text = _html_to_text(_text(story.get("story_text")))
            return Hit(
                title, _text(story.get("url")) or discussion, _clip(text), self.name, _iso_date(story.get("created_at")),
                text or None,
                {"points": story.get("points") if isinstance(story.get("points"), int) else 0, "discussion": discussion},
            )

        return self._collect(self._rows(data.get("hits")), build)


class Gdelt(Source):
    """Worldwide news index. Strict: one request every five seconds."""

    name = "gdelt"
    categories = ("news",)
    min_interval = 5.5
    timeout = 6.0  # it is often slow; do not let it hold up the other sources

    async def search(self, client, query, limit, since):
        query = _literal(query)
        if not query:
            return []
        days = max(1, (date.today() - since).days) if since else 30
        response = await self._get(client, "https://api.gdeltproject.org/api/v2/doc/doc", {
            "query": f"{query} sourcelang:english", "mode": "artlist", "format": "json", "maxrecords": limit,
            "sort": "hybridrel", "timespan": f"{days}d",
        })
        try:
            data = response.json()
        except ValueError as exc:  # GDELT explains bad queries in plain text
            raise SourceError(f"gdelt: {_clip(response.text, 120)}") from exc

        def build(article):
            if not _text(article.get("url")) or not _text(article.get("title")):
                return None
            seen = _text(article.get("seendate"))
            published = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 and seen[:8].isdigit() else None
            return Hit(article["title"], article["url"], "", self.name, published, None,
                       {"domain": _text(article.get("domain")) or None, "language": _text(article.get("language")) or None})

        return self._collect(self._rows(_dict(data).get("articles", [])), build)


def default_sources() -> list[Source]:
    return [DuckDuckGo(), Wikipedia(), Arxiv(), EuropePMC(), OpenAlex(), GitHub(), StackOverflow(), HackerNews(), Gdelt()]
