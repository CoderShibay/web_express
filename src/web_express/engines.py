"""Search engines web_express can ask for candidate results.

Only DuckDuckGo's HTML page for now. Each engine returns plain SearchHit rows so
that more engines can be added and their results merged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import httpx

from .net import HEADERS

TIME_RANGES = {"day": "d", "week": "w", "month": "m", "year": "y"}


class EngineError(Exception):
    """A search engine could not answer; the message is safe to show."""


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


def clean_result_url(href: str) -> str:
    """DuckDuckGo sometimes wraps the target in a /l/?uddg= redirect."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


class _DuckDuckGoParser(HTMLParser):
    """Pull title, url and snippet out of DuckDuckGo's HTML results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self.no_results = False
        self._field: str | None = None
        self._current: dict | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if "no-results" in classes:
            self.no_results = True
        if tag != "a":
            return
        if "result__a" in classes:
            self._current = {"title": "", "url": clean_result_url(attr.get("href") or ""), "snippet": ""}
            self.results.append(self._current)
            self._field = "title"
        elif "result__snippet" in classes and self._current is not None:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self._field = None

    def handle_data(self, data):
        if self._field and self._current is not None:
            self._current[self._field] += data


def parse_duckduckgo(html: str) -> list[SearchHit]:
    """Parse a DuckDuckGo HTML results page; raise EngineError if it is not one."""
    parser = _DuckDuckGoParser()
    parser.feed(html)
    hits = [
        SearchHit(" ".join(r["title"].split()), r["url"], " ".join(r["snippet"].split()))
        for r in parser.results
        if r["url"].startswith("http") and "duckduckgo.com/y.js" not in r["url"]
    ]
    if hits or parser.no_results:
        return hits
    if "anomaly" in html.lower():
        raise EngineError("DuckDuckGo is asking for a human check (too many requests). Wait a few minutes and retry.")
    raise EngineError("DuckDuckGo returned a page web_express could not read; its layout may have changed.")


class DuckDuckGo:
    name = "duckduckgo"
    url = "https://html.duckduckgo.com/html/"
    min_interval = 1.5  # seconds between searches; DuckDuckGo blocks bursts

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def search(self, query: str, freshness: str = "", region: str = "") -> list[SearchHit]:
        form = {"q": query}
        if freshness:
            form["df"] = TIME_RANGES[freshness]
        if region:
            form["kl"] = region
        async with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
                    response = await client.post(self.url, data=form)
            except httpx.HTTPError as exc:
                raise EngineError(f"Could not reach DuckDuckGo ({type(exc).__name__}).") from exc
            finally:
                self._last = time.monotonic()
        if response.status_code != 200:
            raise EngineError(
                f"DuckDuckGo answered HTTP {response.status_code}. It is probably rate-limiting; wait a minute and retry."
            )
        return parse_duckduckgo(response.text)
