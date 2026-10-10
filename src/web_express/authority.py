"""How well-linked a website is, from Common Crawl's open web graph.

Common Crawl publishes, for every release, a list of all domains it saw ordered
by harmonic centrality: roughly, how close a domain is to the rest of the web
by links. The list is sorted best first, so the top of it can be streamed
without downloading the whole 2.5 GB file.

`web-express authority install` stores the top domains (one million by default,
about a 30 MB download) in a small SQLite file. Search then gives a slight
preference to well-linked sites and reports each result's rank. It is an
optional extra: without it nothing changes.

Data: https://commoncrawl.org/web-graphs (Common Crawl terms of use apply).
"""

from __future__ import annotations

import math
import sqlite3
import threading
import zlib
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .embed import data_home

RELEASES = "https://index.commoncrawl.org/graphinfo.json"
RANKS = "https://data.commoncrawl.org/projects/hyperlinkgraph/{id}/domain/{id}-domain-ranks.txt.gz"
DEFAULT_TOP = 1_000_000
MAX_BONUS = 0.05  # added to a result's similarity for the best-linked sites


def default_path() -> Path:
    return data_home() / "authority.sqlite3"


def latest_release(client: httpx.Client) -> str:
    releases = client.get(RELEASES).raise_for_status().json()
    if not isinstance(releases, list) or not releases or not isinstance(releases[0], dict):
        raise RuntimeError("Common Crawl's release list has an unexpected layout.")
    release = releases[0].get("id")
    if not isinstance(release, str) or not release.replace("-", "").isalnum():
        raise RuntimeError("Common Crawl's release list has an unexpected layout.")
    return release


def _lines(response: httpx.Response):
    """Lines of a gzip-compressed text stream, decoded as they arrive."""
    decoder = zlib.decompressobj(47)
    pending = b""
    for raw in response.iter_raw():
        pending += decoder.decompress(raw)
        *lines, pending = pending.split(b"\n")
        yield from lines
        if len(pending) > 1_000_000:
            raise RuntimeError("The ranks file has an unexpected layout.")
    if pending:
        yield pending


def parse_line(line: bytes) -> tuple[str, int] | None:
    """(domain, position) from one line of the ranks file; None for headers and bad lines."""
    if line.startswith(b"#"):
        return None
    fields = line.split(b"\t")
    try:
        position = int(fields[0])
        reversed_name = fields[4].decode("ascii").strip().lower()
    except (IndexError, ValueError, UnicodeError):
        return None
    if position < 1 or not reversed_name or len(reversed_name) > 253:
        return None
    return ".".join(reversed(reversed_name.split("."))), position


def install(top: int = DEFAULT_TOP, path: Path | None = None, transport: httpx.BaseTransport | None = None,
            progress=None) -> dict:
    """Download the best-linked `top` domains of the latest release and store them."""
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.unlink(missing_ok=True)
    count = 0
    try:
        with httpx.Client(follow_redirects=True, timeout=60, transport=transport) as client:
            release = latest_release(client)
            db = sqlite3.connect(str(partial))
            try:
                db.execute("CREATE TABLE domains (domain TEXT PRIMARY KEY, position INTEGER NOT NULL) WITHOUT ROWID")
                db.execute("CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                with client.stream("GET", RANKS.format(id=release)) as response:
                    response.raise_for_status()
                    batch = []
                    for line in _lines(response):
                        row = parse_line(line)
                        if row is None:
                            continue
                        batch.append(row)
                        count += 1
                        if len(batch) >= 20_000:
                            db.executemany("INSERT OR IGNORE INTO domains VALUES (?, ?)", batch)
                            batch.clear()
                            if progress:
                                progress(count)
                        if count >= top:
                            break  # the file is sorted best first: stop reading here
                    db.executemany("INSERT OR IGNORE INTO domains VALUES (?, ?)", batch)
                if count == 0:
                    raise RuntimeError("The ranks file held no usable rows.")
                db.executemany("INSERT INTO info VALUES (?, ?)", [("release", release), ("domains", str(count))])
                db.commit()
            finally:
                db.close()
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return {"release": release, "domains": count, "path": str(path)}


class Authority:
    """Looks up a site's position in the ranking. Missing data means every lookup returns None."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_path()
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._cache: dict[str, int | None] = {}
        if self.path.is_file():
            try:
                self._db = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
                self._db.execute("SELECT position FROM domains LIMIT 1").fetchall()
            except sqlite3.Error:
                self._db = None

    @property
    def available(self) -> bool:
        return self._db is not None

    def info(self) -> dict:
        if self._db is None:
            return {}
        with self._lock:
            return dict(self._db.execute("SELECT key, value FROM info").fetchall())

    def position(self, url: str) -> int | None:
        """Rank of the URL's site (1 is the best-linked), or None if unknown."""
        if self._db is None:
            return None
        try:
            host = (urlsplit(url).hostname or "").lower().rstrip(".")
        except ValueError:
            return None
        if not host:
            return None
        if host in self._cache:
            return self._cache[host]
        labels = host.split(".")
        # The list holds registrable domains: try the host, then each shorter suffix.
        candidates = [".".join(labels[i:]) for i in range(len(labels) - 1)]
        position = None
        try:
            with self._lock:
                for candidate in candidates:
                    row = self._db.execute("SELECT position FROM domains WHERE domain = ?", (candidate,)).fetchone()
                    if row:
                        position = row[0]
                        break
        except sqlite3.Error:
            return None
        if len(self._cache) > 50_000:
            self._cache.clear()
        self._cache[host] = position
        return position

    def bonus(self, url: str) -> float:
        """A small ranking bonus: MAX_BONUS for the very top sites, fading to 0 at rank one million."""
        position = self.position(url)
        if position is None:
            return 0.0
        return MAX_BONUS * max(0.0, 1.0 - math.log10(position) / 6.0)
