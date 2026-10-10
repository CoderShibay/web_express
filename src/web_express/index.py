"""A private index of every page web_express has read, kept on this computer.

It makes repeat topics fast, works offline, and never leaves the machine. One
SQLite file holds page details, the page text in a full-text table, and one
vector per page for search by meaning.

    WEB_EXPRESS_INDEX       "off" disables it
    WEB_EXPRESS_MAX_PAGES   pages kept before the oldest are dropped (default 20000)

What is deliberately not stored: pages whose address carries a password, token
or signature, and pages that look like login, consent or bot-check screens.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from .embed import data_home
from .extract import Document
from .passages import _STOPWORDS

MAX_TEXT_CHARS = 300_000  # per page
MIN_TEXT_CHARS = 200  # shorter pages are usually error screens
MIN_SIMILARITY = 0.35  # a page this close to the query matches by meaning
KEYWORD_FLOOR = 0.20  # a page sharing words with the query but further than this in meaning is dropped
SCHEMA_VERSION = 2
TIDY_EVERY = 200  # writes between housekeeping passes
_TERM = re.compile(r"[^\W_]+", re.UNICODE)
_SENSITIVE = re.compile(
    r"token|secret|passw|pwd|signature|(^|[_-])sig($|[_-])|api[_-]?key|(^|[_-])key($|[_-])|auth|session|"
    r"(^|[_-])sid($|[_-])|credential|jwt|(^|[_-])code($|[_-])|x-amz-|otp|ticket",
    re.IGNORECASE,
)
_GATE = re.compile(
    r"just a moment|checking your browser|verify (that )?you are (a )?human|are you a robot|captcha|"
    r"enable javascript|access denied|403 forbidden|404 not found|page not found|sign in to continue|"
    r"log ?in to continue|please log ?in|accept (all )?cookies to continue|unusual traffic",
    re.IGNORECASE,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,          -- the exact address, normalised (see exact_key)
    url TEXT NOT NULL,
    published TEXT,
    fetched_at REAL NOT NULL,
    content_hash TEXT NOT NULL,
    chars INTEGER NOT NULL,
    extractor TEXT NOT NULL,
    model TEXT,                        -- embedding model that produced vector
    dims INTEGER,                      -- length of vector
    vector BLOB
);
CREATE INDEX IF NOT EXISTS pages_fetched ON pages(fetched_at);
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(title, text, tokenize = 'unicode61 remove_diacritics 2');
"""


@dataclass
class IndexedPage:
    url: str
    title: str
    text: str
    published: str | None
    fetched_at: float
    extractor: str

    def document(self, stale: bool = False) -> Document:
        return Document(self.url, self.title, self.text, self.published, self.extractor, False,
                        fetched_at=self.fetched_at, stale=stale)


def default_path() -> Path:
    return data_home() / "index.sqlite3"


def enabled() -> bool:
    return os.environ.get("WEB_EXPRESS_INDEX", "").strip().lower() != "off"


def _max_pages() -> int:
    try:
        return max(100, int(os.environ.get("WEB_EXPRESS_MAX_PAGES", "20000")))
    except ValueError:
        return 20000


def exact_key(url: str) -> str:
    """The address a page is stored under: the URL itself, minus only what cannot change the page
    (letter case of scheme and host, a default port, the #fragment). Two different pages never share a key."""
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
        if ":" in host:
            host = f"[{host}]"
        default = {"http": 80, "https": 443}.get(parts.scheme.lower())
        port = f":{parts.port}" if parts.port and parts.port != default else ""
        return urlunsplit((parts.scheme.lower(), host + port, parts.path or "/", parts.query, ""))
    except ValueError:
        return url.strip()


def storable(url: str) -> bool:
    """False for addresses that carry credentials: those must never be written to disk."""
    try:
        parts = urlsplit(url)
        if parts.username or parts.password or parts.scheme not in ("http", "https"):
            return False
        names = [name for name, _ in parse_qsl(parts.query, keep_blank_values=True)]
    except ValueError:
        return False
    return not any(_SENSITIVE.search(name) for name in names)


def worth_keeping(document: Document) -> bool:
    """False for pages that are not real content: too short, or a login, consent or bot-check screen."""
    text = document.text.strip()
    if len(text) < MIN_TEXT_CHARS:
        return False
    if len(text) < 2000 and (_GATE.search(document.title or "") or _GATE.search(text[:600])):
        return False
    return True


def _match_expression(query: str) -> str:
    """An FTS5 query matching any content word of query; words are quoted, so no operator can be injected."""
    terms = [t for t in dict.fromkeys(t.lower() for t in _TERM.findall(query)) if t not in _STOPWORDS][:24]
    return " OR ".join(f'"{t}"' for t in terms)


def _clean_vector(vector):
    """The vector as float32 bytes plus its length, or (None, None) if it is not a usable vector."""
    if vector is None:
        return None, None
    try:
        import numpy as np

        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not bool(np.isfinite(array).all()):
            return None, None
        return array.tobytes(), int(array.size)
    except (TypeError, ValueError):
        return None, None


class LocalIndex:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._vectors = None  # (model, dims, data_version, ids, matrix)
        self._writes = 0
        try:
            self._open()
        except sqlite3.DatabaseError:
            # Not a database, or damaged beyond opening: set it aside and start again.
            self.path.replace(self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}"))
            for suffix in ("-wal", "-shm"):
                self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
            self._open()

    def _open(self) -> None:
        new = not self.path.exists() or self.path.stat().st_size == 0
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        try:
            with self._lock:
                if new:
                    self._db.execute("PRAGMA auto_vacuum = INCREMENTAL")  # must precede the first table
                self._db.execute("PRAGMA journal_mode = WAL")
                version = self._db.execute("PRAGMA user_version").fetchone()[0]
                with self._db:
                    self._db.executescript(_SCHEMA)
                    if version == 1:  # version 1 had no dims column and keyed pages by a lossy address
                        self._db.execute("ALTER TABLE pages ADD COLUMN dims INTEGER")
                        self._db.execute("UPDATE pages SET dims = length(vector) / 4 WHERE vector IS NOT NULL")
                        self._db.execute("UPDATE OR IGNORE pages SET key = url")
                    self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except sqlite3.DatabaseError:
            self._db.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- writing ---------------------------------------------------------

    def add(self, document: Document, vector=None, model: str | None = None) -> bool:
        """Store a page, replacing any earlier copy of the same address. Returns whether it was stored."""
        text = document.text[:MAX_TEXT_CHARS]
        if not text.strip() or not storable(document.url):
            return False
        key = exact_key(document.url)
        blob, dims = _clean_vector(vector) if model else (None, None)
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        with self._lock, self._db:
            old = self._db.execute("SELECT id FROM pages WHERE key = ?", (key,)).fetchone()
            if old:
                self._db.execute("DELETE FROM pages_fts WHERE rowid = ?", (old["id"],))
                self._db.execute("DELETE FROM pages WHERE id = ?", (old["id"],))
            cursor = self._db.execute(
                "INSERT INTO pages (key, url, published, fetched_at, content_hash, chars, extractor, model, dims, vector) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, document.url, document.published, time.time(), digest, len(text), document.extractor,
                 model if blob else None, dims, blob),
            )
            self._db.execute(
                "INSERT INTO pages_fts (rowid, title, text) VALUES (?, ?, ?)", (cursor.lastrowid, document.title, text)
            )
            self._prune()
            self._vectors = None
        self._writes += 1
        if self._writes % TIDY_EVERY == 0:
            self.tidy()
        return True

    def _prune(self) -> None:
        extra = self._db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] - _max_pages()
        if extra > 0:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM pages ORDER BY fetched_at LIMIT ?", (extra,))]
            marks = ",".join("?" * len(ids))
            self._db.execute(f"DELETE FROM pages_fts WHERE rowid IN ({marks})", ids)
            self._db.execute(f"DELETE FROM pages WHERE id IN ({marks})", ids)

    def tidy(self) -> None:
        """Merge the full-text index and hand freed space back to the disk."""
        with self._lock:
            with self._db:
                self._db.execute("INSERT INTO pages_fts(pages_fts) VALUES ('optimize')")
            self._db.execute("PRAGMA incremental_vacuum")
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def remove(self, url: str) -> bool:
        with self._lock, self._db:
            row = self._db.execute("SELECT id FROM pages WHERE key = ?", (exact_key(url),)).fetchone()
            if not row:
                return False
            self._db.execute("DELETE FROM pages_fts WHERE rowid = ?", (row["id"],))
            self._db.execute("DELETE FROM pages WHERE id = ?", (row["id"],))
            self._vectors = None
            return True

    def clear(self) -> None:
        with self._lock:
            with self._db:
                self._db.execute("DELETE FROM pages_fts")
                self._db.execute("DELETE FROM pages")
                self._vectors = None
        self.tidy()

    # --- reading ---------------------------------------------------------

    def _page(self, row) -> IndexedPage:
        return IndexedPage(row["url"], row["title"], row["text"], row["published"], row["fetched_at"], row["extractor"])

    _SELECT = (
        "SELECT p.id, p.url, p.published, p.fetched_at, p.extractor, f.title, f.text "
        "FROM pages p JOIN pages_fts f ON f.rowid = p.id "
    )

    def get(self, url: str, max_age: float | None = None) -> IndexedPage | None:
        """The stored copy of exactly this address, if there is one no older than max_age seconds."""
        with self._lock:
            row = self._db.execute(self._SELECT + "WHERE p.key = ?", (exact_key(url),)).fetchone()
        if row is None or (max_age is not None and time.time() - row["fetched_at"] > max_age):
            return None
        return self._page(row)

    def _vector_table(self, model: str, dims: int):
        """(ids, matrix) of the stored vectors for this model and length; rebuilt when the data changes,
        including changes made by another process."""
        import numpy as np

        version = self._db.execute("PRAGMA data_version").fetchone()[0]
        if self._vectors is None or self._vectors[:3] != (model, dims, version):
            ids, rows = [], []
            for row in self._db.execute(
                "SELECT id, vector FROM pages WHERE model = ? AND dims = ? AND vector IS NOT NULL", (model, dims)
            ):
                if len(row["vector"]) != dims * 4:
                    continue  # a damaged row: leave it out rather than misalign every other vector
                vector = np.frombuffer(row["vector"], dtype=np.float32)
                if np.isfinite(vector).all():
                    ids.append(row["id"])
                    rows.append(vector)
            self._vectors = (model, dims, version, ids, np.vstack(rows) if rows else None)
        return self._vectors[3], self._vectors[4]

    def search(self, query: str, limit: int = 10, query_vector=None, model: str | None = None,
               newer_than: float | None = None) -> list[IndexedPage]:
        """Pages matching query by words and, when a query vector is given, by meaning; best first."""
        scores: dict[int, float] = {}
        with self._lock:
            closeness: dict[int, float] = {}
            blob, dims = _clean_vector(query_vector) if model else (None, None)
            if blob is not None:
                import numpy as np

                wanted = np.frombuffer(blob, dtype=np.float32)
                ids, matrix = self._vector_table(model, dims)
                if matrix is not None:
                    similarity = matrix @ wanted
                    closeness = {i: s for i, s in zip(ids, similarity.tolist()) if math.isfinite(s)}
                    best = sorted(closeness, key=lambda i: -closeness[i])[: limit * 3]
                    for rank, page_id in enumerate(best):
                        if closeness[page_id] >= MIN_SIMILARITY:  # not merely the least unrelated page
                            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (60 + rank + 1)
            expression = _match_expression(query)
            if expression:
                rows = self._db.execute(
                    "SELECT rowid FROM pages_fts WHERE pages_fts MATCH ? ORDER BY bm25(pages_fts, 4.0, 1.0) LIMIT ?",
                    (expression, limit * 3),
                ).fetchall()
                for rank, row in enumerate(rows):
                    if closeness.get(row["rowid"], 1.0) < KEYWORD_FLOOR:
                        continue  # shares a word with the query but is about something else
                    scores[row["rowid"]] = scores.get(row["rowid"], 0.0) + 1.0 / (60 + rank + 1)
            if not scores:
                return []
            ordered = sorted(scores, key=lambda i: -scores[i])
            marks = ",".join("?" * len(ordered))
            found = {r["id"]: r for r in self._db.execute(self._SELECT + f"WHERE p.id IN ({marks})", ordered)}
        pages = [self._page(found[i]) for i in ordered if i in found]
        if newer_than is not None:
            pages = [p for p in pages if p.fetched_at >= newer_than]
        return pages[:limit]

    def stats(self) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS pages, COALESCE(SUM(chars), 0) AS chars, COUNT(vector) AS with_vectors, "
                "MIN(fetched_at) AS oldest, MAX(fetched_at) AS newest FROM pages"
            ).fetchone()
        size = 0
        for suffix in ("", "-wal", "-shm"):  # SQLite keeps recent writes in files beside the main one
            part = self.path.with_name(self.path.name + suffix)
            size += part.stat().st_size if part.exists() else 0
        return {"path": str(self.path), "pages": row["pages"], "characters": row["chars"],
                "pages_with_vectors": row["with_vectors"], "oldest": row["oldest"], "newest": row["newest"],
                "file_bytes": size}
