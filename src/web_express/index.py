"""A private index of every page web_express has read, kept on this computer.

It makes repeat topics fast, works offline, and never leaves the machine. One
SQLite file holds page details, the page text in a full-text table, and one
vector per page for search by meaning.

    WEB_EXPRESS_INDEX       "off" disables it
    WEB_EXPRESS_MAX_PAGES   pages kept before the oldest are dropped (default 20000)
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .embed import data_home
from .extract import Document
from .federation import canonical_url
from .passages import _STOPWORDS

MAX_TEXT_CHARS = 300_000  # per page
MIN_SIMILARITY = 0.35  # a page this close to the query matches by meaning
KEYWORD_FLOOR = 0.20  # a page sharing words with the query but further than this in meaning is dropped
SCHEMA_VERSION = 1
_TERM = re.compile(r"[^\W_]+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,          -- canonical URL
    url TEXT NOT NULL,
    published TEXT,
    fetched_at REAL NOT NULL,
    content_hash TEXT NOT NULL,
    chars INTEGER NOT NULL,
    extractor TEXT NOT NULL,
    model TEXT,                        -- embedding model that produced vector
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

    def document(self) -> Document:
        return Document(self.url, self.title, self.text, self.published, self.extractor, False)


def default_path() -> Path:
    return data_home() / "index.sqlite3"


def enabled() -> bool:
    return os.environ.get("WEB_EXPRESS_INDEX", "").strip().lower() != "off"


def _max_pages() -> int:
    try:
        return max(100, int(os.environ.get("WEB_EXPRESS_MAX_PAGES", "20000")))
    except ValueError:
        return 20000


def _match_expression(query: str) -> str:
    """An FTS5 query matching any content word of query; words are quoted, so no operator can be injected."""
    terms = [t for t in dict.fromkeys(t.lower() for t in _TERM.findall(query)) if t not in _STOPWORDS][:24]
    return " OR ".join(f'"{t}"' for t in terms)


class LocalIndex:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._vectors = None  # (model, ids, matrix), rebuilt after a write

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- writing ---------------------------------------------------------

    def add(self, document: Document, vector=None, model: str | None = None) -> None:
        """Store a page, replacing any earlier copy of the same URL."""
        text = document.text[:MAX_TEXT_CHARS]
        if not text.strip():
            return
        key = canonical_url(document.url)
        blob = vector.astype("float32").tobytes() if vector is not None else None
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
        with self._lock, self._db:
            old = self._db.execute("SELECT id FROM pages WHERE key = ?", (key,)).fetchone()
            if old:
                self._db.execute("DELETE FROM pages_fts WHERE rowid = ?", (old["id"],))
                self._db.execute("DELETE FROM pages WHERE id = ?", (old["id"],))
            cursor = self._db.execute(
                "INSERT INTO pages (key, url, published, fetched_at, content_hash, chars, extractor, model, vector) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, document.url, document.published, time.time(), digest, len(text), document.extractor,
                 model if blob else None, blob),
            )
            self._db.execute(
                "INSERT INTO pages_fts (rowid, title, text) VALUES (?, ?, ?)", (cursor.lastrowid, document.title, text)
            )
            self._prune()
            self._vectors = None

    def _prune(self) -> None:
        extra = self._db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] - _max_pages()
        if extra > 0:
            ids = [r["id"] for r in self._db.execute("SELECT id FROM pages ORDER BY fetched_at LIMIT ?", (extra,))]
            marks = ",".join("?" * len(ids))
            self._db.execute(f"DELETE FROM pages_fts WHERE rowid IN ({marks})", ids)
            self._db.execute(f"DELETE FROM pages WHERE id IN ({marks})", ids)

    def remove(self, url: str) -> bool:
        with self._lock, self._db:
            row = self._db.execute("SELECT id FROM pages WHERE key = ?", (canonical_url(url),)).fetchone()
            if not row:
                return False
            self._db.execute("DELETE FROM pages_fts WHERE rowid = ?", (row["id"],))
            self._db.execute("DELETE FROM pages WHERE id = ?", (row["id"],))
            self._vectors = None
            return True

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM pages_fts")
            self._db.execute("DELETE FROM pages")
            self._vectors = None

    # --- reading ---------------------------------------------------------

    def _page(self, row) -> IndexedPage:
        return IndexedPage(row["url"], row["title"], row["text"], row["published"], row["fetched_at"], row["extractor"])

    _SELECT = (
        "SELECT p.id, p.url, p.published, p.fetched_at, p.extractor, f.title, f.text "
        "FROM pages p JOIN pages_fts f ON f.rowid = p.id "
    )

    def get(self, url: str, max_age: float | None = None) -> IndexedPage | None:
        """The stored copy of url, if there is one no older than max_age seconds."""
        with self._lock:
            row = self._db.execute(self._SELECT + "WHERE p.key = ?", (canonical_url(url),)).fetchone()
        if row is None or (max_age is not None and time.time() - row["fetched_at"] > max_age):
            return None
        return self._page(row)

    def _vector_table(self, model: str):
        import numpy as np

        if self._vectors is None or self._vectors[0] != model:
            rows = self._db.execute("SELECT id, vector FROM pages WHERE model = ? AND vector IS NOT NULL", (model,)).fetchall()
            ids = [r["id"] for r in rows]
            matrix = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(len(rows), -1) if rows else None
            self._vectors = (model, ids, matrix)
        return self._vectors[1], self._vectors[2]

    def search(self, query: str, limit: int = 10, query_vector=None, model: str | None = None,
               newer_than: float | None = None) -> list[IndexedPage]:
        """Pages matching query by words and, when a query vector is given, by meaning; best first."""
        scores: dict[int, float] = {}
        with self._lock:
            closeness: dict[int, float] = {}
            if query_vector is not None and model:
                ids, matrix = self._vector_table(model)
                if matrix is not None and matrix.shape[1] == len(query_vector):
                    similarity = matrix @ query_vector
                    closeness = dict(zip(ids, similarity.tolist()))
                    best = similarity.argsort()[::-1][: limit * 3]
                    for rank, position in enumerate(best):
                        if similarity[position] >= MIN_SIMILARITY:  # not merely the least unrelated page
                            scores[ids[position]] = scores.get(ids[position], 0.0) + 1.0 / (60 + rank + 1)
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
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"path": str(self.path), "pages": row["pages"], "characters": row["chars"],
                "pages_with_vectors": row["with_vectors"], "oldest": row["oldest"], "newest": row["newest"],
                "file_bytes": size}
