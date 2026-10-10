"""Corpus packs: ready-made, searchable collections of pages that can be shared.

A pack is one SQLite file in the same layout as the local index, plus a
`pack_info` table naming its licence and source. Installing a pack makes its
pages searchable offline straight away, by words and by meaning, without
fetching anything.

Packs are meant for material that may be redistributed: official
documentation, open-access papers, public-domain and Creative Commons text.
Whoever builds a pack states its licence and attribution; web_express shows
them with every result from that pack.

A pack downloaded from somewhere else is opened read-only, with SQLite's
defences for untrusted files switched on, and is refused if it contains
anything beyond the expected tables.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from .embed import data_home
from .index import LocalIndex
from .net import FetchError, download
from .extract import extract

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,40}")
MAX_PACK_BYTES = 4_000_000_000
INFO_FIELDS = ("name", "title", "licence", "attribution", "source", "built", "model", "pages")
_ALLOWED_OBJECTS = re.compile(
    r"pages|pages_fetched|pack_info|pages_fts(_data|_idx|_content|_docsize|_config)?|sqlite_autoindex_\w+"
)


class PackError(Exception):
    """A pack could not be built, installed or opened; the message is safe to show."""


def packs_dir() -> Path:
    return data_home() / "packs"


def _check_name(name: str) -> str:
    if not NAME.fullmatch(name or ""):
        raise PackError("A pack name is lower-case letters, digits and hyphens, at most 41 characters.")
    return name


class Pack(LocalIndex):
    """A pack opened read-only. Search and get work as on the local index; nothing can be written."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._vectors = None
        self._writes = 0
        if not self.path.is_file():
            raise PackError(f"No pack file at {self.path}.")
        try:
            self._db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA trusted_schema = OFF")
            self._db.execute("PRAGMA query_only = ON")
            objects = self._db.execute("SELECT type, name FROM sqlite_master").fetchall()
        except sqlite3.Error as exc:
            raise PackError("This file is not a web_express pack.") from exc
        names = {row["name"] for row in objects}
        for row in objects:
            if row["type"] not in ("table", "index") or not _ALLOWED_OBJECTS.fullmatch(row["name"]):
                self._db.close()
                raise PackError(f"This pack contains something unexpected ({row['type']} {row['name']!r}) and was refused.")
        if not {"pages", "pages_fts", "pack_info"} <= names:
            self._db.close()
            raise PackError("This file is not a web_express pack.")
        try:
            self.info = {row["key"]: row["value"] for row in self._db.execute("SELECT key, value FROM pack_info")}
            self._db.execute("SELECT id, key, url, published, fetched_at, extractor, model, dims, vector FROM pages LIMIT 1").fetchall()
            self._db.execute("SELECT title, text FROM pages_fts LIMIT 1").fetchall()
        except sqlite3.Error as exc:
            self._db.close()
            raise PackError("This pack's tables do not have the expected layout.") from exc
        self.name = _check_name(str(self.info.get("name", "")))

    def add(self, *args, **kwargs) -> None:
        raise PackError("Packs are read-only.")

    remove = clear = tidy = add


# --- building -------------------------------------------------------------------

async def build(
    name: str,
    urls: list[str],
    output: Path,
    *,
    title: str,
    licence: str,
    attribution: str,
    source: str = "",
    delay: float = 1.0,
    embedder=None,
    model_id: str | None = None,
    loader=None,
    progress=None,
) -> dict:
    """Fetch each URL, and write the pages as a pack file at output.

    One page at a time with a pause between requests to the same site: a pack
    build should not lean on anyone's server. Pages that cannot be read are
    skipped and reported.
    """
    _check_name(name)
    if not licence.strip() or not attribution.strip():
        raise PackError("A pack must state its licence and attribution.")
    if not urls:
        raise PackError("Give at least one URL.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part-{uuid.uuid4().hex[:8]}")

    async def default_loader(url: str):
        return await asyncio.to_thread(extract, await download(url))

    loader = loader or default_loader
    store = LocalIndex(partial)
    failed: dict[str, str] = {}
    last_request: dict[str, float] = {}
    try:
        for number, url in enumerate(dict.fromkeys(u.strip() for u in urls if u.strip()), 1):
            host = urlsplit(url).hostname or ""
            wait = delay - (time.monotonic() - last_request.get(host, 0.0))
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                document = await loader(url)
            except FetchError as exc:
                failed[url] = str(exc)
                continue
            finally:
                last_request[host] = time.monotonic()
            vector = None
            if embedder is not None and document.text.strip():
                vector = await asyncio.to_thread(_page_vector, document, embedder)
            store.add(document, vector, model_id if vector is not None else None)
            if progress:
                progress(number, url)
        pages = store.stats()["pages"]
        if pages == 0:
            raise PackError("None of the URLs could be read; no pack was written.")
        info = {"name": name, "title": title, "licence": licence, "attribution": attribution, "source": source,
                "built": time.strftime("%Y-%m-%d"), "model": model_id or "", "pages": str(pages)}
        with store._lock, store._db:
            store._db.execute("CREATE TABLE pack_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            store._db.executemany("INSERT INTO pack_info VALUES (?, ?)", list(info.items()))
        store._db.execute("PRAGMA journal_mode = DELETE")  # one self-contained file, no -wal beside it
        store._db.execute("VACUUM")
        store.close()
        os.replace(partial, output)
    finally:
        try:
            store.close()
        except sqlite3.Error:
            pass
        for leftover in (partial, partial.with_name(partial.name + "-wal"), partial.with_name(partial.name + "-shm")):
            leftover.unlink(missing_ok=True)
    return {**info, "file": str(output), "sha256": file_sha256(output), "failed": failed}


def _page_vector(document, embedder):
    from .passages import split_passages

    passages = split_passages(document.text)
    sample = passages[:: max(1, len(passages) // 24)][:24]
    vectors = embedder.embed([document.title or " "] + [p.text for p in sample])
    vector = vectors.mean(axis=0)
    norm = float((vector @ vector) ** 0.5)
    return vector / norm if norm > 0 else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- installing -----------------------------------------------------------------

def install(source: str, sha256: str | None = None, transport: httpx.BaseTransport | None = None) -> dict:
    """Install a pack from a file path or an https address. Returns its details."""
    folder = packs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    staging = folder / f".incoming-{uuid.uuid4().hex[:8]}"
    try:
        if re.match(r"https?://", source):
            if not source.startswith("https://"):
                raise PackError("Packs are only downloaded over https.")
            size = 0
            try:
                with httpx.Client(follow_redirects=True, timeout=60, transport=transport) as client:
                    with client.stream("GET", source) as response:
                        response.raise_for_status()
                        with staging.open("wb") as handle:
                            for block in response.iter_bytes():
                                size += len(block)
                                if size > MAX_PACK_BYTES:
                                    raise PackError("This pack is larger than the 4 GB limit.")
                                handle.write(block)
            except httpx.HTTPError as exc:
                raise PackError(f"Could not download the pack ({type(exc).__name__}).") from exc
        else:
            origin = Path(source).expanduser()
            if not origin.is_file():
                raise PackError(f"No file at {origin}.")
            shutil.copyfile(origin, staging)
        actual = file_sha256(staging)
        if sha256 and actual != sha256.strip().lower():
            raise PackError("The pack does not match the checksum you gave; it was not installed.")
        pack = Pack(staging)  # refuses anything that is not a clean pack
        info, name = dict(pack.info), pack.name
        pack.close()
        os.replace(staging, folder / f"{name}.sqlite3")
        return {**info, "sha256": actual, "file": str(folder / f"{name}.sqlite3")}
    finally:
        staging.unlink(missing_ok=True)


def installed() -> list[Pack]:
    """Every pack in the packs folder that opens cleanly."""
    folder = packs_dir()
    packs = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.sqlite3")):
            try:
                packs.append(Pack(path))
            except PackError:
                continue
    return packs


def remove(name: str) -> bool:
    path = packs_dir() / f"{_check_name(name)}.sqlite3"
    if not path.is_file():
        return False
    path.unlink()
    return True
