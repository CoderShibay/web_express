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
#
# A pack from someone else is never used as it arrives. Its rows are read as
# plain values from three ordinary tables, checked one by one, and written into
# a new file that web_express creates itself. Nothing else in the incoming file
# is touched: no view, trigger, virtual table or generated column in it can run.

_EXPECTED_COLUMNS = {
    "pages": ["id", "key", "url", "published", "fetched_at", "content_hash", "chars", "extractor", "model", "dims", "vector"],
    "pages_fts_content": ["id", "c0", "c1"],
    "pack_info": ["key", "value"],
}
MAX_PACK_PAGES = 500_000
MAX_VALUE_BYTES = 2_000_000  # the longest text or blob accepted from an incoming pack
MAX_INFO_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


def _clean_info(value) -> str:
    return _CONTROL.sub(" ", value).strip()[:MAX_INFO_CHARS] if isinstance(value, str) else ""


def _import(incoming: Path, output: Path) -> dict:
    """Copy the pages of an untrusted pack file into a fresh, trusted one at output. Returns its details."""
    from .extract import Document
    from .index import _clean_vector

    try:
        source = sqlite3.connect(f"{incoming.resolve().as_uri()}?mode=ro", uri=True)
    except (sqlite3.Error, ValueError) as exc:
        raise PackError("This file is not a web_express pack.") from exc
    store = None
    try:
        try:
            source.execute("PRAGMA trusted_schema = OFF")
            source.execute("PRAGMA query_only = ON")
            source.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
            source.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
        except sqlite3.Error as exc:
            raise PackError("This file is not a web_express pack.") from exc
        try:
            for table, expected in _EXPECTED_COLUMNS.items():
                kind = source.execute("SELECT type FROM sqlite_master WHERE name = ?", (table,)).fetchall()
                if kind != [("table",)]:
                    raise PackError("This file is not a web_express pack.")
                columns = source.execute(f"PRAGMA table_xinfo({table})").fetchall()
                if [c[1] for c in columns] != expected or any(c[6] for c in columns):  # c[6]: hidden or generated
                    raise PackError("This pack's tables do not have the expected layout.")
            info = {}
            for key, value in source.execute("SELECT key, value FROM pack_info LIMIT 50"):
                if isinstance(key, str) and key in INFO_FIELDS:
                    info[key] = _clean_info(value)
            name = _check_name(info.get("name", ""))
            for needed in ("title", "licence", "attribution"):
                if not info.get(needed):
                    raise PackError(f"This pack does not state its {needed} and was refused.")
            model = info.get("model", "")

            output.unlink(missing_ok=True)
            store = LocalIndex(output)
            texts = source.execute("SELECT id, c0, c1 FROM pages_fts_content")
            details = {row[0]: row for row in source.execute(
                "SELECT id, url, published, extractor, model, dims, vector FROM pages LIMIT ?", (MAX_PACK_PAGES + 1,))}
            if len(details) > MAX_PACK_PAGES:
                raise PackError("This pack holds more pages than web_express accepts.")
            pages = 0
            for page_id, title, text in texts:
                row = details.get(page_id)
                if row is None or not isinstance(text, str) or not isinstance(row[1], str):
                    continue
                _, url, published, extractor, row_model, dims, blob = row
                if urlsplit(url).scheme not in ("http", "https"):
                    continue
                vector = None
                if model and row_model == model and isinstance(blob, bytes) and isinstance(dims, int) and len(blob) == dims * 4:
                    import numpy as np

                    vector = np.frombuffer(blob, dtype=np.float32)
                    if _clean_vector(vector)[0] is None:
                        vector = None
                document = Document(url, title if isinstance(title, str) else "", text,
                                    published if isinstance(published, str) else None,
                                    extractor if isinstance(extractor, str) else "pack", False)
                if store.add(document, vector, model if vector is not None else None):
                    pages += 1
        except sqlite3.Error as exc:
            raise PackError("This pack could not be read safely and was refused.") from exc
        if pages == 0:
            raise PackError("This pack holds no usable pages.")
        info["pages"] = str(pages)
        with store._lock, store._db:
            store._db.execute("CREATE TABLE pack_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            store._db.executemany("INSERT INTO pack_info VALUES (?, ?)", list(info.items()))
        store._db.execute("PRAGMA journal_mode = DELETE")
        store._db.execute("VACUUM")
        return info
    finally:
        source.close()
        if store is not None:
            store.close()


def _download(source: str, target: Path, transport: httpx.BaseTransport | None) -> None:
    """Download a pack over https, refusing any redirect that leaves https, any encoding, and anything too big."""
    url = source
    try:
        with httpx.Client(follow_redirects=False, timeout=60, transport=transport,
                          headers={"Accept-Encoding": "identity"}) as client:
            for _ in range(6):
                if not url.startswith("https://"):
                    raise PackError("Packs are only downloaded over https, including after redirects.")
                with client.stream("GET", url) as response:
                    if response.is_redirect and "location" in response.headers:
                        url = str(httpx.URL(url).join(response.headers["location"]))
                        continue
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
                        raise PackError("The server sent the pack in an encoding web_express does not accept.")
                    declared = response.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > MAX_PACK_BYTES:
                        raise PackError("This pack is larger than the 4 GB limit.")
                    size = 0
                    with target.open("wb") as handle:
                        for block in response.iter_raw():
                            size += len(block)
                            if size > MAX_PACK_BYTES:
                                raise PackError("This pack is larger than the 4 GB limit.")
                            handle.write(block)
                    return
            raise PackError("Too many redirects.")
    except httpx.HTTPError as exc:
        raise PackError(f"Could not download the pack ({type(exc).__name__}).") from exc


def install(source: str, sha256: str | None = None, transport: httpx.BaseTransport | None = None,
            replace: bool = False) -> dict:
    """Install a pack from a file path or an https address. Returns its details.

    sha256, if given, must match the file as it arrives. An installed pack of the
    same name is left alone unless replace is set.
    """
    folder = packs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    staging, converted = folder / f".incoming-{token}", folder / f".converted-{token}"
    try:
        if re.match(r"https?://", source):
            _download(source, staging, transport)
        else:
            origin = Path(source).expanduser()
            if not origin.is_file():
                raise PackError(f"No file at {origin}.")
            if origin.stat().st_size > MAX_PACK_BYTES:
                raise PackError("This pack is larger than the 4 GB limit.")
            shutil.copyfile(origin, staging)
        actual = file_sha256(staging)
        if sha256 and actual != sha256.strip().lower():
            raise PackError("The pack does not match the checksum you gave; it was not installed.")
        info = _import(staging, converted)
        destination = folder / f"{info['name']}.sqlite3"
        if destination.exists() and not replace:
            raise PackError(f"A pack named {info['name']} is already installed. Remove it first, or install with replace.")
        os.replace(converted, destination)
        return {**info, "sha256": actual, "file": str(destination)}
    finally:
        for leftover in (staging, converted, converted.with_name(converted.name + "-wal"),
                         converted.with_name(converted.name + "-shm"), converted.with_name(converted.name + "-journal")):
            leftover.unlink(missing_ok=True)


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
