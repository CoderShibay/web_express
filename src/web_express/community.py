"""Tools for a shared, community-built index, made of corpus packs called shards.

The design keeps three promises:

1. Nothing leaves a computer by itself. `export_nominations` writes a plain
   list of addresses to a file; the person decides whether to share that file.
   The list holds addresses only: no searches, no dates, no page text, and
   never an address that carries a credential.
2. Contributors propose addresses, never content. Whoever builds a shard
   fetches and embeds every page themselves, so nobody can plant text or
   vectors in the index.
3. A build can be checked. The manifest records every shard's checksum, the
   selection rules and the embedding model, so a second builder given the same
   nominations can confirm the result, and an installer refuses any shard that
   does not match.

Selection rules guard quality without a human reviewer: an address must be
proposed by several independent contributors, its site must be well linked on
the open web (see authority.py), and no single site may fill the index.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__, packs
from .index import exact_key, storable

MANIFEST_VERSION = 1


class CommunityError(Exception):
    """A nomination list or manifest could not be used; the message is safe to show."""


def _public_host(url: str) -> str:
    """The host of url if it looks like a public website, else ''."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
    except ValueError:
        return ""
    if parts.scheme != "https" or "." not in host or host.replace(".", "").isdigit() or ":" in host:
        return ""
    if host == "localhost" or host.endswith((".local", ".internal", ".lan", ".home", ".corp", ".test", ".invalid")):
        return ""
    return host


def nominable(url: str) -> bool:
    """Whether an address may be proposed: public https, no credentials, not absurdly long."""
    return bool(_public_host(url)) and storable(url) and len(url) <= 500


def export_nominations(index, output: Path, limit: int = 5000) -> int:
    """Write addresses from the local index to output, one per line, shuffled. Returns how many.

    Shuffled and undated so the file says nothing about when or in what order pages were read.
    """
    with index._lock:
        urls = [row["url"] for row in index._db.execute("SELECT url FROM pages")]
    chosen = sorted({exact_key(u) for u in urls if nominable(u)})
    random.SystemRandom().shuffle(chosen)
    chosen = chosen[: max(0, limit)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")
    return len(chosen)


def read_nominations(path: Path) -> set[str]:
    """The valid addresses in one contributor's file; anything else in it is ignored."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise CommunityError(f"Could not read {path}.") from exc
    return {exact_key(line.strip()) for line in lines[:200_000] if nominable(line.strip())}


def select(
    nomination_files: list[Path],
    *,
    quorum: int = 2,
    per_site: int = 200,
    authority=None,
    max_site_rank: int | None = 1_000_000,
    blocked: set[str] | None = None,
) -> tuple[list[str], dict]:
    """Choose the addresses to index. Returns (addresses, a report of what was dropped and why).

    quorum          contributors that must propose an address (one vote per file)
    per_site        most addresses accepted from one site
    authority       an Authority; with max_site_rank, sites outside that rank are dropped
    blocked         addresses or sites that must never be included (takedowns)
    """
    votes: Counter[str] = Counter()
    for path in nomination_files:
        votes.update(read_nominations(path))
    blocked = {b.strip().lower() for b in (blocked or set()) if b.strip()}
    report = {"proposed": len(votes), "below_quorum": 0, "blocked": 0, "unranked_site": 0, "site_limit": 0}
    by_site: dict[str, list[str]] = defaultdict(list)
    for url in sorted(votes, key=lambda u: (-votes[u], u)):
        host = _public_host(url)
        if votes[url] < quorum:
            report["below_quorum"] += 1
        elif url.lower() in blocked or any(host == b or host.endswith("." + b) for b in blocked):
            report["blocked"] += 1
        elif authority is not None and authority.available and max_site_rank is not None and (
            (authority.position(url) or max_site_rank + 1) > max_site_rank
        ):
            report["unranked_site"] += 1
        elif len(by_site[host]) >= per_site:
            report["site_limit"] += 1
        else:
            by_site[host].append(url)
    chosen = sorted(url for urls in by_site.values() for url in urls)
    report["selected"] = len(chosen)
    report["sites"] = len([urls for urls in by_site.values() if urls])
    return chosen, report


async def build_shards(
    urls: list[str],
    folder: Path,
    *,
    name: str,
    licence: str,
    attribution: str,
    shard_size: int = 1000,
    rules: dict | None = None,
    embedder=None,
    model_id: str | None = None,
    delay: float = 1.0,
    loader=None,
    progress=None,
) -> dict:
    """Build shards of at most shard_size pages from urls into folder, and write manifest.json there."""
    packs._check_name(name)
    if not urls:
        raise CommunityError("No addresses were selected; nothing to build.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shards, failed = [], {}
    ordered = sorted(dict.fromkeys(urls))  # a stable order, so two builders cut the same shards
    for number, start in enumerate(range(0, len(ordered), shard_size), 1):
        shard_name = f"{name}-{number:04d}"
        try:
            built = await packs.build(
                shard_name, ordered[start : start + shard_size], folder / f"{shard_name}.pack",
                title=f"{name} shard {number}", licence=licence, attribution=attribution, source="community nominations",
                delay=delay, embedder=embedder, model_id=model_id, loader=loader, progress=progress,
            )
        except packs.PackError as exc:
            failed[shard_name] = str(exc)
            continue
        failed.update(built["failed"])
        shards.append({"name": shard_name, "file": f"{shard_name}.pack", "sha256": built["sha256"], "pages": int(built["pages"])})
    if not shards:
        raise CommunityError("No shard could be built: none of the selected pages could be read.")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "name": name,
        "built": time.strftime("%Y-%m-%d"),
        "builder": f"web_express {__version__}",
        "model": model_id or "",
        "licence": licence,
        "attribution": attribution,
        "rules": rules or {},
        "addresses": len(ordered),
        "pages": sum(s["pages"] for s in shards),
        "unreadable": len(failed),
        "shards": shards,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "failed": failed}


def read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CommunityError(f"Could not read a manifest at {path}.") from exc
    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(shards, list) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise CommunityError("This manifest has a layout web_express does not understand.")
    for shard in shards:
        if not (isinstance(shard, dict) and isinstance(shard.get("file"), str) and isinstance(shard.get("sha256"), str)
                and packs.NAME.fullmatch(str(shard.get("name", ""))) and shard["file"] == f"{shard['name']}.pack"):
            raise CommunityError("This manifest lists a shard with an invalid name, file or checksum.")
    return manifest


def install_from_manifest(manifest_path: Path) -> list[str]:
    """Install every shard listed in a manifest that sits beside its shard files.

    Each shard must match the checksum in the manifest, and all are checked before any is installed.
    """
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    folder = manifest_path.parent
    for shard in manifest["shards"]:
        path = folder / shard["file"]
        if not path.is_file():
            raise CommunityError(f"Shard file {shard['file']} is missing.")
        if packs.file_sha256(path) != shard["sha256"].lower():
            raise CommunityError(f"Shard {shard['file']} does not match the manifest; nothing was installed.")
    installed = []
    for shard in manifest["shards"]:
        try:
            installed.append(packs.install(str(folder / shard["file"]), sha256=shard["sha256"], replace=True)["name"])
        except packs.PackError as exc:
            raise CommunityError(f"Shard {shard['file']} was refused: {exc}") from exc
    return installed
