"""Tools for a shared, community-built index, made of corpus packs called shards.

What this design does and does not promise:

1. Nothing leaves a computer by itself. `export_nominations` writes a plain
   list of addresses to a file; the person reviews it and decides whether to
   share it. The list holds addresses only: no dates, no page text.
2. An address list still says something about its owner, so the export is
   deliberately narrow. Only plain https pages on public sites are proposed:
   nothing with a query string, nothing that looks like a search, login, reset,
   share or account page, nothing with a long random-looking part in its path,
   and nothing on a domain listed in WEB_EXPRESS_PRIVATE_DOMAINS. That removes
   the common leaks; it cannot know that a readable address is private to you.
   Read the file before sharing it.
3. Contributors propose addresses, never content. Whoever builds a shard
   fetches and embeds every page themselves, through the same guarded
   downloader as everything else, so nobody can plant text or vectors.
4. An installer can check what it downloads: the manifest lists every shard's
   checksum, and a shard that does not match, or whose name differs from the
   manifest, is refused. Builds are not bit-for-bit repeatable (pages change
   between fetches), so the manifest proves "this is what the builder
   published", not "anyone would have built the same".

Selection rules guard quality: an address must appear in several contributor
files, its site must be well linked on the open web (see authority.py), and no
single site may fill the index. These rules stop accidents and casual abuse.
They do not stop one determined person posing as several contributors; that
needs contributor identities, which this tooling does not provide.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__, packs
from .authority import registrable_domain
from .index import exact_key, storable

MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 5_000_000
MAX_SHARDS = 10_000
MAX_SHARD_SIZE = 100_000
_PRIVATE_WORDS = re.compile(
    r"(^|[/_.-])(search|results?|login|logout|signin|signup|sign-in|sign-up|register|account|accounts|profile|settings|"
    r"reset|recover|password|token|auth|oauth|sso|session|invite|invitation|share|shared|unsubscribe|verify|confirm|"
    r"checkout|cart|order|orders|billing|invoice|payment|admin|private|internal|draft|drafts|preview|inbox|messages?|"
    r"dashboard|my|me|u|user|users|d|document|documents|file|files|browse|ticket|tickets|issue-private)([/_.-]|$)",
    re.IGNORECASE,
)
_RANDOM_PART = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9_-]{16,}$|^[0-9a-fA-F]{12,}$|^\d{7,}$")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CommunityError(Exception):
    """A nomination list or manifest could not be used; the message is safe to show."""


def _private_domains() -> list[str]:
    raw = os.environ.get("WEB_EXPRESS_PRIVATE_DOMAINS", "")
    return [d.strip().lower().lstrip(".") for d in raw.split(",") if d.strip()]


def _public_host(url: str) -> str:
    """The host of url if it looks like a public website, else ''."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
        port = parts.port
    except ValueError:
        return ""
    if parts.scheme != "https" or port not in (None, 443) or ":" in host or host.replace(".", "").isdigit():
        return ""
    if registrable_domain(host) is None:  # not under a real public suffix: localhost, intranet names, .corp, .test
        return ""
    if re.search(r"(^|[.-])\d{1,3}[.-]\d{1,3}[.-]\d{1,3}[.-]\d{1,3}([.-]|$)", host):
        return ""  # names that spell out an IP address, as wildcard DNS services use
    return host


def site_of(url: str) -> str:
    """The registrable site of an address (someone.github.io, bbc.co.uk), used for per-site limits."""
    host = _public_host(url)
    return (registrable_domain(host) or host) if host else ""


def nominable(url: str) -> bool:
    """Whether an address may be proposed for a shared index. Deliberately strict: see the module notes."""
    host = _public_host(url)
    if not host or not storable(url) or len(url) > 300:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.query or parts.fragment or parts.username:
        return False  # query strings carry searches, IDs and tokens: never proposed
    if any(host == d or host.endswith("." + d) for d in _private_domains()):
        return False
    path = parts.path
    if _PRIVATE_WORDS.search(path) or _UUID.search(path) or "%" in path or "@" in path:
        return False
    return not any(_RANDOM_PART.match(segment) for segment in re.split(r"[/.]", path) if segment)


def export_nominations(index, output: Path, limit: int = 5000, authority=None) -> int:
    """Write addresses from the local index to output, one per line, shuffled. Returns how many.

    Shuffled and undated so the file says nothing about when or in what order pages were read.
    With site-ranking data, only pages on ranked (publicly well-linked) sites are written.
    """
    with index._lock:
        urls = [row["url"] for row in index._db.execute("SELECT url FROM pages")]
    ranked = authority is not None and getattr(authority, "available", False)
    chosen = sorted({exact_key(u) for u in urls if nominable(u) and (not ranked or authority.position(u) is not None)})
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


def _blocked(url: str, rules: list[tuple[str, str]]) -> bool:
    host, site = _public_host(url), site_of(url)
    path = urlsplit(url).path or "/"
    for rule_host, rule_path in rules:
        if host == rule_host or host.endswith("." + rule_host) or site == rule_host:
            if rule_path in ("", "/") or path == rule_path or path.startswith(rule_path.rstrip("/") + "/"):
                return True
    return False


def _block_rules(entries) -> list[tuple[str, str]]:
    """Takedown entries as (host, path prefix). 'bad.example' blocks a whole site;
    'https://bad.example/path' blocks that page and everything under it, whatever the query."""
    rules = []
    for entry in entries or ():
        entry = entry.strip()
        if not entry:
            continue
        try:
            parts = urlsplit(entry if "//" in entry else "//" + entry)
            host = (parts.hostname or "").lower().rstrip(".")
        except ValueError:
            continue
        if host:
            rules.append((host, parts.path))
    return rules


def select(
    nomination_files: list[Path],
    *,
    quorum: int = 2,
    per_site: int = 200,
    authority=None,
    max_site_rank: int | None = 1_000_000,
    blocked=None,
) -> tuple[list[str], dict]:
    """Choose the addresses to index. Returns (addresses, a report of what was dropped and why).

    quorum          contributor files that must propose an address (one vote per distinct file)
    per_site        most addresses accepted from one registrable site, subdomains included
    authority       an Authority; with max_site_rank, sites outside that rank are dropped
    blocked         takedown entries: sites, or addresses whose path and everything under it is excluded
    """
    votes: Counter[str] = Counter()
    seen_files: set[str] = set()
    seen_contents: set[str] = set()
    duplicates = 0
    for path in nomination_files:
        resolved = str(Path(path).expanduser().resolve())
        addresses = read_nominations(Path(path))
        fingerprint = hashlib.sha256("\n".join(sorted(addresses)).encode()).hexdigest()
        if resolved in seen_files or fingerprint in seen_contents:
            duplicates += 1  # the same file, or the same list under another name: one vote, not two
            continue
        seen_files.add(resolved)
        seen_contents.add(fingerprint)
        votes.update(addresses)
    rules = _block_rules(blocked)
    report = {"contributor_files": len(seen_files), "duplicate_files_ignored": duplicates, "proposed": len(votes),
              "below_quorum": 0, "blocked": 0, "unranked_site": 0, "site_limit": 0}
    by_site: dict[str, list[str]] = defaultdict(list)
    for url in sorted(votes, key=lambda u: (-votes[u], u)):
        site = site_of(url)
        if votes[url] < quorum:
            report["below_quorum"] += 1
        elif _blocked(url, rules):
            report["blocked"] += 1
        elif authority is not None and authority.available and max_site_rank is not None and (
            (authority.position(url) or max_site_rank + 1) > max_site_rank
        ):
            report["unranked_site"] += 1
        elif len(by_site[site]) >= per_site:
            report["site_limit"] += 1
        else:
            by_site[site].append(url)
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
    """Build shards of at most shard_size pages from urls into folder, and write manifest.json there.

    Every address is checked again here, whatever produced the list: only addresses
    that could have been proposed are fetched.
    """
    packs._check_name(name)
    if not isinstance(shard_size, int) or not 1 <= shard_size <= MAX_SHARD_SIZE:
        raise CommunityError(f"shard size must be between 1 and {MAX_SHARD_SIZE}.")
    ordered = sorted({exact_key(u.strip()) for u in urls if nominable(u.strip())})  # a stable order
    refused = len({u.strip() for u in urls if u.strip()}) - len(ordered)
    if not ordered:
        raise CommunityError("No usable addresses were given; nothing to build.")
    if len(ordered) > shard_size * MAX_SHARDS:
        raise CommunityError("Too many addresses for one release; raise the shard size or split the list.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shards, failed = [], {}
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
        "model": model_id or "none",
        "licence": licence,
        "attribution": attribution,
        "rules": rules or {},
        "addresses": len(ordered),
        "pages": sum(s["pages"] for s in shards),
        "unreadable": len(failed),
        "shards": shards,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest, "failed": failed, "refused_addresses": refused}


def read_manifest(path: Path) -> dict:
    path = Path(path)
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise CommunityError("This manifest is too large.")
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CommunityError(f"Could not read a manifest at {path}.") from exc
    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(shards, list) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise CommunityError("This manifest has a layout web_express does not understand.")
    if not 1 <= len(shards) <= MAX_SHARDS:
        raise CommunityError("This manifest lists no shards, or too many.")
    names = set()
    for shard in shards:
        if not (isinstance(shard, dict) and isinstance(shard.get("file"), str) and isinstance(shard.get("sha256"), str)
                and isinstance(shard.get("name"), str) and packs.NAME.fullmatch(shard["name"])
                and shard["file"] == f"{shard['name']}.pack" and _SHA256.fullmatch(shard["sha256"].lower())):
            raise CommunityError("This manifest lists a shard with an invalid name, file or checksum.")
        if shard["name"] in names:
            raise CommunityError("This manifest lists the same shard twice.")
        names.add(shard["name"])
    return manifest


def install_from_manifest(manifest_path: Path, replace: bool = False) -> list[str]:
    """Install every shard listed in a manifest that sits beside its shard files: all of them, or none.

    Each shard must match its checksum, be a valid pack, and carry the same name inside
    as the manifest gives it. A pack already installed under one of those names is left
    alone, and the install refused, unless replace is set.
    """
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    folder = manifest_path.parent
    prepared: list[tuple[dict, Path]] = []
    moved_aside: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for shard in manifest["shards"]:  # check and convert everything before changing anything
            path = folder / shard["file"]
            if not path.is_file():
                raise CommunityError(f"Shard file {shard['file']} is missing.")
            try:
                info, converted = packs.prepare(str(path), sha256=shard["sha256"])
            except packs.PackError as exc:
                raise CommunityError(f"Shard {shard['file']} was refused: {exc}") from exc
            prepared.append((info, converted))
            if info["name"] != shard["name"]:
                raise CommunityError(f"Shard {shard['file']} names itself {info['name']!r}, not {shard['name']!r}; nothing was installed.")
            if packs.destination(info["name"]).exists() and not replace:
                raise CommunityError(f"A pack named {info['name']} is already installed; nothing was installed.")
        try:
            for info, converted in prepared:
                target = packs.destination(info["name"])
                if target.exists():
                    aside = target.with_name(f".replaced-{target.name}")
                    os.replace(target, aside)
                    moved_aside.append((aside, target))
                os.replace(converted, target)
                installed.append(target)
        except OSError as exc:  # put everything back as it was
            for target in installed:
                target.unlink(missing_ok=True)
            for aside, target in moved_aside:
                os.replace(aside, target)
            moved_aside.clear()
            raise CommunityError(f"The install could not be completed and was undone ({exc}).") from exc
        return [info["name"] for info, _ in prepared]
    finally:
        for _, converted in prepared:
            packs.discard(converted)
        for aside, _ in moved_aside:
            aside.unlink(missing_ok=True)
