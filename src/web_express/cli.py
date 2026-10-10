"""The web-express command.

    web-express                 run the MCP server (what an agent launches)
    web-express setup           download the ranking model now, instead of on first search
    web-express init            register web_express with the AI agents installed here
    web-express status          show what is installed and configured
    web-express search QUERY    run a search from the terminal
    web-express fetch URL       read a page from the terminal
    web-express index ...       inspect or clear the local index
    web-express pack ...        build, install, list or remove corpus packs
    web-express authority ...   install or inspect the site-ranking data
    web-express community ...   propose addresses and build or install shared shards
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import __version__

SERVER_NAME = "web_express"


# --- how an agent should launch the server ------------------------------------

def launch_command() -> list[str]:
    """The command an MCP client should run to start the server."""
    script = shutil.which("web-express")
    beside = Path(sys.executable).with_name("web-express.exe" if os.name == "nt" else "web-express")
    if beside.is_file():
        return [str(beside)]
    if script:
        return [str(Path(script).resolve())]
    return [sys.executable, "-m", "web_express"]


# --- registering with agents ---------------------------------------------------

def _home() -> Path:
    return Path.home()


def json_clients() -> dict[str, Path]:
    """Agents whose MCP servers live under "mcpServers" in a JSON file."""
    home = _home()
    if sys.platform == "darwin":
        desktop = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif os.name == "nt":
        desktop = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "Claude" / "claude_desktop_config.json"
    else:
        desktop = home / ".config" / "Claude" / "claude_desktop_config.json"
    return {
        "claude-desktop": desktop,
        "cursor": home / ".cursor" / "mcp.json",
        "gemini": home / ".gemini" / "settings.json",
        "antigravity": home / ".gemini" / "config" / "mcp_config.json",
    }


def codex_config() -> Path:
    return _home() / ".codex" / "config.toml"


def _backup(path: Path) -> Path:
    """Copy path to a new backup file beside it. Never overwrites an earlier backup."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for attempt in range(1000):
        suffix = f".bak-web_express-{stamp}" + (f"-{attempt}" if attempt else "")
        copy = path.with_name(path.name + suffix)
        try:
            with copy.open("xb") as handle:  # fails if the name is taken, so two runs cannot share a backup
                handle.write(path.read_bytes())
        except FileExistsError:
            continue
        shutil.copymode(path, copy)
        return copy
    raise OSError(f"could not create a backup of {path}")


def _write(path: Path, data: bytes) -> None:
    """Replace path with data in one step, keeping its permissions (these files can hold tokens)."""
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    handle, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")  # created 0600
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(partial, mode)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


def register_json(path: Path, command: list[str], remove: bool = False, dry_run: bool = False) -> str:
    """Add or remove the server in a JSON MCP config. Returns a line describing what happened."""
    if not path.parent.is_dir():
        return "not installed"
    data: dict = {}
    bom = b""
    if path.is_file():
        raw = path.read_bytes()
        bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
        try:
            data = json.loads(raw[len(bom):].decode("utf-8") or "{}")
        except ValueError:
            return f"skipped: {path} is not plain JSON; add the server by hand"
        if not isinstance(data, dict):
            return f"skipped: {path} has an unexpected layout"
    servers = data.get("mcpServers")
    if servers is None:
        servers = {}
    if not isinstance(servers, dict):
        return f"skipped: {path} has an unexpected layout"
    existing = servers.get(SERVER_NAME)
    if remove:
        if SERVER_NAME not in servers:
            return "not registered"
        del servers[SERVER_NAME]
    else:
        if existing is not None and not isinstance(existing, dict):
            return f"skipped: {path} has an unexpected layout"
        entry = dict(existing or {})  # keep any extra settings the user added, such as env
        entry["command"], entry["args"] = command[0], command[1:]
        if entry == existing:
            return "already registered"
        servers[SERVER_NAME] = entry
    data["mcpServers"] = servers
    if dry_run:
        return f"would {'remove from' if remove else 'update'} {path}"
    backup = f" (backup: {_backup(path).name})" if path.is_file() else ""
    _write(path, bom + (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return f"{'removed from' if remove else 'registered in'} {path}{backup}"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # JSON string syntax is valid TOML basic-string syntax


_TABLE_HEADER = re.compile(r"^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*(#.*)?$")


def _header_name(line: str) -> str | None:
    match = _TABLE_HEADER.match(line.rstrip("\r\n"))
    return re.sub(r"\s*\.\s*", ".", match.group(1)) if match else None


def register_codex(path: Path, command: list[str], remove: bool = False, dry_run: bool = False) -> str:
    """Add or remove the [mcp_servers.web_express] table in Codex's config.toml.

    Only the `command` and `args` lines are written; every other line, including
    other settings in the same table, comments and line endings, is left as it is.
    If the file is laid out in a way this cannot edit safely, nothing is changed.
    """
    import tomllib

    if not path.parent.is_dir():
        return "not installed"
    by_hand = f"skipped: could not edit {path} safely; add the server by hand"
    try:
        text = path.read_bytes().decode("utf-8") if path.is_file() else ""
        current = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return f"skipped: {path} is not valid TOML; add the server by hand"
    servers = current.get("mcp_servers", {})
    if not isinstance(servers, dict) or not isinstance(servers.get(SERVER_NAME, {}), dict):
        return f"skipped: {path} has an unexpected layout"
    newline = "\r\n" if "\r\n" in text else "\n"
    table = f"mcp_servers.{SERVER_NAME}"
    wanted = {"command": command[0], "args": command[1:]}
    new_lines = [f"command = {_toml_string(command[0])}", f"args = [{', '.join(_toml_string(a) for a in command[1:])}]"]
    existing = servers.get(SERVER_NAME)
    lines = text.splitlines(keepends=True)

    def spans(prefix_too: bool) -> list[tuple[int, int]]:
        """Line ranges of the server's table, and of its sub-tables when prefix_too."""
        found, start = [], None
        for number, line in enumerate(lines + ["[end]"]):
            name = _header_name(line)
            if name is None:
                continue
            if start is not None:
                found.append((start, number))
                start = None
            if name == table or (prefix_too and name.startswith(table + ".")):
                start = number
        return found

    if existing is None:
        if remove:
            return "not registered"
        if not text or text.endswith(("\n\n", "\r\n\r\n")):
            gap = ""
        elif text.endswith("\n"):
            gap = newline
        else:
            gap = newline * 2
        new_text = text + gap + newline.join([f"[{table}]", *new_lines]) + newline
    elif remove:
        doomed = spans(prefix_too=True)
        if not doomed:
            return by_hand
        keep = [line for number, line in enumerate(lines) if not any(a <= number < b for a, b in doomed)]
        new_text = "".join(keep)
    else:
        if {k: existing.get(k) for k in wanted} == wanted:
            return "already registered"
        own = spans(prefix_too=False)
        if len(own) != 1:
            return by_hand
        start, end = own[0]
        body = []
        for line in lines[start + 1 : end]:
            if re.match(r"\s*(command|args)\s*=", line):
                continue  # replaced below; a value spread over several lines fails the check afterwards
            body.append(line)
        new_text = "".join(lines[:start] + [lines[start].rstrip("\r\n") + newline]
                           + [entry + newline for entry in new_lines] + body + lines[end:])

    # Prove the edit did exactly what was intended and nothing else, or refuse it.
    try:
        after = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:
        return by_hand
    expected = json.loads(json.dumps(current, default=str))
    expected_servers = expected.setdefault("mcp_servers", {})
    if remove:
        expected_servers.pop(SERVER_NAME, None)
        if not expected_servers and "mcp_servers" not in after:
            expected.pop("mcp_servers")
    else:
        expected_servers[SERVER_NAME] = {**(existing or {}), **wanted}
    if json.loads(json.dumps(after, default=str)) != json.loads(json.dumps(expected, default=str)):
        return by_hand
    if dry_run:
        return f"would {'remove from' if remove else 'update'} {path}"
    backup = f" (backup: {_backup(path).name})" if path.is_file() else ""
    _write(path, new_text.encode("utf-8"))
    return f"{'removed from' if remove else 'registered in'} {path}{backup}"


def register_claude_code(command: list[str], remove: bool = False, dry_run: bool = False) -> str:
    """Claude Code keeps its own config; ask its CLI to make the change."""
    claude = shutil.which("claude")
    if not claude:
        return "not installed"
    if remove:
        arguments = [claude, "mcp", "remove", "--scope", "user", SERVER_NAME]
    else:
        arguments = [claude, "mcp", "add", "--scope", "user", SERVER_NAME, "--", *command]
    if dry_run:
        return "would run: " + " ".join(arguments)
    try:
        done = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed: {exc}"
    message = (done.stdout or done.stderr).strip().splitlines()
    summary = message[-1] if message else ""
    if done.returncode == 0:
        return "removed" if remove else "registered"
    if "already exists" in (done.stdout + done.stderr):
        return "already registered"
    return f"failed: {summary}"


CLIENTS = ("claude-code", "claude-desktop", "cursor", "codex", "gemini", "antigravity")


def init(clients: list[str], remove: bool, dry_run: bool) -> dict[str, str]:
    command = launch_command()
    results: dict[str, str] = {}
    paths = json_clients()
    for client in clients:
        if client == "claude-code":
            results[client] = register_claude_code(command, remove, dry_run)
        elif client == "codex":
            results[client] = register_codex(codex_config(), command, remove, dry_run)
        else:
            results[client] = register_json(paths[client], command, remove, dry_run)
    return results


# --- commands -------------------------------------------------------------------

def _print_json(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def cmd_setup(args) -> int:
    from . import embed

    choice = os.environ.get("WEB_EXPRESS_MODEL", "xsmall").strip().lower()
    spec = embed.MODELS.get(choice)
    print(f"Data folder: {embed.data_home()}")
    if spec is None:
        print(f"Semantic ranking is switched off (WEB_EXPRESS_MODEL={choice}). Nothing to download.")
        return 0
    if embed.is_installed(spec):
        print(f"Ranking model {spec.name} is already installed.")
    else:
        print(f"Downloading ranking model {spec.name} (about 25 MB)...")
        try:
            embed.install(spec)
        except Exception as exc:
            print(f"Download failed: {exc}\nweb_express still works, with keyword ranking.", file=sys.stderr)
            return 1
    try:
        embedder = embed.Embedder(spec, embed.model_dir(spec))
        embedder.embed(["test"])
    except Exception as exc:
        print(f"The model is installed but could not be loaded: {exc}", file=sys.stderr)
        return 1
    print("Ranking model is ready.")
    return 0


def status() -> dict:
    from . import embed, index
    from .sources import default_sources

    choice = os.environ.get("WEB_EXPRESS_MODEL", "xsmall").strip().lower()
    spec = embed.MODELS.get(choice)
    report: dict = {
        "version": __version__,
        "launch_command": launch_command(),
        "data_folder": str(embed.data_home()),
        "ranking_model": "off" if spec is None else (f"{spec.name} (installed)" if embed.is_installed(spec) else f"{spec.name} (not downloaded yet)"),
        "sources": {s.name: "on" if s.available() else "off (needs WEB_EXPRESS_CONTACT)" for s in default_sources()},
        "packs": [],
        "site_ranking": "not installed",
        "settings": {name: "set" if os.environ.get(name, "").strip() else "not set"
                     for name in ("WEB_EXPRESS_CONTACT", "WEB_EXPRESS_GITHUB_TOKEN", "WEB_EXPRESS_OPENALEX_KEY")},
    }
    from . import authority, packs

    for pack in packs.installed():
        report["packs"].append(f"{pack.name} ({pack.info.get('pages', '?')} pages, {pack.info.get('licence', '')})")
        pack.close()
    ranks = authority.Authority()
    if ranks.available:
        details = ranks.info()
        report["site_ranking"] = f"{details.get('domains', '?')} sites from {details.get('release', '?')}"
    if not index.enabled():
        report["local_index"] = "off"
    elif not index.default_path().exists():
        report["local_index"] = "empty"
    else:
        try:
            store = index.LocalIndex()
            report["local_index"] = store.stats()
            store.close()
        except Exception as exc:
            report["local_index"] = f"unavailable: {exc}"
    return report


def cmd_status(args) -> int:
    _print_json(status())
    return 0


def cmd_init(args) -> int:
    clients = list(CLIENTS) if args.client == "all" else [args.client]
    results = init(clients, args.remove, args.dry_run)
    width = max(map(len, results))
    for client, outcome in results.items():
        print(f"{client.ljust(width)}  {outcome}")
    if not args.dry_run and not args.remove and any(o.startswith("registered") for o in results.values()):
        print("\nRestart those agents to load web_express.")
    problems = [o for o in results.values() if o.startswith(("failed", "skipped"))]
    if args.client != "all" and results[args.client] == "not installed":
        problems.append("not installed")  # the one agent asked for is not here
    return 1 if problems else 0


def cmd_search(args) -> int:
    from . import server

    async def run():
        try:
            return await server.search(args.query, category=args.category, max_results=args.max_results,
                                       contents=not args.links_only, freshness=args.freshness)
        finally:
            await server.flush()

    try:
        answer = asyncio.run(run())
    except (RuntimeError, ValueError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(answer)
        return 0
    print(f"sources: {', '.join(answer['sources_used'])} | ranking: {answer['ranking']}")
    for name, problem in answer.get("source_errors", {}).items():
        print(f"  ! {problem}")
    for number, result in enumerate(answer["results"], 1):
        print(f"\n{number}. {result['title']}\n   {result['url']}")
        if result.get("content_error"):
            print(f"   (could not read: {result['content_error']})")
        for highlight in result.get("highlights", []):
            print("   > " + " ".join(highlight["text"].split())[:300])
    return 0


def cmd_fetch(args) -> int:
    from . import server

    async def run():
        try:
            return await server.fetch(args.url, query=args.query, max_chars=args.max_chars, refresh=args.refresh)
        finally:
            await server.flush()

    try:
        answer = asyncio.run(run())
    except (RuntimeError, ValueError) as exc:
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(answer)
    elif "highlights" in answer:
        for highlight in answer["highlights"]:
            print("> " + highlight["text"] + "\n")
    else:
        print(answer["text"])
    return 0


def cmd_index(args) -> int:
    from . import index

    if not index.enabled():
        print("The local index is switched off (WEB_EXPRESS_INDEX=off).")
        return 0
    if args.action == "stats" and not index.default_path().exists():
        print("The local index is empty.")
        return 0
    store = index.LocalIndex()
    try:
        if args.action == "stats":
            _print_json(store.stats())
        elif args.action == "clear":
            pages = store.stats()["pages"]
            if not args.yes:
                print(f"This deletes all {pages} stored pages. Run again with --yes to confirm.")
                return 1
            store.clear()
            print(f"Deleted {pages} stored pages.")
        elif args.action == "remove":
            if not args.url:
                print("Give the URL to remove.", file=sys.stderr)
                return 1
            print("Removed." if store.remove(args.url) else "That page is not in the index.")
    finally:
        store.close()
    return 0


def _load_embedder():
    """The ranking model, loaded now (builds wait for it; searches never do). None if unavailable."""
    from . import embed
    from .server import _model_id

    spec = embed.MODELS.get(os.environ.get("WEB_EXPRESS_MODEL", "xsmall").strip().lower())
    if spec is None:
        return None, None
    try:
        embedder = embed.Embedder(spec, embed.install(spec))
    except Exception as exc:
        print(f"Building without search-by-meaning vectors: the model is unavailable ({exc}).", file=sys.stderr)
        return None, None
    return embedder, _model_id(embedder)


def _read_lines(path: str) -> list[str]:
    return [line.strip() for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def cmd_pack(args) -> int:
    from . import packs

    try:
        if args.action == "list":
            found = packs.installed()
            if not found:
                print("No packs are installed.")
            for pack in found:
                print(f"{pack.name}: {pack.info.get('title', '')} | {pack.info.get('pages', '?')} pages | "
                      f"{pack.info.get('licence', '')} | {pack.info.get('attribution', '')}")
                pack.close()
        elif args.action == "install":
            info = packs.install(args.target, sha256=args.sha256, replace=args.replace)
            print(f"Installed {info['name']}: {info.get('pages', '?')} pages, licence {info.get('licence', '')}.")
            print(f"sha256 {info['sha256']}")
        elif args.action == "remove":
            print("Removed." if packs.remove(args.target) else "No pack of that name is installed.")
        elif args.action == "build":
            for needed in ("urls", "title", "licence", "attribution"):
                if not getattr(args, needed):
                    print(f"pack build needs --{needed}.", file=sys.stderr)
                    return 1
            embedder, model_id = _load_embedder()
            output = Path(args.output or f"{args.target}.pack").expanduser()
            built = asyncio.run(packs.build(
                args.target, _read_lines(args.urls), output, title=args.title, licence=args.licence,
                attribution=args.attribution, source=args.source, embedder=embedder, model_id=model_id,
                progress=lambda number, url: print(f"  {number}: {url}"),
            ))
            print(f"Wrote {built['file']}: {built['pages']} pages.\nsha256 {built['sha256']}")
            for url, reason in built["failed"].items():
                print(f"  skipped {url}: {reason}")
    except (packs.PackError, OSError) as exc:
        print(f"Pack {args.action} failed: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_authority(args) -> int:
    from . import authority

    if args.action == "install":
        print(f"Downloading the {args.top:,} best-linked sites from Common Crawl's web graph...")
        try:
            info = authority.install(top=args.top, progress=lambda count: print(f"  {count:,}", end="\r"))
        except Exception as exc:
            print(f"Install failed: {exc}", file=sys.stderr)
            return 1
        print(f"Stored {info['domains']:,} sites from release {info['release']}.")
        return 0
    ranks = authority.Authority()
    if not ranks.available:
        print("Site-ranking data is not installed. Run: web-express authority install")
        return 0
    _print_json(ranks.info())
    return 0


def cmd_community(args) -> int:
    from . import authority, community, index, packs

    try:
        if args.action == "export":
            if not index.enabled() or not index.default_path().exists():
                print("The local index is empty; there is nothing to propose.")
                return 0
            store = index.LocalIndex()
            try:
                count = community.export_nominations(store, Path(args.paths[0]).expanduser(), args.limit)
            finally:
                store.close()
            print(f"Wrote {count} addresses to {args.paths[0]}. Nothing was sent anywhere; share the file if you choose to.")
        elif args.action == "select":
            if not args.output:
                print("community select needs --output.", file=sys.stderr)
                return 1
            blocked = set(_read_lines(args.blocked)) if args.blocked else set()
            ranks = authority.Authority()
            chosen, report = community.select(
                [Path(p).expanduser() for p in args.paths], quorum=args.quorum, per_site=args.per_site,
                authority=ranks, max_site_rank=args.max_site_rank, blocked=blocked,
            )
            Path(args.output).expanduser().write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")
            if not ranks.available:
                report["note"] = "site-ranking data is not installed, so the site-rank rule was not applied"
            _print_json(report)
        elif args.action == "build":
            for needed in ("name", "licence", "attribution", "output"):
                if not getattr(args, needed):
                    print(f"community build needs --{needed}.", file=sys.stderr)
                    return 1
            embedder, model_id = _load_embedder()
            result = asyncio.run(community.build_shards(
                _read_lines(args.paths[0]), Path(args.output).expanduser(), name=args.name, licence=args.licence,
                attribution=args.attribution, shard_size=args.shard_size, embedder=embedder, model_id=model_id,
                rules={"quorum": args.quorum, "per_site": args.per_site, "max_site_rank": args.max_site_rank},
                progress=lambda number, url: print(f"  {number}: {url}"),
            ))
            print(f"Built {len(result['shards'])} shards, {result['pages']} pages, {result['unreadable']} unreadable. "
                  f"Manifest: {Path(args.output).expanduser() / 'manifest.json'}")
        elif args.action == "install":
            names = community.install_from_manifest(Path(args.paths[0]).expanduser())
            print(f"Installed {len(names)} shards: {', '.join(names)}")
    except (community.CommunityError, packs.PackError, OSError) as exc:
        print(f"community {args.action} failed: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web-express", description="Free, local web search for AI agents.")
    parser.add_argument("--version", action="version", version=f"web-express {__version__}")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("serve", help="run the MCP server (the default)")
    commands.add_parser("setup", help="download the ranking model now")
    commands.add_parser("status", help="show what is installed and configured")

    init_parser = commands.add_parser("init", help="register web_express with the AI agents installed here")
    init_parser.add_argument("--client", choices=(*CLIENTS, "all"), default="all")
    init_parser.add_argument("--remove", action="store_true", help="unregister instead")
    init_parser.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")

    search = commands.add_parser("search", help="run a search from the terminal")
    search.add_argument("query")
    search.add_argument("--category", default="auto", choices=("auto", "web", "papers", "code", "news"))
    search.add_argument("--max-results", type=int, default=5)
    search.add_argument("--freshness", default="", choices=("", "day", "week", "month", "year"))
    search.add_argument("--links-only", action="store_true", help="do not read the result pages")
    search.add_argument("--json", action="store_true")

    fetch = commands.add_parser("fetch", help="read a page from the terminal")
    fetch.add_argument("url")
    fetch.add_argument("--query", default="", help="show only the passages matching these words")
    fetch.add_argument("--max-chars", type=int, default=12000)
    fetch.add_argument("--refresh", action="store_true")
    fetch.add_argument("--json", action="store_true")

    index = commands.add_parser("index", help="inspect or clear the local index")
    index.add_argument("action", choices=("stats", "clear", "remove"))
    index.add_argument("url", nargs="?")
    index.add_argument("--yes", action="store_true", help="confirm clear")

    pack = commands.add_parser("pack", help="build, install, list or remove corpus packs")
    pack.add_argument("action", choices=("list", "install", "remove", "build"))
    pack.add_argument("target", nargs="?", default="", help="pack name, or the file or https address to install")
    pack.add_argument("--sha256", help="install: refuse the pack unless it has this checksum")
    pack.add_argument("--replace", action="store_true", help="install: replace an installed pack of the same name")
    pack.add_argument("--urls", help="build: a file with one address per line")
    pack.add_argument("--title", default="")
    pack.add_argument("--licence", default="", help="build: the licence the pages may be shared under")
    pack.add_argument("--attribution", default="", help="build: who to credit")
    pack.add_argument("--source", default="")
    pack.add_argument("--output", help="build: where to write the pack file")

    ranks = commands.add_parser("authority", help="install or inspect the site-ranking data")
    ranks.add_argument("action", choices=("install", "status"))
    ranks.add_argument("--top", type=int, default=1_000_000, help="how many of the best-linked sites to keep")

    shared = commands.add_parser("community", help="propose addresses and build or install shared shards")
    shared.add_argument("action", choices=("export", "select", "build", "install"))
    shared.add_argument("paths", nargs="+", help="export: output file; select: nomination files; build: address file; install: manifest")
    shared.add_argument("--limit", type=int, default=5000, help="export: most addresses to write")
    shared.add_argument("--output", help="select: address file to write; build: folder for shards")
    shared.add_argument("--quorum", type=int, default=2, help="contributors that must propose an address")
    shared.add_argument("--per-site", type=int, default=200, help="most addresses accepted from one site")
    shared.add_argument("--max-site-rank", type=int, default=1_000_000)
    shared.add_argument("--blocked", help="select: file of addresses or sites never to include")
    shared.add_argument("--name", default="")
    shared.add_argument("--licence", default="")
    shared.add_argument("--attribution", default="")
    shared.add_argument("--shard-size", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "serve"):
        from . import server

        server.main()
        return 0
    handler = {"setup": cmd_setup, "status": cmd_status, "init": cmd_init, "search": cmd_search,
               "fetch": cmd_fetch, "index": cmd_index, "pack": cmd_pack, "authority": cmd_authority,
               "community": cmd_community}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
