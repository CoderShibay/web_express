"""The web-express command.

    web-express                 run the MCP server (what an agent launches)
    web-express setup           download the ranking model now, instead of on first search
    web-express init            register web_express with the AI agents installed here
    web-express status          show what is installed and configured
    web-express search QUERY    run a search from the terminal
    web-express fetch URL       read a page from the terminal
    web-express index ...       inspect or clear the local index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
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
    copy = path.with_name(f"{path.name}.bak-web_express-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, copy)
    return copy


def _write(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".web_express-tmp")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def register_json(path: Path, command: list[str], remove: bool = False, dry_run: bool = False) -> str:
    """Add or remove the server in a JSON MCP config. Returns a line describing what happened."""
    if not path.parent.is_dir():
        return "not installed"
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except ValueError:
            return f"skipped: {path} is not plain JSON; add the server by hand"
        if not isinstance(data, dict):
            return f"skipped: {path} has an unexpected layout"
    servers = data.get("mcpServers")
    if servers is None:
        servers = {}
    if not isinstance(servers, dict):
        return f"skipped: {path} has an unexpected layout"
    entry = {"command": command[0], "args": command[1:]}
    if remove:
        if SERVER_NAME not in servers:
            return "not registered"
        del servers[SERVER_NAME]
    elif servers.get(SERVER_NAME) == entry:
        return "already registered"
    else:
        servers[SERVER_NAME] = entry
    data["mcpServers"] = servers
    if dry_run:
        return f"would {'remove from' if remove else 'update'} {path}"
    backup = f" (backup: {_backup(path).name})" if path.is_file() else ""
    _write(path, json.dumps(data, indent=2) + "\n")
    return f"{'removed from' if remove else 'registered in'} {path}{backup}"


def _toml_string(value: str) -> str:
    return json.dumps(value)  # JSON string syntax is valid TOML basic-string syntax


def register_codex(path: Path, command: list[str], remove: bool = False, dry_run: bool = False) -> str:
    """Add or remove the [mcp_servers.web_express] table in Codex's config.toml."""
    import tomllib

    if not path.parent.is_dir():
        return "not installed"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    try:
        current = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return f"skipped: {path} is not valid TOML; add the server by hand"
    header = f"[mcp_servers.{SERVER_NAME}]"
    present = SERVER_NAME in (current.get("mcp_servers") or {})
    block = f"{header}\ncommand = {_toml_string(command[0])}\nargs = [{', '.join(_toml_string(a) for a in command[1:])}]\n"
    if present:
        lines = text.splitlines(keepends=True)
        start = next((i for i, line in enumerate(lines) if line.strip() == header), None)
        if start is None:
            return f"skipped: {path} defines {SERVER_NAME} in a form web-express will not rewrite; edit it by hand"
        end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
        existing = current["mcp_servers"][SERVER_NAME]
        if not remove and existing == {"command": command[0], "args": command[1:]}:
            return "already registered"
        new_text = "".join(lines[:start]) + ("" if remove else block + ("\n" if end < len(lines) else "")) + "".join(lines[end:])
    elif remove:
        return "not registered"
    else:
        new_text = text + ("" if not text or text.endswith("\n\n") else "\n" if text.endswith("\n") else "\n\n") + block
    try:
        tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:
        return f"skipped: could not edit {path} safely; add the server by hand"
    if dry_run:
        return f"would {'remove from' if remove else 'update'} {path}"
    backup = f" (backup: {_backup(path).name})" if path.is_file() else ""
    _write(path, new_text)
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
        "settings": {name: "set" if os.environ.get(name, "").strip() else "not set"
                     for name in ("WEB_EXPRESS_CONTACT", "WEB_EXPRESS_GITHUB_TOKEN", "WEB_EXPRESS_OPENALEX_KEY")},
    }
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
    return 0


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "serve"):
        from . import server

        server.main()
        return 0
    handler = {"setup": cmd_setup, "status": cmd_status, "init": cmd_init, "search": cmd_search,
               "fetch": cmd_fetch, "index": cmd_index}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
