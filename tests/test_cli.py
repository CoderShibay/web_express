import json
import tomllib

import pytest

from web_express import cli

COMMAND = ["/opt/wx/bin/web-express"]


def test_json_config_is_created_updated_and_backed_up(tmp_path):
    path = tmp_path / "app" / "mcp.json"
    assert cli.register_json(path, COMMAND) == "not installed"  # the agent's folder does not exist
    path.parent.mkdir()
    assert cli.register_json(path, COMMAND).startswith("registered in")
    assert json.loads(path.read_text()) == {"mcpServers": {"web_express": {"command": COMMAND[0], "args": []}}}
    assert cli.register_json(path, COMMAND) == "already registered"

    path.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}))
    message = cli.register_json(path, ["python", "-m", "web_express"])
    data = json.loads(path.read_text())
    assert data["theme"] == "dark" and data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["web_express"] == {"command": "python", "args": ["-m", "web_express"]}
    assert "backup:" in message and len(list(path.parent.glob("mcp.json.bak-web_express-*"))) == 1


def test_json_dry_run_remove_and_unreadable_files(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text('{"mcpServers": {}}')
    assert cli.register_json(path, COMMAND, dry_run=True).startswith("would update")
    assert json.loads(path.read_text()) == {"mcpServers": {}}
    assert cli.register_json(path, COMMAND, remove=True) == "not registered"
    cli.register_json(path, COMMAND)
    assert cli.register_json(path, COMMAND, remove=True).startswith("removed from")
    assert json.loads(path.read_text())["mcpServers"] == {}

    for broken in ("{ // comment\n}", "[1, 2]", '{"mcpServers": []}'):
        path.write_text(broken)
        assert cli.register_json(path, COMMAND).startswith("skipped") and path.read_text() == broken


def test_codex_toml_keeps_everything_else(tmp_path):
    path = tmp_path / "config.toml"
    original = 'model = "gpt"\n\n[mcp_servers.other]\nurl = "http://127.0.0.1:1/mcp"\n\n[profiles.fast]\nmodel = "x"\n'
    path.write_text(original)
    assert cli.register_codex(path, ['C:\\Program Files\\wx\\web-express.exe', "serve"]).startswith("registered in")
    data = tomllib.loads(path.read_text())
    assert data["mcp_servers"]["web_express"] == {"command": 'C:\\Program Files\\wx\\web-express.exe', "args": ["serve"]}
    assert data["mcp_servers"]["other"]["url"] and data["profiles"]["fast"]["model"] == "x" and data["model"] == "gpt"
    assert cli.register_codex(path, ['C:\\Program Files\\wx\\web-express.exe', "serve"]) == "already registered"

    cli.register_codex(path, COMMAND)  # a new install location replaces the old entry
    assert tomllib.loads(path.read_text())["mcp_servers"]["web_express"] == {"command": COMMAND[0], "args": []}
    assert cli.register_codex(path, COMMAND, remove=True).startswith("removed from")
    assert tomllib.loads(path.read_text()) == tomllib.loads(original)
    assert cli.register_codex(path, COMMAND, remove=True) == "not registered"


def test_codex_entry_in_the_middle_of_the_file_is_replaced_in_place(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[mcp_servers.web_express]\ncommand = "old"\nargs = []\n\n[mcp_servers.other]\nurl = "u"\n')
    cli.register_codex(path, COMMAND)
    data = tomllib.loads(path.read_text())
    assert data["mcp_servers"]["web_express"]["command"] == COMMAND[0] and data["mcp_servers"]["other"] == {"url": "u"}


def test_codex_refuses_files_it_cannot_edit_safely(tmp_path):
    path = tmp_path / "config.toml"
    for broken in ("not = [valid", 'mcp_servers = { web_express = { command = "x" } }\n'):
        path.write_text(broken)
        assert cli.register_codex(path, COMMAND).startswith("skipped") and path.read_text() == broken


def test_init_only_touches_agents_that_are_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_home", lambda: tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    (tmp_path / ".cursor").mkdir()
    results = cli.init(list(cli.CLIENTS), remove=False, dry_run=False)
    assert results["cursor"].startswith("registered in")
    assert {results[c] for c in ("claude-code", "claude-desktop", "codex", "gemini", "antigravity")} == {"not installed"}
    assert sorted(p.name for p in tmp_path.iterdir()) == [".cursor"]


def test_status_and_index_commands(capsys, monkeypatch):
    report = cli.status()
    assert report["local_index"] == "empty" and report["ranking_model"] == "off"
    assert report["sources"]["wikipedia"].startswith("off") and report["settings"]["WEB_EXPRESS_GITHUB_TOKEN"] == "not set"
    monkeypatch.setenv("WEB_EXPRESS_GITHUB_TOKEN", "ghp_secret")
    assert "ghp_secret" not in json.dumps(cli.status())

    assert cli.main(["index", "stats"]) == 0 and "empty" in capsys.readouterr().out
    assert cli.main(["index", "clear"]) == 1 and "--yes" in capsys.readouterr().out
    assert cli.main(["index", "clear", "--yes"]) == 0
    assert cli.main(["setup"]) == 0 and "switched off" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["search"])


def test_rewritten_configs_keep_their_permissions(tmp_path):
    import os
    import stat

    if os.name == "nt":
        pytest.skip("POSIX permissions")
    for name, register in (("mcp.json", cli.register_json), ("config.toml", cli.register_codex)):
        path = tmp_path / name
        path.write_text("{}" if name.endswith("json") else "")
        path.chmod(0o600)
        old = os.umask(0o022)
        try:
            assert register(path, COMMAND).startswith("registered")
        finally:
            os.umask(old)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        backups = list(tmp_path.glob(name + ".bak-web_express-*"))
        assert backups and all(stat.S_IMODE(b.stat().st_mode) == 0o600 for b in backups)
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_backups_made_in_the_same_second_do_not_overwrite_each_other(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text('{"original": true}')
    cli.register_json(path, COMMAND)
    cli.register_json(path, ["other"])
    cli.register_json(path, COMMAND)
    backups = sorted(tmp_path.glob("mcp.json.bak-web_express-*"))
    assert len(backups) == 3 and any(json.loads(b.read_text()) == {"original": True} for b in backups)


def test_json_keeps_extra_settings_and_a_byte_order_mark(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"mcpServers": {"web_express": {"command": "old", "env": {"K": "v"}}}}).encode())
    assert cli.register_json(path, COMMAND).startswith("registered in")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert json.loads(raw[3:])["mcpServers"]["web_express"] == {"command": COMMAND[0], "env": {"K": "v"}, "args": []}


def test_codex_update_keeps_other_settings_comments_and_line_endings(tmp_path):
    path = tmp_path / "config.toml"
    original = ('# my settings\r\nmodel = "gpt"\r\n\r\n[mcp_servers.web_express]  # search\r\ncommand = "old"\r\nargs = []\r\n'
                'startup_timeout_sec = 30\r\nenv = { TOKEN = "secret" }\r\n# keep me\r\n\r\n[mcp_servers.web_express.extra]\r\nx = 1\r\n')
    path.write_bytes(original.encode())
    assert cli.register_codex(path, COMMAND).startswith("registered in")
    text = path.read_bytes().decode()
    table = tomllib.loads(text)["mcp_servers"]["web_express"]
    assert table == {"command": COMMAND[0], "args": [], "startup_timeout_sec": 30, "env": {"TOKEN": "secret"}, "extra": {"x": 1}}
    assert "# my settings" in text and "# keep me" in text and "# search" in text and "\n" not in text.replace("\r\n", "")


def test_codex_remove_takes_sub_tables_too(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[mcp_servers.other]\nurl = "u"\n\n[mcp_servers.web_express]\ncommand = "x"\nargs = []\n\n'
                    '[mcp_servers.web_express.env]\nTOKEN = "secret"\n\n[profiles.fast]\nmodel = "m"\n')
    assert cli.register_codex(path, COMMAND, remove=True).startswith("removed from")
    data = tomllib.loads(path.read_text())
    assert data == {"mcp_servers": {"other": {"url": "u"}}, "profiles": {"fast": {"model": "m"}}}
    assert "secret" not in path.read_text()


def test_codex_refuses_layouts_it_cannot_edit_exactly(tmp_path):
    path = tmp_path / "config.toml"
    cases = ['mcp_servers = 42\n',
             '[mcp_servers]\nweb_express = 7\n',
             '[mcp_servers.web_express]\ncommand = "old"\nargs = [\n  "a",\n  "b",\n]\n',  # value over several lines
             '[mcp_servers]\nweb_express.command = "old"\n']                               # dotted keys
    for content in cases:
        path.write_text(content)
        assert cli.register_codex(path, COMMAND).startswith("skipped") and path.read_text() == content
    assert not list(tmp_path.glob("*.bak-*"))


def test_init_exit_code_reports_problems(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_home", lambda: tmp_path)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["init"]) == 0  # nothing installed is not an error when asking for all
    assert cli.main(["init", "--client", "cursor"]) == 1  # but the one asked for is missing
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".cursor" / "mcp.json").write_text("{ // comments\n}")
    assert cli.main(["init", "--client", "cursor"]) == 1 and "skipped" in capsys.readouterr().out
    (tmp_path / ".cursor" / "mcp.json").write_text("{}")
    assert cli.main(["init", "--client", "cursor"]) == 0


def test_pack_and_community_commands_work_end_to_end(tmp_path, monkeypatch, capsys):
    from web_express import packs
    from web_express.extract import Document
    from web_express.index import LocalIndex, default_path

    text = "A real article with enough words to be worth keeping in an index. " * 6

    async def fake_download(url, **kwargs):
        return Document(url, "Title", text, None, "plain", False)

    monkeypatch.setattr(packs, "download", fake_download)
    monkeypatch.setattr(packs, "extract", lambda document: document)
    monkeypatch.setattr(packs.asyncio, "sleep", lambda seconds: _nothing())

    urls = tmp_path / "urls.txt"
    urls.write_text("# my list\nhttps://one-example.org/a\nhttps://two-example.org/b\n")
    out = tmp_path / "demo.pack"
    assert cli.main(["pack", "build", "demo"]) == 1  # no licence, no pack
    assert cli.main(["pack", "build", "demo", "--urls", str(urls), "--title", "Demo", "--licence", "CC0-1.0",
                     "--attribution", "Me", "--output", str(out)]) == 0
    assert cli.main(["pack", "install", str(out), "--sha256", "0" * 64]) == 1
    assert cli.main(["pack", "install", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["pack", "list"]) == 0 and "demo: Demo | 2 pages | CC0-1.0 | Me" in capsys.readouterr().out
    assert cli.status()["packs"] == ["demo (2 pages, CC0-1.0)"]
    assert cli.main(["pack", "remove", "demo"]) == 0 and packs.installed() == []

    store = LocalIndex(default_path())
    for url in ("https://one-example.org/a", "https://two-example.org/b", "https://wiki.corp/private"):
        store.add(Document(url, "Title", text, None, "plain", False))
    store.close()
    mine, theirs, chosen = tmp_path / "mine.txt", tmp_path / "theirs.txt", tmp_path / "chosen.txt"
    assert cli.main(["community", "export", str(mine)]) == 0
    assert sorted(mine.read_text().split()) == ["https://one-example.org/a", "https://two-example.org/b"]
    theirs.write_text("https://one-example.org/a\n")
    assert cli.main(["community", "select", str(mine), str(theirs), "--output", str(chosen)]) == 0
    assert chosen.read_text().split() == ["https://one-example.org/a"]
    release = tmp_path / "release"
    assert cli.main(["community", "build", str(chosen), "--name", "pilot", "--licence", "CC0-1.0",
                     "--attribution", "Us", "--output", str(release)]) == 0
    assert cli.main(["community", "install", str(release / "manifest.json")]) == 0
    assert [p.name for p in packs.installed()] == ["pilot-0001"]
    assert cli.main(["community", "install", str(tmp_path / "missing.json")]) == 1
    assert cli.main(["authority", "status"]) == 0 and "not installed" in capsys.readouterr().out


async def _nothing():
    return None
