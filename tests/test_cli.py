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
