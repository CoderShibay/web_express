import pytest

from web_express import embed, server


@pytest.fixture(autouse=True)
async def isolated(tmp_path, monkeypatch):
    """No test touches the real data folder or downloads the real model."""
    monkeypatch.setenv("WEB_EXPRESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEB_EXPRESS_MODEL", "off")
    embed.reset()
    monkeypatch.setattr(server, "_index", None)
    monkeypatch.setattr(server, "_index_failed", False)
    yield
    await server.flush()
    if server._index is not None:
        server._index.close()
    embed.reset()
