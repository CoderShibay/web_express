import pytest

from web_express import embed


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test touches the real data folder or downloads the real model."""
    monkeypatch.setenv("WEB_EXPRESS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WEB_EXPRESS_MODEL", "off")
    embed.reset()
    yield
    embed.reset()
