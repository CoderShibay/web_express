import sqlite3

import httpx
import numpy as np
import pytest

from web_express import packs
from web_express.extract import Document
from web_express.net import FetchError

PAGES = {
    "https://docs.test/a": Document("https://docs.test/a", "Timeouts", "Use asyncio.timeout to cancel slow work. " * 8, None, "plain", False),
    "https://docs.test/b": Document("https://docs.test/b", "Queues", "A queue passes items between tasks. " * 8, None, "plain", False),
}


async def loader(url):
    if url not in PAGES:
        raise FetchError("Page answered HTTP 404.")
    return PAGES[url]


class TinyEmbedder:
    def embed(self, texts):
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row] = [1, 0] if "timeout" in text.lower() or "slow" in text.lower() else [0, 1]
        return out


async def make(tmp_path, **overrides):
    options = dict(title="Test docs", licence="CC0-1.0", attribution="The Test Project", source="https://docs.test/",
                   delay=0, loader=loader)
    options.update(overrides)
    return await packs.build("test-docs", [*PAGES, "https://docs.test/missing"], tmp_path / "test-docs.pack", **options)


async def test_build_install_search_and_remove(tmp_path):
    built = await make(tmp_path, embedder=TinyEmbedder(), model_id="tiny@1")
    assert built["pages"] == "2" and built["failed"] == {"https://docs.test/missing": "Page answered HTTP 404."}
    assert [p.name for p in tmp_path.iterdir()] == ["test-docs.pack"]  # no temporary files left behind

    installed = packs.install(str(tmp_path / "test-docs.pack"), sha256=built["sha256"])
    assert installed["licence"] == "CC0-1.0" and installed["name"] == "test-docs"
    (pack,) = packs.installed()
    assert pack.info["attribution"] == "The Test Project"
    assert [p.url for p in pack.search("queue")] == ["https://docs.test/b"]
    by_meaning = pack.search("stop something that hangs", query_vector=np.array([1, 0], dtype=np.float32), model="tiny@1")
    assert [p.url for p in by_meaning] == ["https://docs.test/a"]
    with pytest.raises(packs.PackError, match="read-only"):
        pack.add(PAGES["https://docs.test/a"])
    with pytest.raises(sqlite3.Error):
        pack._db.execute("DELETE FROM pages")
    pack.close()
    assert packs.remove("test-docs") and not packs.remove("test-docs") and packs.installed() == []


async def test_a_pack_must_state_its_licence_and_have_a_sane_name(tmp_path):
    with pytest.raises(packs.PackError, match="licence"):
        await make(tmp_path, licence=" ")
    for name in ("", "Has Spaces", "../escape", "x" * 60):
        with pytest.raises(packs.PackError, match="pack name"):
            await packs.build(name, list(PAGES), tmp_path / "x.pack", title="t", licence="l", attribution="a", loader=loader)
    with pytest.raises(packs.PackError, match="None of the URLs"):
        await packs.build("empty", ["https://docs.test/missing"], tmp_path / "e.pack", title="t", licence="l",
                          attribution="a", delay=0, loader=loader)
    assert list(tmp_path.iterdir()) == []


async def test_install_checks_the_checksum_and_refuses_tampered_files(tmp_path):
    built = await make(tmp_path)
    path = tmp_path / "test-docs.pack"
    with pytest.raises(packs.PackError, match="checksum"):
        packs.install(str(path), sha256="0" * 64)

    (tmp_path / "junk.pack").write_bytes(b"not a database at all")
    with pytest.raises(packs.PackError, match="not a web_express pack"):
        packs.install(str(tmp_path / "junk.pack"))

    for statement in ("CREATE TRIGGER evil AFTER INSERT ON pages BEGIN DELETE FROM pages; END",
                      "CREATE VIEW sneaky AS SELECT * FROM pages", "CREATE TABLE extra (x)"):
        copy = tmp_path / "tampered.pack"
        copy.write_bytes(path.read_bytes())
        db = sqlite3.connect(copy)
        db.execute(statement)
        db.commit()
        db.close()
        with pytest.raises(packs.PackError, match="unexpected"):
            packs.install(str(copy))

    renamed = tmp_path / "renamed.pack"
    renamed.write_bytes(path.read_bytes())
    db = sqlite3.connect(renamed)
    db.execute("UPDATE pack_info SET value = '../../outside' WHERE key = 'name'")
    db.commit()
    db.close()
    with pytest.raises(packs.PackError, match="pack name"):
        packs.install(str(renamed))
    assert packs.installed() == [] and not any(p.name.startswith(".incoming") for p in packs.packs_dir().iterdir())
    assert built["sha256"] == packs.file_sha256(path)


async def test_install_from_https_only(tmp_path):
    await make(tmp_path)
    body = (tmp_path / "test-docs.pack").read_bytes()
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=body))
    assert packs.install("https://packs.test/test-docs.pack", transport=transport)["name"] == "test-docs"
    with pytest.raises(packs.PackError, match="https"):
        packs.install("http://packs.test/test-docs.pack", transport=transport)
    with pytest.raises(packs.PackError, match="Could not download"):
        packs.install("https://packs.test/x.pack", transport=httpx.MockTransport(lambda r: httpx.Response(404)))
