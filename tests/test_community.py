import json

import pytest

from web_express import community, packs
from web_express.extract import Document
from web_express.index import LocalIndex
from web_express.net import FetchError

TEXT = "A real article with enough words to be worth keeping in an index. " * 6


def doc(url):
    return Document(url, "Title", TEXT, None, "plain", False)


async def loader(url):
    if "unreadable" in url:
        raise FetchError("Page answered HTTP 404.")
    return doc(url)


def write(tmp_path, name, urls):
    path = tmp_path / name
    path.write_text("\n".join(urls) + "\n")
    return path


def test_only_public_credential_free_addresses_can_be_proposed():
    good = ["https://example.org/a", "https://docs.example.co.uk/x?page=2"]
    bad = ["http://example.org/a", "https://localhost/x", "https://intranet/x", "https://wiki.corp/x", "https://10.0.0.5/x",
           "https://[::1]/x", "https://example.org/x?token=abc", "https://user:pw@example.org/", "ftp://example.org/x",
           "https://example.org/" + "a" * 600, "not a url", ""]
    assert all(community.nominable(u) for u in good) and not any(community.nominable(u) for u in bad)


def test_export_lists_addresses_only_shuffled_and_filtered(tmp_path):
    index = LocalIndex(tmp_path / "index.sqlite3")
    for url in ["https://example.org/a", "https://example.org/b", "http://plain.example.org/c", "https://wiki.corp/secret"]:
        index.add(doc(url))
    out = tmp_path / "share" / "nominations.txt"
    assert community.export_nominations(index, out) == 2
    assert sorted(out.read_text().split()) == ["https://example.org/a", "https://example.org/b"]
    assert "Title" not in out.read_text() and TEXT[:20] not in out.read_text()
    assert community.export_nominations(index, out, limit=1) == 1
    index.close()


def test_selection_needs_several_contributors_and_caps_each_site(tmp_path):
    a = write(tmp_path, "a.txt", ["https://one.example/1", "https://one.example/2", "https://one.example/3",
                                  "https://two.example/1", "https://solo.example/1", "https://blocked.example/1",
                                  "https://one.example/1", "garbage", "https://x.example/?token=1"])
    b = write(tmp_path, "b.txt", ["https://one.example/1", "https://one.example/2", "https://one.example/3",
                                  "https://two.example/1", "https://blocked.example/1"])
    chosen, report = community.select([a, b], quorum=2, per_site=2, blocked={"blocked.example"})
    assert chosen == ["https://one.example/1", "https://one.example/2", "https://two.example/1"]
    assert report == {"proposed": 6, "below_quorum": 1, "blocked": 1, "unranked_site": 0, "site_limit": 1,
                      "selected": 3, "sites": 2}
    # repeating an address inside one file is still one vote
    assert community.select([a], quorum=2)[0] == []


def test_selection_can_require_a_well_linked_site(tmp_path):
    class Ranks:
        available = True

        def position(self, url):
            return {"one.example": 50, "two.example": 900_000}.get(url.split("/")[2])

    a = write(tmp_path, "a.txt", ["https://one.example/1", "https://two.example/1", "https://nobody.example/1"])
    chosen, report = community.select([a], quorum=1, authority=Ranks(), max_site_rank=100_000)
    assert chosen == ["https://one.example/1"] and report["unranked_site"] == 2


async def test_shards_manifest_and_verified_install(tmp_path):
    urls = [f"https://site{i}.example/page" for i in range(5)] + ["https://site9.example/unreadable"]
    out = tmp_path / "release"
    result = await community.build_shards(urls, out, name="pilot", licence="CC0-1.0", attribution="Pilot builders",
                                          shard_size=3, rules={"quorum": 2}, delay=0, loader=loader)
    assert [s["pages"] for s in result["shards"]] == [3, 2] and result["pages"] == 5 and result["unreadable"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rules"] == {"quorum": 2} and [s["name"] for s in manifest["shards"]] == ["pilot-0001", "pilot-0002"]
    assert "failed" not in manifest  # addresses that failed are not published

    assert community.install_from_manifest(out / "manifest.json") == ["pilot-0001", "pilot-0002"]
    assert sorted(p.name for p in packs.installed()) == ["pilot-0001", "pilot-0002"]
    for pack in packs.installed():
        packs.remove(pack.name)

    # a shard swapped after the manifest was written is refused, and nothing is installed
    (out / "pilot-0002.pack").write_bytes((out / "pilot-0001.pack").read_bytes())
    with pytest.raises(community.CommunityError, match="does not match"):
        community.install_from_manifest(out / "manifest.json")
    assert packs.installed() == []


async def test_bad_manifests_are_refused(tmp_path):
    for content in ("not json", "[]", '{"manifest_version": 99, "shards": []}',
                    '{"manifest_version": 1, "shards": [{"name": "../x", "file": "../x.pack", "sha256": "0"}]}',
                    '{"manifest_version": 1, "shards": [{"name": "ok", "file": "other.pack", "sha256": "0"}]}'):
        path = tmp_path / "manifest.json"
        path.write_text(content)
        with pytest.raises(community.CommunityError):
            community.install_from_manifest(path)
    with pytest.raises(community.CommunityError, match="No addresses"):
        await community.build_shards([], tmp_path / "r", name="pilot", licence="l", attribution="a")
    with pytest.raises(community.CommunityError, match="No shard could be built"):
        await community.build_shards(["https://a.example/unreadable"], tmp_path / "r", name="pilot", licence="l",
                                     attribution="a", delay=0, loader=loader)
