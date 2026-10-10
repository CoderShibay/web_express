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


def test_only_plain_public_pages_can_be_proposed(monkeypatch):
    good = ["https://example.org/a", "https://docs.example.co.uk/guide/install.html", "https://arxiv.org/abs/1909.06684",
            "https://en.wikipedia.org/wiki/Wilms_tumor"]
    bad = [
        "http://example.org/a", "https://localhost/x", "https://intranet/x", "https://wiki.corp/x", "https://10.0.0.5/x",
        "https://[::1]/x", "https://user:pw@example.org/", "ftp://example.org/x", "https://example.org:8443/x",
        "https://127.0.0.1.nip.io/private", "https://10-0-0-5.sslip.io/x", "https://example.test/x",
        "https://example.org/x?token=abc", "https://example.org/search?q=rare+medical+diagnosis", "https://example.org/a?page=2",
        "https://example.org/reset/VERY_SECRET_TOKEN", "https://example.org/reset/abc", "https://docs.example.org/document/personal-document-id",
        "https://example.org/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit", "https://example.org/u/alice",
        "https://example.org/invite/x", "https://example.org/files/550e8400-e29b-41d4-a716-446655440000",
        "https://example.org/orders/12345678", "https://example.org/a/deadbeefdeadbeef", "https://example.org/a%20b",
        "https://example.org/" + "a" * 600, "not a url", "",
    ]
    assert [u for u in good if not community.nominable(u)] == []
    assert [u for u in bad if community.nominable(u)] == []
    monkeypatch.setenv("WEB_EXPRESS_PRIVATE_DOMAINS", "employee-company.com, example.co.uk")
    assert not community.nominable("https://jira.employee-company.com/browse/PRIVATE-7")
    assert not community.nominable("https://docs.example.co.uk/guide/install.html")
    assert community.nominable("https://example.org/a")


def test_export_lists_addresses_only_shuffled_and_filtered(tmp_path):
    index = LocalIndex(tmp_path / "index.sqlite3")
    for url in ["https://example.org/a", "https://example.org/b", "http://plain.example.org/c", "https://wiki.corp/secret",
                "https://example.org/search?q=my+illness", "https://unranked-example.net/a"]:
        index.add(doc(url))
    out = tmp_path / "share" / "nominations.txt"
    assert community.export_nominations(index, out) == 3
    assert sorted(out.read_text().split()) == ["https://example.org/a", "https://example.org/b", "https://unranked-example.net/a"]
    assert "Title" not in out.read_text() and TEXT[:20] not in out.read_text()

    class Ranks:
        available = True

        def position(self, url):
            return 5 if "//example.org/" in url else None

    assert community.export_nominations(index, out, authority=Ranks()) == 2  # only well-linked public sites
    assert community.export_nominations(index, out, limit=1) == 1
    index.close()


def test_selection_needs_several_distinct_files_and_caps_each_site(tmp_path):
    a = write(tmp_path, "a.txt", ["https://one-example.org/1", "https://one-example.org/2", "https://one-example.org/3",
                                  "https://two-example.org/1", "https://solo-example.org/1", "https://blocked-example.org/1",
                                  "https://one-example.org/1", "garbage", "https://x-example.org/?token=1"])
    b = write(tmp_path, "b.txt", ["https://one-example.org/1", "https://one-example.org/2", "https://one-example.org/3",
                                  "https://two-example.org/1", "https://blocked-example.org/1"])
    chosen, report = community.select([a, b], quorum=2, per_site=2, blocked={"blocked-example.org"})
    assert chosen == ["https://one-example.org/1", "https://one-example.org/2", "https://two-example.org/1"]
    assert report == {"contributor_files": 2, "duplicate_files_ignored": 0, "proposed": 6, "below_quorum": 1, "blocked": 1,
                      "unranked_site": 0, "site_limit": 1, "selected": 3, "sites": 2}
    assert community.select([a], quorum=2)[0] == []  # repeating an address inside one file is one vote

    # the same file twice, or the same list saved under another name, is one contributor
    copy = write(tmp_path, "copy.txt", reversed(a.read_text().split()))
    chosen, report = community.select([a, a, copy], quorum=2)
    assert chosen == [] and report["contributor_files"] == 1 and report["duplicate_files_ignored"] == 2


def test_the_site_cap_counts_subdomains_together(tmp_path):
    urls = [f"https://s{i}.example.com/page" for i in range(4)] + ["https://alice.github.io/a", "https://bob.github.io/a"]
    chosen, report = community.select([write(tmp_path, "a.txt", urls)], quorum=1, per_site=1)
    assert chosen == ["https://alice.github.io/a", "https://bob.github.io/a", "https://s0.example.com/page"]
    assert report["site_limit"] == 3  # hosting sites such as github.io count per owner


def test_takedowns_cover_a_site_or_a_path_and_everything_under_it(tmp_path):
    urls = ["https://bad-example.org/path", "https://bad-example.org/path/deeper", "https://bad-example.org/pathology",
            "https://bad-example.org/other", "https://sub.gone-example.org/x", "https://fine-example.org/x"]
    chosen, report = community.select([write(tmp_path, "a.txt", urls)], quorum=1,
                                      blocked=["https://bad-example.org/path?utm=1", "gone-example.org", "  "])
    assert chosen == ["https://bad-example.org/other", "https://bad-example.org/pathology", "https://fine-example.org/x"]
    assert report["blocked"] == 3


def test_selection_can_require_a_well_linked_site(tmp_path):
    class Ranks:
        available = True

        def position(self, url):
            return {"one-example.org": 50, "two-example.org": 900_000}.get(url.split("/")[2])

    a = write(tmp_path, "a.txt", ["https://one-example.org/1", "https://two-example.org/1", "https://nobody-example.org/1"])
    chosen, report = community.select([a], quorum=1, authority=Ranks(), max_site_rank=100_000)
    assert chosen == ["https://one-example.org/1"] and report["unranked_site"] == 2


async def build(tmp_path, name="pilot", urls=None, **options):
    urls = urls or [f"https://site{i}-example.org/page" for i in range(5)] + ["https://site9-example.org/unreadable"]
    settings = dict(name=name, licence="CC0-1.0", attribution="Pilot builders", shard_size=3, delay=0, loader=loader)
    settings.update(options)
    return await community.build_shards(urls, tmp_path / name, **settings)


async def test_shards_manifest_and_verified_install(tmp_path):
    result = await build(tmp_path, rules={"quorum": 2})
    out = tmp_path / "pilot"
    assert [s["pages"] for s in result["shards"]] == [3, 2] and result["pages"] == 5 and result["unreadable"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["rules"] == {"quorum": 2} and [s["name"] for s in manifest["shards"]] == ["pilot-0001", "pilot-0002"]
    assert manifest["model"] == "none" and "failed" not in manifest  # addresses that failed are not published

    assert community.install_from_manifest(out / "manifest.json") == ["pilot-0001", "pilot-0002"]
    assert sorted(p.name for p in packs.installed()) == ["pilot-0001", "pilot-0002"]
    with pytest.raises(community.CommunityError, match="already installed"):
        community.install_from_manifest(out / "manifest.json")
    assert community.install_from_manifest(out / "manifest.json", replace=True) == ["pilot-0001", "pilot-0002"]
    assert sorted(p.name for p in packs.packs_dir().iterdir()) == ["pilot-0001.sqlite3", "pilot-0002.sqlite3"]


async def test_build_refuses_addresses_that_could_not_have_been_proposed(tmp_path):
    urls = ["http://localhost/private", "https://127.0.0.1.nip.io/private", "https://example.org/search?q=x",
            "https://example.org/fine"]
    seen = []

    async def recording(url):
        seen.append(url)
        return doc(url)

    result = await build(tmp_path, urls=urls, loader=recording)
    assert seen == ["https://example.org/fine"] and result["refused_addresses"] == 3
    with pytest.raises(community.CommunityError, match="No usable addresses"):
        await build(tmp_path, name="none", urls=["http://localhost/private"])
    for size in (0, -1, 10**9):
        with pytest.raises(community.CommunityError, match="shard size"):
            await build(tmp_path, name="sized", shard_size=size)
    with pytest.raises(community.CommunityError, match="No shard could be built"):
        await build(tmp_path, name="dead", urls=["https://a-example.org/unreadable"])


async def test_a_swapped_or_misnamed_shard_installs_nothing(tmp_path):
    await build(tmp_path)
    out = tmp_path / "pilot"
    original = (out / "pilot-0002.pack").read_bytes()

    # swapped after the manifest was written
    (out / "pilot-0002.pack").write_bytes((out / "pilot-0001.pack").read_bytes())
    with pytest.raises(community.CommunityError, match="checksum"):
        community.install_from_manifest(out / "manifest.json")
    assert packs.installed() == []

    # the manifest's checksum matches, but the pack inside carries another name
    await build(tmp_path, name="victim", urls=["https://other-example.org/page"])
    manifest = json.loads((out / "manifest.json").read_text())
    (out / "pilot-0002.pack").write_bytes((tmp_path / "victim" / "victim-0001.pack").read_bytes())
    manifest["shards"][1]["sha256"] = packs.file_sha256(out / "pilot-0002.pack")
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(community.CommunityError, match="names itself"):
        community.install_from_manifest(out / "manifest.json", replace=True)
    assert packs.installed() == []  # the valid first shard was not left behind

    # a file that matches its checksum but is not a pack at all
    (out / "pilot-0002.pack").write_bytes(b"not a pack")
    manifest["shards"][1]["sha256"] = packs.file_sha256(out / "pilot-0002.pack")
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(community.CommunityError, match="refused"):
        community.install_from_manifest(out / "manifest.json")
    assert packs.installed() == [] and list(packs.packs_dir().iterdir()) == []
    assert original  # (kept to show what the honest shard was)


async def test_bad_manifests_are_refused(tmp_path):
    ok = "0" * 64
    for content in ("not json", "[]", '{"manifest_version": 99, "shards": []}', '{"manifest_version": 1, "shards": []}',
                    '{"manifest_version": 1, "shards": [{"name": "../x", "file": "../x.pack", "sha256": "%s"}]}' % ok,
                    '{"manifest_version": 1, "shards": [{"name": "ok", "file": "other.pack", "sha256": "%s"}]}' % ok,
                    '{"manifest_version": 1, "shards": [{"name": "ok", "file": "ok.pack", "sha256": "short"}]}',
                    '{"manifest_version": 1, "shards": [{"name": "ok", "file": "ok.pack", "sha256": "%s"}, '
                    '{"name": "ok", "file": "ok.pack", "sha256": "%s"}]}' % (ok, ok)):
        path = tmp_path / "manifest.json"
        path.write_text(content)
        with pytest.raises(community.CommunityError):
            community.install_from_manifest(path)
