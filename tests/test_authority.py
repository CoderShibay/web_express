import gzip

import httpx
import pytest

from web_express.authority import MAX_BONUS, Authority, _lines, install, parse_line

RANKS = b"""#harmonicc_pos\t#harmonicc_val\t#pr_pos\t#pr_val\t#host_rev\t#n_hosts
1\t2.9E7\t3\t0.009\tcom.facebook\t3054
2\t2.8E7\t1\t0.014\torg.wikipedia\t700
3\t2.7E7\t2\t0.012\tuk.co.bbc\t18924
garbage line
4\t2.6E7\t5\t0.006\tcom.example\t5
5\t2.5E7\t4\t0.007\tcom.never-reached\t1
"""


def transport(ranks=RANKS, releases=None):
    def handler(request):
        if request.url.path.endswith("graphinfo.json"):
            return httpx.Response(200, json=[{"id": "cc-main-2026-test"}] if releases is None else releases)
        assert "cc-main-2026-test/domain/cc-main-2026-test-domain-ranks.txt.gz" in str(request.url)
        return httpx.Response(200, stream=httpx.ByteStream(gzip.compress(ranks)))

    return httpx.MockTransport(handler)


def test_parse_line():
    assert parse_line(b"3\t2.7E7\t2\t0.012\tuk.co.bbc\t18924") == ("bbc.co.uk", 3)
    for bad in (b"#header", b"", b"x\ty", b"0\t1\t1\t1\tcom.zero\t1", b"1\t1\t1\t1\t\xff\xfe\t1"):
        assert parse_line(bad) is None


def test_install_keeps_only_the_top_domains_and_lookups_use_the_site_not_the_page(tmp_path):
    path = tmp_path / "authority.sqlite3"
    assert install(top=4, path=path, transport=transport()) == {"release": "cc-main-2026-test", "domains": 4, "path": str(path)}
    authority = Authority(path)
    assert authority.available and authority.info() == {"release": "cc-main-2026-test", "domains": "4"}
    assert authority.position("https://en.wikipedia.org/wiki/Wilms%27_tumor") == 2
    assert authority.position("https://www.bbc.co.uk/news") == 3
    assert authority.position("https://never-reached.com/") is None  # beyond the top 4
    assert authority.position("https://co.uk/") is None and authority.position("not a url") is None
    assert authority.position("https://1.2.3.4/") is None and authority.position("https://[::1]/") is None
    assert authority.bonus("https://facebook.com/") == MAX_BONUS
    assert 0 < authority.bonus("https://example.com/x") < MAX_BONUS and authority.bonus("https://unknown.test/") == 0


def test_without_data_every_lookup_is_neutral(tmp_path):
    authority = Authority(tmp_path / "missing.sqlite3")
    assert not authority.available and authority.position("https://facebook.com/") is None
    assert authority.bonus("https://facebook.com/") == 0.0 and authority.info() == {}
    (tmp_path / "broken.sqlite3").write_bytes(b"not a database")
    assert not Authority(tmp_path / "broken.sqlite3").available


def test_a_failed_install_leaves_the_previous_data_in_place(tmp_path):
    path = tmp_path / "authority.sqlite3"
    install(top=2, path=path, transport=transport())
    with pytest.raises(RuntimeError):
        install(path=path, transport=transport(ranks=b"#only a header\n"))
    with pytest.raises(RuntimeError):
        install(path=path, transport=transport(releases=[{"id": "../../etc"}]))
    assert Authority(path).position("https://facebook.com/") == 1
    assert [p.name for p in tmp_path.iterdir()] == ["authority.sqlite3"]


def test_hosting_suffixes_and_international_names(tmp_path):
    ranks = (b"1\t1\t1\t1\tio.github\t9\n2\t1\t1\t1\tuk.co\t9\n3\t1\t1\t1\tcom.xn--xample-9ua\t1\n"
             b"4\t1\t1\t1\torg.python\t5\n5\t1\t1\t1\t4.3\t1\n")
    path = tmp_path / "a.sqlite3"
    install(path=path, transport=transport(ranks=ranks))
    authority = Authority(path)
    assert authority.position("https://github.io/") == 1
    assert authority.position("https://attacker.github.io/") is None  # its own site, not github.io
    assert authority.position("https://attacker.co.uk/") is None
    assert authority.position("https://éxample.com/") == 3
    assert authority.position("https://docs.python.org/3/") == 4
    assert authority.position("https://1.2.3.4/") is None


def test_relative_paths_work(tmp_path, monkeypatch):
    install(top=2, path=tmp_path / "authority.sqlite3", transport=transport())
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    assert Authority(Path("authority.sqlite3")).position("https://facebook.com/") == 1


def test_a_truncated_download_is_an_error_and_keeps_the_old_data(tmp_path):
    path = tmp_path / "authority.sqlite3"
    install(top=2, path=path, transport=transport())
    many = b"".join(b"%d\t1\t1\t1\tcom.site%d\t1\n" % (i, i) for i in range(1, 400))
    cut = gzip.compress(many)[:-20]

    def handler(request):
        if request.url.path.endswith("graphinfo.json"):
            return httpx.Response(200, json=[{"id": "cc-main-2026-test"}])
        return httpx.Response(200, stream=httpx.ByteStream(cut))

    with pytest.raises(RuntimeError, match="ended early|damaged"):
        install(top=100_000, path=path, transport=httpx.MockTransport(handler))
    assert Authority(path).info()["domains"] == "2"


def test_a_highly_compressed_stream_is_read_in_small_pieces():
    import tracemalloc

    bomb = gzip.compress(b"x\n" * 5_000_000)
    assert len(bomb) < 20_000

    class Response:
        def iter_raw(self):
            yield bomb

    tracemalloc.start()
    with pytest.raises(RuntimeError, match="far larger"):
        for _ in _lines(Response(), max_bytes=1_000_000):
            pass
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 2_000_000
