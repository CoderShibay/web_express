import gzip

import httpx
import pytest

from web_express.authority import MAX_BONUS, Authority, install, parse_line

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
