import httpx
import pytest

from web_express.cache import TTLCache
from web_express.engines import EngineError, clean_result_url, parse_duckduckgo
from web_express.extract import extract
from web_express.net import BlockedAddress, Download, FetchError, download, resolve_public
from web_express.passages import split_passages, top_passages

PUBLIC = "93.184.216.34"


def reply(body=b"", content_type="text/html", status=200, **headers):
    """A response whose body is streamed, as real network responses are."""
    return httpx.Response(status, headers={"content-type": content_type, **headers}, stream=httpx.ByteStream(body))


def resolver_for(mapping):
    async def resolve(host, port):
        return mapping[host]

    return resolve


# --- addresses ---------------------------------------------------------------

@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fe80::1%en0"])
async def test_private_addresses_are_refused(address):
    with pytest.raises(BlockedAddress):
        await resolve_public("evil.test", 80, resolver_for({"evil.test": [address]}))


async def test_mixed_public_and_private_answers_are_refused():
    with pytest.raises(BlockedAddress):
        await resolve_public("evil.test", 80, resolver_for({"evil.test": [PUBLIC, "127.0.0.1"]}))


async def test_public_address_prefers_ipv4():
    resolve = resolver_for({"ok.test": ["2606:4700:4700::1111", PUBLIC]})
    assert await resolve_public("ok.test", 443, resolve) == PUBLIC


# --- downloads ---------------------------------------------------------------

async def test_download_connects_to_the_checked_address_with_the_real_host():
    seen = {}

    def handler(request):
        seen["host"], seen["header"] = request.url.host, request.headers["host"]
        return reply(b"<p>hi</p>", "text/html; charset=utf-8")

    result = await download(
        "http://ok.test/page", resolver=resolver_for({"ok.test": [PUBLIC]}), transport=httpx.MockTransport(handler)
    )
    assert seen == {"host": PUBLIC, "header": "ok.test"}
    assert result.url == "http://ok.test/page" and result.body == b"<p>hi</p>" and not result.truncated


async def test_redirect_to_a_private_address_is_refused():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://inside.test/admin"})

    resolve = resolver_for({"ok.test": [PUBLIC], "inside.test": ["127.0.0.1"]})
    with pytest.raises(BlockedAddress):
        await download("http://ok.test/", resolver=resolve, transport=httpx.MockTransport(handler))


async def test_download_stops_at_the_size_limit(monkeypatch):
    monkeypatch.setattr("web_express.net.MAX_TEXT_BYTES", 1000)

    async def body():
        for _ in range(100):
            yield b"x" * 500

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=body())

    result = await download(
        "http://ok.test/big", resolver=resolver_for({"ok.test": [PUBLIC]}), transport=httpx.MockTransport(handler)
    )
    assert len(result.body) == 1000 and result.truncated


async def test_unsupported_content_type_is_refused_without_reading():
    def handler(request):
        return reply(b"\x00" * 10, "video/mp4")

    with pytest.raises(FetchError, match="Unsupported content type"):
        await download(
            "http://ok.test/v", resolver=resolver_for({"ok.test": [PUBLIC]}), transport=httpx.MockTransport(handler)
        )


async def fetch_ok(url="http://ok.test/", handler=None, **mapping):
    resolve = resolver_for({"ok.test": [PUBLIC], **mapping})
    return await download(url, resolver=resolve, transport=httpx.MockTransport(handler))


async def test_a_response_exactly_at_the_limit_is_not_truncated(monkeypatch):
    monkeypatch.setattr("web_express.net.MAX_TEXT_BYTES", 1000)
    result = await fetch_ok(handler=lambda r: reply(b"x" * 1000, "text/plain"))
    assert len(result.body) == 1000 and not result.truncated
    result = await fetch_ok(handler=lambda r: reply(b"x" * 1001, "text/plain"))
    assert len(result.body) == 1000 and result.truncated


async def test_a_compressed_bomb_is_cut_off_without_expanding_in_memory(monkeypatch):
    import gzip
    import tracemalloc

    monkeypatch.setattr("web_express.net.MAX_TEXT_BYTES", 1000)
    bomb = gzip.compress(b"0" * 30_000_000)
    assert len(bomb) < 40_000
    tracemalloc.start()
    result = await fetch_ok(handler=lambda r: reply(bomb, "text/plain", **{"content-encoding": "gzip"}))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(result.body) == 1000 and result.truncated and peak < 2_000_000


async def test_gzip_and_deflate_bodies_are_decoded():
    import gzip
    import zlib

    for encoding, data in (("gzip", gzip.compress(b"hello")), ("deflate", zlib.compress(b"hello"))):
        result = await fetch_ok(handler=lambda r: reply(data, "text/plain", **{"content-encoding": encoding}))
        assert result.body == b"hello"
    with pytest.raises(FetchError, match="content encoding"):
        await fetch_ok(handler=lambda r: reply(b"x", "text/plain", **{"content-encoding": "br"}))
    with pytest.raises(FetchError, match="decompressed"):
        await fetch_ok(handler=lambda r: reply(b"not gzip", "text/plain", **{"content-encoding": "gzip"}))


async def test_proxy_settings_from_the_environment_are_ignored(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    captured = {}
    real = httpx.AsyncClient

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("web_express.net.httpx.AsyncClient", spy)
    await fetch_ok(handler=lambda r: reply(b"ok", "text/plain"))
    assert captured["trust_env"] is False


async def test_international_and_ipv6_hosts_get_a_valid_host_header():
    seen = []

    def handler(request):
        seen.append(request.headers["host"])
        return reply(b"ok", "text/plain")

    await fetch_ok("http://éxample.test/", handler, **{"xn--xample-9ua.test": [PUBLIC]})
    await fetch_ok("http://[2606:4700:4700::1111]:8080/a", handler, **{"2606:4700:4700::1111": ["2606:4700:4700::1111"]})
    assert seen == ["xn--xample-9ua.test", "[2606:4700:4700::1111]:8080"]


@pytest.mark.parametrize("url", ["http://[", "http://", "not a url", "http://ok.test:99999/"])
async def test_malformed_urls_are_a_clean_error(url):
    with pytest.raises(FetchError):
        await fetch_ok(url, lambda r: reply(b"ok", "text/plain"))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://ok.test/x", "http://user:pw@ok.test/"])
async def test_bad_urls_are_refused(url):
    with pytest.raises(FetchError):
        await download(url, resolver=resolver_for({"ok.test": [PUBLIC]}))


# --- extraction --------------------------------------------------------------

def page(html):
    return Download("https://ok.test/a", 200, "text/html", html.encode(), "utf-8", False)


def test_page_wrapped_in_a_form_is_not_lost():
    html = "<html><head><title>Shop</title></head><body><form><div><p>The warranty lasts two years.</p></div></form></body></html>"
    document = extract(page(html))
    assert "warranty lasts two years" in document.text and document.title == "Shop"


def test_plain_fallback_keeps_links_as_absolute_urls():
    document = extract(page('<body><p>See <a href="/docs">the docs</a> now.</p></body>'))
    assert "[the docs](https://ok.test/docs)" in document.text


def test_a_malformed_link_does_not_break_the_page():
    document = extract(page('<body><p>Before <a href="http://[">bad link</a> after.</p></body>'))
    assert "Before bad link after." in document.text


def test_scripts_and_styles_are_dropped():
    document = extract(page("<body><script>var secret=1</script><style>p{}</style><p>Visible.</p></body>"))
    assert "Visible." in document.text and "secret" not in document.text


def test_meta_charset_is_used_when_the_header_has_none():
    body = '<html><head><meta charset="windows-1252"></head><body><p>café</p></body></html>'.encode("windows-1252")
    assert "café" in Download("https://ok.test/", 200, "text/html", body, None, False).text()


# --- passages ----------------------------------------------------------------

def test_passages_are_exact_slices_of_the_text():
    text = "# Title\n\nFirst paragraph here.\n\n" + ("A long sentence about nothing. " * 80) + "\n\nLast one."
    passages = split_passages(text)
    assert passages and all(text[p.start : p.end] == p.text for p in passages)
    assert all(len(p.text) <= 900 for p in passages)


def test_the_matching_passage_ranks_first():
    filler = "\n\n".join(f"Paragraph {i} talks about gardening, soil and watering schedules in general terms. " * 6 for i in range(6))
    target = "The sliding window overlap should be set to 0.5 to remove boundary artifacts. " * 5
    text = filler + "\n\n" + target + "\n\n" + filler
    best = top_passages("sliding window overlap artifacts", text, 2)
    assert best and "sliding window overlap" in best[0].text


def test_highlights_work_for_non_latin_scripts():
    assert top_passages("東京", "前文。東京についての重要な説明です。", 3)
    assert top_passages("привет", "Это привет миру.", 3)
    assert top_passages("CAFÉ", "Un café noir.", 3)


def test_no_matching_words_gives_no_highlights():
    assert top_passages("zzzqqq", "Nothing relevant here at all.", 3) == []


# --- search engine parsing ---------------------------------------------------

RESULTS = """
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=x">First <b>hit</b></a>
<a class="result__snippet" href="x">A snippet   here.</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>
"""


def test_results_are_parsed_and_ads_dropped():
    hits = parse_duckduckgo(RESULTS)
    assert [(h.title, h.url, h.snippet) for h in hits] == [("First hit", "https://example.org/a", "A snippet here.")]


def test_a_real_no_results_page_is_an_empty_list():
    assert parse_duckduckgo('<div class="no-results">No results.</div>') == []


def test_a_block_page_is_an_error_not_an_empty_list():
    with pytest.raises(EngineError, match="human check"):
        parse_duckduckgo("<html>anomaly detected</html>")
    with pytest.raises(EngineError, match="layout may have changed"):
        parse_duckduckgo("<html><body>something else</body></html>")


def test_clean_result_url_leaves_plain_links_alone():
    assert clean_result_url("https://example.org/x?uddg=1") == "https://example.org/x?uddg=1"


# --- cache -------------------------------------------------------------------

def test_cache_evicts_oldest_and_expires(monkeypatch):
    cache = TTLCache(max_entries=2, ttl=10)
    cache.set("a", 1), cache.set("b", 2), cache.get("a"), cache.set("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and len(cache) == 2
    now = __import__("time").monotonic()
    monkeypatch.setattr("web_express.cache.time.monotonic", lambda: now + 11)
    assert cache.get("a") is None
