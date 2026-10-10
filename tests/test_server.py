import asyncio

import pytest

from web_express import server
from web_express.extract import Document
from web_express.federation import Federation, Outcome
from web_express.net import FetchError
from web_express.sources import Hit, Source


class Fake(Source):
    def __init__(self, name, categories, hits):
        super().__init__()
        self.name, self.categories, self.hits, self.calls = name, categories, hits, 0

    async def search(self, client, query, limit, since):
        self.calls += 1
        await asyncio.sleep(0.01)
        if isinstance(self.hits, Exception):
            raise self.hits
        return list(self.hits)


@pytest.fixture
def setup(monkeypatch):
    def install(sources, pages=None):
        monkeypatch.setattr(server, "_federation", Federation(sources))
        monkeypatch.setattr(server, "_search_cache", server.TTLCache(16, 60))
        monkeypatch.setattr(server, "_document_cache", server.TTLCache(16, 60))
        monkeypatch.setattr(server, "_partial_cache", server.TTLCache(16, 60))
        loads = []

        async def fake_download(url, **kwargs):
            loads.append(url)
            await asyncio.sleep(0.01)
            page = (pages or {}).get(url)
            if isinstance(page, Exception):
                raise page
            return page

        monkeypatch.setattr(server, "download", fake_download)
        monkeypatch.setattr(server, "extract", lambda d: d if isinstance(d, Document) else (_ for _ in ()).throw(d))
        return loads

    return install


def doc(url, text):
    return Document(url, "Page title", text, "2026-01-01", "plain", False)


async def test_search_uses_source_content_without_fetching_and_pages_otherwise(setup):
    paper = Hit("Paper", "https://p.test/1", "abs", "p1", "2026-02-02",
                "Sliding window overlap removes artifacts in tumor segmentation.", {"doi": "10.1/x", "authors": []})
    web = Hit("Blog", "https://w.test/a", "snip", "web1")
    loads = setup([Fake("web1", ("web",), [web]), Fake("p1", ("papers",), [paper])],
                  {"https://w.test/a": doc("https://w.test/a", "Intro.\n\nThe tumor segmentation study used overlap.")})
    out = await server.search("tumor segmentation study overlap")
    assert out["categories"] == ["web", "papers"] and out["sources_used"] == ["web1", "p1"]
    by_url = {r["url"]: r for r in out["results"]}
    assert by_url["https://p.test/1"]["content_from"] == "source" and by_url["https://p.test/1"]["details"] == {"doi": "10.1/x"}
    assert "start" not in by_url["https://p.test/1"]["highlights"][0]
    page = by_url["https://w.test/a"]
    assert page["content_from"] == "page" and page["published"] == "2026-01-01"
    assert loads == ["https://w.test/a"]


async def test_one_unreadable_page_does_not_sink_the_search(setup):
    hits = [Hit("Good", "https://w.test/good", source="web1"), Hit("Blocked", "https://w.test/blocked", source="web1"),
            Hit("Crash", "https://w.test/crash", source="web1")]
    setup([Fake("web1", ("web",), hits)], {
        "https://w.test/good": doc("https://w.test/good", "pizza in naples is good"),
        "https://w.test/blocked": FetchError("Page answered HTTP 403."),
        "https://w.test/crash": ValueError("boom"),
    })
    out = await server.search("pizza naples")
    errors = [r.get("content_error") for r in out["results"]]
    assert errors == [None, "Page answered HTTP 403.", "Could not read this page (ValueError)."]


async def test_domain_filters_apply_to_every_source(setup):
    hits = [Hit("A", "https://arxiv.org/abs/1", source="web1"), Hit("B", "https://sub.example.org/x", source="web1"),
            Hit("C", "https://notexample.org/x", source="web1")]
    setup([Fake("web1", ("web",), hits)])
    out = await server.search("q", contents=False, include_domains=["https://www.example.org/path", "arxiv.org"])
    assert [r["title"] for r in out["results"]] == ["A", "B"]
    out = await server.search("q", contents=False, exclude_domains=["arxiv.org"])
    assert [r["title"] for r in out["results"]] == ["B", "C"]


async def test_concurrent_identical_requests_share_one_search_and_one_download(setup):
    web = Fake("web1", ("web",), [Hit("A", "https://w.test/a", source="web1")])
    loads = setup([web], {"https://w.test/a": doc("https://w.test/a", "pizza")})
    first, second = await asyncio.gather(server.search("pizza"), server.search("pizza"))
    assert web.calls == 1 and loads == ["https://w.test/a"] and first["results"] == second["results"]
    assert (await server.search("pizza"))["cached"] is True and web.calls == 1


async def test_failures_are_reported_and_total_failure_is_an_error(setup):
    from web_express.sources import SourceError

    good = Fake("web1", ("web",), [Hit("A", "https://w.test/a", source="web1")])
    bad = Fake("web2", ("web",), SourceError("web2: rate limit reached (HTTP 429)."))
    setup([good, bad])
    out = await server.search("q", contents=False)
    assert out["source_errors"] == {"web2": "web2: rate limit reached (HTTP 429)."} and out["count"] == 1
    setup([Fake("web2", ("web",), SourceError("web2: down."))])
    with pytest.raises(RuntimeError, match="No search source answered"):
        await server.search("q")


async def test_slow_extraction_counts_against_the_page_deadline(setup, monkeypatch):
    setup([], {"https://w.test/slow": doc("https://w.test/slow", "x")})

    def slow(downloaded):
        __import__("time").sleep(0.3)
        return downloaded

    monkeypatch.setattr(server, "extract", slow)
    with pytest.raises(FetchError, match="too long"):
        await server._load("https://w.test/slow", timeout=0.05)


async def test_bad_arguments_are_rejected(setup):
    setup([])
    for kwargs in ({"query": " "}, {"query": "q", "freshness": "decade"}, {"query": "q", "category": "recipes"}):
        with pytest.raises(ValueError):
            await server.search(**kwargs)


async def test_filters_normalise_hosts_and_reject_non_domains(setup):
    hits = [Hit("Dot", "https://sub.example.org./x", source="web1"), Hit("Port", "https://example.org:8443/x", source="web1"),
            Hit("Other", "https://other.test/x", source="web1")]
    setup([Fake("web1", ("web",), hits)])
    out = await server.search("q", contents=False, exclude_domains=["example.org:443"])
    assert [r["title"] for r in out["results"]] == ["Other"]
    out = await server.search("q", contents=False, include_domains=["EXAMPLE.org."])
    assert [r["title"] for r in out["results"]] == ["Dot", "Port"]
    with pytest.raises(ValueError, match="not a domain"):
        await server.search("q", exclude_domains=["   "])


async def test_a_redirect_outside_the_filters_returns_no_content(setup):
    hit = Hit("Go", "https://allowed.test/go", source="web1")
    setup([Fake("web1", ("web",), [hit])], {"https://allowed.test/go": doc("https://blocked.test/secret", "secret pizza text")})
    out = await server.search("pizza", exclude_domains=["blocked.test"])
    result = out["results"][0]
    assert result["final_url"] == "https://blocked.test/secret" and "highlights" not in result
    assert "outside the domain filters" in result["content_error"]


async def test_partial_answers_are_retried_sooner_than_complete_ones(setup, monkeypatch):
    from web_express.sources import SourceError

    good = Fake("web1", ("web",), [Hit("A", "https://w.test/a", source="web1")])
    flaky = Fake("web2", ("web",), SourceError("web2: down."))
    setup([good, flaky])
    await server.search("q", contents=False)
    assert len(server._partial_cache) == 1 and len(server._search_cache) == 0
    flaky.hits = [Hit("B", "https://w.test/b", source="web2")]
    server._federation._paused.clear()
    monkeypatch.setattr(server, "_partial_cache", server.TTLCache(16, 60))  # as if the short time limit passed
    out = await server.search("q", contents=False)
    assert out["sources_used"] == ["web1", "web2"] and len(server._search_cache) == 1


async def test_results_are_reranked_by_meaning_when_the_model_is_ready(setup, monkeypatch):
    from tests.test_ranking import FakeEmbedder

    hits = [Hit("Pizza recipes from Naples", "https://w.test/1", source="web1"),
            Hit("Something unrelated", "https://w.test/2", source="web1"),
            Hit("How to abort a slow job", "https://w.test/3", source="web1")]
    setup([Fake("web1", ("web",), hits)])
    out = await server.search("stop it", contents=False)
    assert [r["url"][-1] for r in out["results"]] == ["1", "2", "3"] and out["ranking"].startswith("keyword")

    monkeypatch.setattr(server.embed, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(server.embed, "status", lambda: "ready")
    out = await server.search("stop it", contents=False, max_results=1)
    assert out["results"][0]["url"] == "https://w.test/3" and out["ranking"] == "semantic"
