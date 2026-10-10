from datetime import date

import httpx
import pytest

from web_express.federation import Federation, canonical_url, fuse, guess_categories
from web_express.sources import (
    Arxiv, EuropePMC, GitHub, HackerNews, Hit, OpenAlex, Source, SourceError, StackOverflow, Wikipedia,
)

ARXIV = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/1909.06684v2</id><title>Kidney  Tumor
 Segmentation</title><summary> A boundary-aware
 network. </summary><published>2019-09-14T21:49:09Z</published>
<author><name>A. One</name></author><author><name>B. Two</name></author></entry></feed>"""


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def ask(source, handler, query="q", limit=5, since=None):
    source.min_interval = 0
    async with client_for(handler) as client:
        return await source.search(client, query, limit, since)


# --- sources -----------------------------------------------------------------

async def test_arxiv_parses_entries_and_drops_the_version():
    hits = await ask(Arxiv(), lambda r: httpx.Response(200, text=ARXIV), "kidney tumor")
    assert len(hits) == 1
    hit = hits[0]
    assert hit.url == "https://arxiv.org/abs/1909.06684" and hit.title == "Kidney Tumor Segmentation"
    assert hit.published == "2019-09-14" and hit.content == "A boundary-aware network."
    assert hit.extra["pdf_url"] == "https://arxiv.org/pdf/1909.06684" and hit.extra["authors"] == ["A. One", "B. Two"]


async def test_arxiv_query_is_built_from_plain_terms():
    seen = {}

    def handler(request):
        seen["q"] = request.url.params["search_query"]
        return httpx.Response(200, text=ARXIV)

    await ask(Arxiv(), handler, 'wilms "tumor" (ct)')
    assert seen["q"] == "all:wilms tumor ct"


async def test_europepmc_strips_html_from_abstracts_and_filters_by_date():
    seen = {}

    def handler(request):
        seen["query"] = request.url.params["query"]
        return httpx.Response(200, json={"resultList": {"result": [{
            "id": "42", "source": "MED", "title": "Wilms tumor models.", "doi": "10.1/x",
            "abstractText": "<h4>Background</h4>Creating 3D models.", "firstPublicationDate": "2026-04-25", "isOpenAccess": "Y",
        }, {"title": "no id"}]}})

    hits = await ask(EuropePMC(), handler, "wilms", since=date(2026, 1, 1))
    assert "FIRST_PDATE:[2026-01-01 TO" in seen["query"]
    assert len(hits) == 1 and hits[0].url == "https://europepmc.org/article/MED/42"
    assert "<h4>" not in hits[0].content and "Creating 3D models." in hits[0].content
    assert hits[0].extra["doi"] == "10.1/x" and hits[0].extra["open_access"] is True


async def test_openalex_rebuilds_the_abstract():
    payload = {"results": [{
        "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/Y", "title": "T", "publication_date": "2023-04-03",
        "abstract_inverted_index": {"learning": [1], "Deep": [0], "works": [2]}, "cited_by_count": 3,
        "primary_location": {"landing_page_url": "https://pub.test/t"}, "open_access": {"oa_url": None},
    }]}
    hits = await ask(OpenAlex(), lambda r: httpx.Response(200, json=payload))
    assert hits[0].content == "Deep learning works" and hits[0].url == "https://pub.test/t" and hits[0].extra["doi"] == "10.1/Y"


async def test_stackoverflow_attaches_the_top_answer():
    def handler(request):
        if request.url.path.endswith("/search/advanced"):
            return httpx.Response(200, json={"items": [{
                "question_id": 7, "title": "How to &quot;cancel&quot;?", "link": "https://stackoverflow.com/q/7",
                "body": "<p>I need to cancel.</p>", "creation_date": 1476349016, "score": 16, "is_answered": True,
            }]})
        assert request.url.path.endswith("/questions/7/answers")
        return httpx.Response(200, json={"items": [
            {"question_id": 7, "score": 26, "is_accepted": True, "body": "<p>Use <code>task.cancel()</code>.</p>"},
            {"question_id": 7, "score": 2, "is_accepted": False, "body": "<p>Worse idea.</p>"},
        ]})

    hits = await ask(StackOverflow(), handler)
    assert hits[0].title == 'How to "cancel"?' and hits[0].published == "2016-10-13"
    assert "Accepted answer (score 26)" in hits[0].content and "task.cancel()" in hits[0].content
    assert "Worse idea" not in hits[0].content


async def test_github_and_hackernews_shapes():
    repo = {"items": [{"full_name": "a/b", "html_url": "https://github.com/a/b", "description": None,
                       "stargazers_count": 9, "pushed_at": "2023-12-12T02:58:36Z", "license": None}]}
    hits = await ask(GitHub(), lambda r: httpx.Response(200, json=repo))
    assert (hits[0].title, hits[0].published, hits[0].extra["stars"]) == ("a/b", "2023-12-12", 9)
    story = {"hits": [{"title": "Show HN", "url": None, "objectID": "5", "points": 1, "created_at": "2025-04-16T08:06:44Z"}]}
    hits = await ask(HackerNews(), lambda r: httpx.Response(200, json=story))
    assert hits[0].url == "https://news.ycombinator.com/item?id=5"


async def test_rate_limits_and_bad_answers_become_source_errors():
    with pytest.raises(SourceError, match="rate limit"):
        await ask(GitHub(), lambda r: httpx.Response(429))
    with pytest.raises(SourceError, match="unreadable"):
        await ask(GitHub(), lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(SourceError, match="unreadable"):
        await ask(Arxiv(), lambda r: httpx.Response(200, text="not xml"), "x")


def test_wikipedia_is_off_without_a_contact(monkeypatch):
    monkeypatch.delenv("WEB_EXPRESS_CONTACT", raising=False)
    assert not Wikipedia().available()
    monkeypatch.setenv("WEB_EXPRESS_CONTACT", "https://example.org/me")
    assert Wikipedia().available()


# --- routing and merging -----------------------------------------------------

def test_guess_categories():
    assert guess_categories("best pizza in naples") == ["web"]
    assert "papers" in guess_categories("wilms tumor segmentation study")
    assert "code" in guess_categories("TypeError in asyncio.wait_for()")
    assert "news" in guess_categories("bitcoin etf", freshness="day")


def test_canonical_url_merges_trivial_variants():
    assert canonical_url("http://www.Example.org/a/?utm_source=x&b=1#top") == "https://example.org/a?b=1"
    assert canonical_url("https://arxiv.org/pdf/1909.06684v2") == canonical_url("http://arxiv.org/abs/1909.06684")
    assert canonical_url("https://example.org/a") != canonical_url("https://example.org/b")


def test_fuse_rewards_agreement_and_merges_details():
    first = [Hit("A", "https://x.test/a", source="one"), Hit("B", "https://x.test/b", source="one")]
    second = [Hit("B again", "https://www.x.test/b/", source="two", content="full text", published="2026-01-02"),
              Hit("C", "https://x.test/c", source="two")]
    hits = fuse([first, second])
    assert [h.title for h in hits] == ["B", "A", "C"]
    assert hits[0].content == "full text" and hits[0].extra["sources"] == ["one", "two"]


def test_fuse_merges_by_doi_and_drops_old_results():
    a = Hit("P", "https://europepmc.org/article/MED/1", source="europepmc", extra={"doi": "10.1/X"}, published="2020-01-01")
    b = Hit("P", "https://pub.test/p", source="openalex", extra={"doi": "10.1/x"}, published="2020-01-01")
    assert len(fuse([[a], [b]])) == 1
    assert fuse([[a]], since=date(2024, 1, 1)) == []


class Fake(Source):
    def __init__(self, name, categories, result):
        super().__init__()
        self.name, self.categories, self.result, self.calls = name, categories, result, 0

    async def search(self, client, query, limit, since):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def test_federation_routes_by_category_and_reports_failures():
    web = Fake("web1", ("web",), [Hit("W", "https://w.test/", source="web1")])
    papers = Fake("p1", ("papers",), [Hit("P", "https://p.test/", source="p1")])
    broken = Fake("p2", ("papers",), SourceError("p2: rate limit reached (HTTP 429)."))
    federation = Federation([web, papers, broken])

    outcome = await federation.search("anything", category="papers")
    assert [h.title for h in outcome.hits] == ["P"] and outcome.used == ["p1"]
    assert outcome.errors == {"p2": "p2: rate limit reached (HTTP 429)."} and web.calls == 0

    outcome = await federation.search("anything", category="papers")  # the broken source is now paused
    assert broken.calls == 1 and "paused" in outcome.errors["p2"]

    with pytest.raises(ValueError):
        await federation.search("x", category="recipes")


async def test_federation_auto_adds_specialists_only_when_hinted():
    web = Fake("web1", ("web",), [Hit("W", "https://w.test/", source="web1")])
    papers = Fake("p1", ("papers",), [Hit("P", "https://p.test/", source="p1")])
    federation = Federation([web, papers])
    assert (await federation.search("best pizza in naples")).used == ["web1"]
    assert (await federation.search("tumor segmentation study")).used == ["web1", "p1"]
