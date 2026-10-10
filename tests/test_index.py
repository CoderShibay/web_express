import time

import numpy as np
import pytest

from web_express.extract import Document
from web_express.index import LocalIndex, _match_expression


def doc(url, title, text, published=None):
    return Document(url, title, text, published, "plain", False)


def unit(*values):
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def index(tmp_path):
    index = LocalIndex(tmp_path / "index.sqlite3")
    yield index
    index.close()


def test_pages_round_trip_and_survive_reopening(tmp_path):
    path = tmp_path / "index.sqlite3"
    first = LocalIndex(path)
    first.add(doc("https://example.org/a?utm_source=x", "Kidney tumours", "Wilms tumour is a childhood kidney cancer.", "2026-01-02"))
    first.close()
    second = LocalIndex(path)
    page = second.get("http://www.example.org/a/")  # a trivially different link to the same page
    assert page and page.title == "Kidney tumours" and page.published == "2026-01-02"
    assert page.document().text == "Wilms tumour is a childhood kidney cancer."
    second.close()


def test_a_newer_copy_replaces_the_old_one(index):
    index.add(doc("https://example.org/a", "Old", "old words"))
    index.add(doc("https://example.org/a", "New", "new words"))
    assert index.stats()["pages"] == 1 and index.get("https://example.org/a").title == "New"
    assert index.search("old") == [] and [p.title for p in index.search("new")] == ["New"]


def test_keyword_search_ranks_title_matches_first(index):
    index.add(doc("https://example.org/1", "Cooking pasta", "A note that mentions asyncio once."))
    index.add(doc("https://example.org/2", "Asyncio timeouts", "How to cancel a task."))
    assert [p.url for p in index.search("asyncio")] == ["https://example.org/2", "https://example.org/1"]


def test_search_by_meaning_finds_pages_without_shared_words(index):
    index.add(doc("https://example.org/1", "Pizza", "Margherita with basil."), unit(1, 0, 0), "m")
    index.add(doc("https://example.org/2", "Tumours", "Nephroblastoma in children."), unit(0, 1, 0), "m")
    assert index.search("kidney cancer") == []
    found = index.search("kidney cancer", query_vector=unit(0.1, 1, 0), model="m")
    assert [p.url for p in found] == ["https://example.org/2"]
    assert index.search("kidney cancer", query_vector=unit(0.1, 1, 0), model="other-model") == []


def test_max_age_and_newer_than(index, monkeypatch):
    index.add(doc("https://example.org/a", "A", "words here"))
    assert index.get("https://example.org/a", max_age=60)
    real = time.time()
    monkeypatch.setattr("web_express.index.time.time", lambda: real + 3600)
    assert index.get("https://example.org/a", max_age=60) is None
    assert index.search("words", newer_than=real + 1800) == []
    assert len(index.search("words", newer_than=real - 10)) == 1


def test_oldest_pages_are_dropped_past_the_limit(index, monkeypatch):
    monkeypatch.setenv("WEB_EXPRESS_MAX_PAGES", "100")
    for i in range(105):
        index.add(doc(f"https://example.org/{i}", f"Page {i}", f"text number {i}"))
    assert index.stats()["pages"] == 100
    assert index.get("https://example.org/0") is None and index.get("https://example.org/104")


def test_remove_clear_and_empty_pages(index):
    index.add(doc("https://example.org/empty", "Empty", "   "))
    assert index.stats()["pages"] == 0
    index.add(doc("https://example.org/a", "A", "words"))
    assert index.remove("https://example.org/a") and not index.remove("https://example.org/a")
    index.add(doc("https://example.org/b", "B", "words"))
    index.clear()
    assert index.stats()["pages"] == 0 and index.search("words") == []


def test_queries_cannot_inject_search_operators(index):
    index.add(doc("https://example.org/a", "A", "plain words"))
    for query in ['"', 'words" OR "x', "NEAR(", "title:x", "a AND NOT b", "*", "-words", ""]:
        index.search(query)  # must not raise
    assert _match_expression('say "hi" NOT now') == '"say" OR "hi" OR "not" OR "now"'
