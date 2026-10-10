import time

import numpy as np
import pytest

from web_express.extract import Document
from web_express.index import LocalIndex, _match_expression, storable, worth_keeping


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
    first.add(doc("https://example.org/a?page=2", "Kidney tumours", "Wilms tumour is a childhood kidney cancer.", "2026-01-02"))
    first.close()
    second = LocalIndex(path)
    page = second.get("HTTPS://Example.org:443/a?page=2#section")  # the same address, written differently
    assert page and page.title == "Kidney tumours" and page.published == "2026-01-02"
    assert page.document().text == "Wilms tumour is a childhood kidney cancer." and page.document().fetched_at
    second.close()


def test_different_pages_never_share_a_slot(index):
    index.add(doc("https://example.org/account?utm_source=mail", "A", "private-A text"))
    index.add(doc("http://www.example.org/account/", "B", "public-B text"))
    index.add(doc("https://example.org/account?id=2", "C", "third page"))
    assert index.stats()["pages"] == 3
    assert index.get("https://example.org/account?utm_source=mail").text == "private-A text"
    assert index.get("http://www.example.org/account/").text == "public-B text"
    assert index.get("https://example.org/account") is None


def test_addresses_carrying_credentials_are_never_stored(index):
    secrets = ["https://example.org/download?access_token=TOPSECRET&X-Amz-Signature=SIGNED",
               "https://user:pw@example.org/x", "https://example.org/cb?code=abc", "https://example.org/x?API_KEY=1",
               "https://example.org/x?sessionid=9", "https://example.org/x?sig=1"]
    for url in secrets:
        assert index.add(doc(url, "T", "some body text")) is False and not storable(url)
    assert index.stats()["pages"] == 0
    raw = index.path.read_bytes() + index.path.with_name(index.path.name + "-wal").read_bytes()
    assert b"TOPSECRET" not in raw and b"SIGNED" not in raw
    for url in ("https://example.org/x?page=2&q=keyboard", "https://example.org/monkey?design=1"):
        assert storable(url)


def test_gate_pages_are_not_worth_keeping():
    filler = "Real article text about tumours. " * 20
    assert worth_keeping(doc("https://e.org/a", "Tumours", filler))
    assert not worth_keeping(doc("https://e.org/a", "Short", "tiny"))
    assert not worth_keeping(doc("https://e.org/a", "Just a moment...", "Checking your browser before accessing. " * 8))
    assert not worth_keeping(doc("https://e.org/a", "Welcome", "Please verify you are human to continue. " * 8))
    assert worth_keeping(doc("https://e.org/a", "CAPTCHA history", "An article that discusses captcha systems. " * 80))


def test_vectors_of_the_wrong_shape_are_ignored_not_spliced(index):
    index.add(doc("https://example.org/2d", "Two", "two dims page"), unit(1, 0), "same")
    index.add(doc("https://example.org/4d", "Four", "four dims page"), unit(1, 0, 0, 0), "same")
    index.add(doc("https://example.org/nan", "NaN", "nan page"), np.array([np.nan, 1.0], dtype=np.float32), "same")
    index.add(doc("https://example.org/matrix", "Matrix", "matrix page"), np.ones((2, 2), dtype=np.float32), "same")
    assert index.stats()["pages"] == 4 and index.stats()["pages_with_vectors"] == 2
    assert index.search("zzz", query_vector=unit(1, 0, 0), model="same") == []  # no stored vector has 3 dims
    assert [p.url for p in index.search("zzz", query_vector=unit(1, 0), model="same")] == ["https://example.org/2d"]
    index._db.execute("UPDATE pages SET vector = x'00112233' WHERE url = 'https://example.org/2d'")
    index._db.commit()
    index._vectors = None
    assert index.search("zzz", query_vector=unit(1, 0), model="same") == []


def test_writes_from_another_connection_are_seen(tmp_path):
    path = tmp_path / "index.sqlite3"
    reader, writer = LocalIndex(path), LocalIndex(path)
    reader.add(doc("https://example.org/1", "One", "first page"), unit(1, 0), "m")
    assert len(reader.search("zzz", query_vector=unit(0, 1), model="m")) == 0
    writer.add(doc("https://example.org/2", "Two", "second page"), unit(0, 1), "m")
    assert [p.url for p in reader.search("zzz", query_vector=unit(0, 1), model="m")] == ["https://example.org/2"]
    reader.close(), writer.close()


def test_a_damaged_file_is_set_aside_and_a_fresh_index_started(tmp_path):
    path = tmp_path / "index.sqlite3"
    path.write_bytes(b"this is not a database, " * 100)
    index = LocalIndex(path)
    assert index.add(doc("https://example.org/a", "A", "works again"))
    assert len(list(tmp_path.glob("index.sqlite3.corrupt-*"))) == 1
    index.close()


def test_replacing_pages_does_not_grow_the_file_without_bound(tmp_path, monkeypatch):
    monkeypatch.setattr("web_express.index.TIDY_EVERY", 50)
    index = LocalIndex(tmp_path / "index.sqlite3")
    body = "lorem ipsum dolor sit amet " * 400
    for i in range(40):
        index.add(doc(f"https://example.org/{i}", f"Page {i}", body + str(i)))
    index.tidy()
    baseline = index.stats()["file_bytes"]
    for round_number in range(10):
        for i in range(40):
            index.add(doc(f"https://example.org/{i}", f"Page {i}", body + f"{round_number}-{i}"))
    index.tidy()
    assert index.stats()["file_bytes"] < baseline * 1.5
    index.close()


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


def test_common_words_alone_do_not_match(index):
    index.add(doc("https://example.org/a", "Asyncio", "What a timeout is and how it works."))
    assert index.search("what is the best way") == [] and _match_expression("what is it") == ""


def test_a_shared_word_is_not_enough_when_the_meaning_is_far(index):
    index.add(doc("https://example.org/1", "Python the snake", "The python is a large snake."), unit(1, 0, 0), "m")
    index.add(doc("https://example.org/2", "Python the language", "Python is a programming language."), unit(0, 1, 0), "m")
    found = index.search("python", query_vector=unit(0, 1, 0), model="m")
    assert [p.url for p in found] == ["https://example.org/2"]
