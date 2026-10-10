import hashlib

import httpx
import numpy as np
import pytest

from web_express import embed
from web_express.ranking import EMBED_PER_PAGE, _candidates, best_passages, rerank


class FakeEmbedder:
    """Vectors by topic word, so tests control what is 'similar' without the real model."""

    topics = ["cancel", "pizza", "tumor"]
    synonyms = {"abort": "cancel", "stop": "cancel", "margherita": "pizza", "nephroblastoma": "tumor"}

    def embed(self, texts):
        out = np.zeros((len(texts), 3 + 64), dtype=np.float32)
        for row, text in enumerate(texts):
            words = [self.synonyms.get(w, w) for w in text.lower().replace(".", " ").split()]
            for column, topic in enumerate(self.topics):
                out[row, column] = words.count(topic)
            if not out[row].any():  # no known topic: point somewhere of its own
                out[row, 3 + sum(map(ord, text)) % 64] = 1.0
            out[row] /= np.linalg.norm(out[row])
        return out


TEXT = "\n\n".join([
    "Introduction to the event loop and how it schedules work. " * 9,
    "You can abort a running job by calling its abort method, which will abort it at the next await. " * 5,
    "A margherita needs tomato, mozzarella and basil, baked very hot. " * 8,
])


def test_without_a_model_ranking_is_keyword_only():
    assert best_passages("stop a task", TEXT, 2) == []  # no shared words
    assert "event loop" in best_passages("event loop", TEXT, 1)[0].text


def test_with_a_model_meaning_finds_what_keywords_miss():
    best = best_passages("stop a task", TEXT, 1, FakeEmbedder())
    assert best and "abort" in best[0].text
    assert best_passages("quantum chromodynamics", TEXT, 3, FakeEmbedder()) == []


def test_passages_stay_exact_slices_with_a_model():
    for passage in best_passages("stop margherita", TEXT, 3, FakeEmbedder()):
        assert TEXT[passage.start : passage.end] == passage.text


def test_long_pages_are_sampled_not_embedded_whole():
    calls = []

    class Counting(FakeEmbedder):
        def embed(self, texts):
            calls.append(len(texts))
            return super().embed(texts)

    long_text = "\n\n".join(f"Paragraph {i} about nothing in particular, padded out to a fair length. " * 8 for i in range(300))
    best_passages("stop", long_text, 3, Counting())
    assert calls and calls[0] <= 65


def test_rerank_blends_meaning_with_the_incoming_order():
    titles = ["Pizza recipes from Naples", "Something unrelated", "How to abort a slow job", "Cancel a task"]
    assert rerank("stop it", titles) == [0, 1, 2, 3]
    order = rerank("stop it", titles, FakeEmbedder())
    assert order[:2] == [2, 3] and set(order) == {0, 1, 2, 3}


# --- model installation ------------------------------------------------------

def spec_for(files):
    return embed.ModelSpec("tiny", "org/tiny", "abcdef1234567890", files, 4, "sentence_embedding")


def test_install_verifies_checksums_and_leaves_nothing_behind_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_EXPRESS_HOME", str(tmp_path))
    good = spec_for({"model.onnx": ("onnx/m.onnx", hashlib.sha256(b"weights").hexdigest())})
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=b"weights")

    folder = embed.install(good, transport=httpx.MockTransport(handler))
    assert (folder / "model.onnx").read_bytes() == b"weights" and embed.is_installed(good)
    assert seen == ["https://huggingface.co/org/tiny/resolve/abcdef1234567890/onnx/m.onnx"]
    embed.install(good, transport=httpx.MockTransport(handler))
    assert len(seen) == 1  # already present and correct: not downloaded again

    bad = spec_for({"other.onnx": ("onnx/o.onnx", "0" * 64)})
    with pytest.raises(RuntimeError, match="checksum"):
        embed.install(bad, transport=httpx.MockTransport(handler))
    assert not embed.is_installed(bad) and [p.name for p in folder.iterdir()] == ["model.onnx"]


def test_get_embedder_never_blocks_and_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_EXPRESS_HOME", str(tmp_path))
    monkeypatch.setenv("WEB_EXPRESS_MODEL", "off")
    embed.reset()
    assert embed.get_embedder() is None and embed.status() == "off"
    monkeypatch.setenv("WEB_EXPRESS_MODEL", "nonsense")
    embed.reset()
    assert embed.get_embedder() is None and embed.status().startswith("unavailable")
    embed.reset()


def test_long_page_sampling_fills_every_slot_and_reaches_both_ends():
    for total in (65, 66, 200, 1000):
        chosen = _candidates(list(range(total)), [0.0] * total)  # no keyword matches at all
        assert len(chosen) == len(set(chosen)) == EMBED_PER_PAGE and 0 in chosen and total - 1 in chosen
    keyword = [0.0] * 200
    keyword[150], keyword[7] = 3.0, 1.0
    chosen = _candidates(sorted(range(200), key=lambda i: (-keyword[i], i)), keyword)
    assert chosen[:2] == [150, 7] and len(chosen) == len(set(chosen)) == EMBED_PER_PAGE and 199 in chosen


def test_broken_vectors_rank_last_instead_of_scrambling_the_order():
    class Broken:
        def embed(self, texts):
            out = np.zeros((len(texts), 2), dtype=np.float32)
            out[0] = [1, 0]
            out[1] = [np.nan, 0]
            out[2] = [0.9, 0]
            return out

    assert rerank("q", ["nan one", "good one"], Broken()) == [1, 0]


# --- loading state ------------------------------------------------------------

def wait_for(predicate, seconds=3.0):
    import time

    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_a_loader_started_before_a_reset_cannot_overwrite_newer_state(monkeypatch):
    import threading

    release = threading.Event()

    def slow_install(spec):
        release.wait(3)
        return embed.model_dir(spec)

    monkeypatch.setenv("WEB_EXPRESS_MODEL", "xsmall")
    monkeypatch.setattr(embed, "install", slow_install)
    monkeypatch.setattr(embed, "Embedder", lambda spec, folder: object())
    embed.reset()
    assert embed.get_embedder() is None and embed.status() == "downloading"
    embed.reset()
    monkeypatch.setenv("WEB_EXPRESS_MODEL", "off")
    assert embed.get_embedder() is None and embed.status() == "off"
    release.set()
    assert not wait_for(lambda: embed.status() != "off", 0.3)
    assert embed.get_embedder() is None


def test_corrupt_files_are_replaced_and_a_failed_load_is_retried_later(monkeypatch):
    calls = []

    def failing_then_fine(spec):
        calls.append("install")
        if len(calls) == 1:
            raise OSError("network down")
        return embed.model_dir(spec)

    monkeypatch.setenv("WEB_EXPRESS_MODEL", "xsmall")
    monkeypatch.setattr(embed, "install", failing_then_fine)
    monkeypatch.setattr(embed, "Embedder", lambda spec, folder: "the embedder")
    embed.reset()
    assert embed.get_embedder() is None
    assert wait_for(lambda: embed.status().startswith("unavailable"))
    assert embed.get_embedder() is None and calls == ["install"]  # not retried straight away

    monkeypatch.setattr(embed, "_retry_at", 0.0)  # as if the delay has passed
    assert embed.get_embedder() is None
    assert wait_for(lambda: embed.status() == "ready") and embed.get_embedder() == "the embedder"


def test_install_redownloads_a_file_that_is_present_but_wrong(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_EXPRESS_HOME", str(tmp_path))
    spec = spec_for({"model.onnx": ("onnx/m.onnx", hashlib.sha256(b"weights").hexdigest())})
    folder = embed.model_dir(spec)
    folder.mkdir(parents=True)
    (folder / "model.onnx").write_bytes(b"garbage")
    assert embed.is_installed(spec)  # present, but that proves nothing
    embed.install(spec, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"weights")))
    assert (folder / "model.onnx").read_bytes() == b"weights"
