"""A small local embedding model, so ranking can go by meaning and not only by words.

The model runs on the CPU through ONNX Runtime; no PyTorch and no GPU needed.
It is downloaded once (about 25 MB) into the web_express data folder and checked
against a pinned checksum. Until it is there, callers fall back to keyword
ranking: get_embedder() never blocks a search on a download.

    WEB_EXPRESS_HOME    data folder (default: ~/.web_express)
    WEB_EXPRESS_MODEL   "xsmall" (default) or "off" to disable semantic ranking
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger("web_express")

MAX_TOKENS = 512
BATCH_SIZE = 16
CACHE_ENTRIES = 4096


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str
    files: dict[str, tuple[str, str]]  # local name -> (path in repo, sha256)
    dimensions: int
    output: str  # name of the pooled, sentence-level output


MODELS = {
    "xsmall": ModelSpec(
        name="mxbai-embed-xsmall-v1",
        repo="mixedbread-ai/mxbai-embed-xsmall-v1",
        revision="e6ac24e5d6efb8782b59de1647b3ececb4ece94e",
        files={
            "model.onnx": ("onnx/model_int8.onnx", "952f996d8cf46c311ee8654a750fa942b71c8b94aabe69d043dbb2bcaff5528e"),
            "tokenizer.json": ("tokenizer.json", "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
        },
        dimensions=384,
        output="sentence_embedding",
    ),
}


def data_home() -> Path:
    custom = os.environ.get("WEB_EXPRESS_HOME", "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".web_express"


def model_dir(spec: ModelSpec) -> Path:
    return data_home() / "models" / f"{spec.name}-{spec.revision[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_installed(spec: ModelSpec) -> bool:
    folder = model_dir(spec)
    return all((folder / name).is_file() for name in spec.files)


def install(spec: ModelSpec, transport: httpx.BaseTransport | None = None) -> Path:
    """Download the model files, verify each against its checksum, and return the folder."""
    folder = model_dir(spec)
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60, transport=transport) as client:
        for name, (remote, expected) in spec.files.items():
            target = folder / name
            if target.is_file() and _sha256(target) == expected:
                continue
            partial = folder / f"{name}.part-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            digest = hashlib.sha256()
            try:
                url = f"https://huggingface.co/{spec.repo}/resolve/{spec.revision}/{remote}"
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for block in response.iter_bytes():
                            handle.write(block)
                            digest.update(block)
                if digest.hexdigest() != expected:
                    raise RuntimeError(f"{name} failed its checksum; the download was discarded.")
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
    return folder


class Embedder:
    """Turns texts into unit-length vectors; the dot product of two is their similarity."""

    def __init__(self, spec: ModelSpec, folder: Path) -> None:
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer

        self._np = numpy
        self.spec = spec
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            str(folder / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._cache: OrderedDict[bytes, object] = OrderedDict()
        self._lock = threading.Lock()

    def embed(self, texts: list[str]):
        """Return an array of shape (len(texts), dimensions), one unit vector per text."""
        np = self._np
        keys = [hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=16).digest() for t in texts]
        result = np.zeros((len(texts), self.spec.dimensions), dtype=np.float32)
        missing: list[int] = []
        with self._lock:
            for i, key in enumerate(keys):
                vector = self._cache.get(key)
                if vector is None:
                    missing.append(i)
                else:
                    self._cache.move_to_end(key)
                    result[i] = vector
        missing.sort(key=lambda i: len(texts[i]))  # similar lengths together: less padding
        for offset in range(0, len(missing), BATCH_SIZE):
            batch = missing[offset : offset + BATCH_SIZE]
            encoded = self._tokenizer.encode_batch([texts[i] or " " for i in batch])
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
            }
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
            vectors = np.nan_to_num(self._session.run([self.spec.output], feed)[0].astype(np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0)
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            with self._lock:
                for i, vector in zip(batch, vectors):
                    result[i] = vector
                    self._cache[keys[i]] = vector
                while len(self._cache) > CACHE_ENTRIES:
                    self._cache.popitem(last=False)
        return result


_state_lock = threading.Lock()
_install_lock = threading.Lock()  # one download at a time in this process
_embedder: Embedder | None = None
_status = "not loaded"  # not loaded | loading | downloading | ready | off | unavailable: <reason>
_generation = 0  # bumped by reset(); a loader started before a reset must not commit its result
_failures = 0
_retry_at = 0.0  # monotonic time before which a failed load is not retried
RETRY_BASE = 60.0
RETRY_MAX = 3600.0


def status() -> str:
    return _status


def _prepare(spec: ModelSpec, generation: int) -> None:
    global _embedder, _status, _failures, _retry_at
    try:
        with _install_lock:
            folder = install(spec)  # verifies checksums; downloads only what is missing or corrupt
        embedder = Embedder(spec, folder)
        with _state_lock:
            if generation == _generation:
                _embedder, _status, _failures = embedder, "ready", 0
    except Exception as exc:  # no network, unsupported platform, broken file: keep working without it
        log.warning("semantic ranking is unavailable: %s", exc)
        with _state_lock:
            if generation == _generation:
                _failures += 1
                _retry_at = time.monotonic() + min(RETRY_BASE * 2 ** (_failures - 1), RETRY_MAX)
                _status = f"unavailable: {type(exc).__name__}"


def get_embedder() -> Embedder | None:
    """The embedder if it is ready, else None.

    Starts loading in the background on first call, and again after a failure
    once a growing delay has passed. Never blocks.
    """
    global _status
    if _embedder is not None:
        return _embedder
    with _state_lock:
        if _embedder is not None:
            return _embedder
        failed = _status.startswith("unavailable") and _failures > 0
        if _status != "not loaded" and not (failed and time.monotonic() >= _retry_at):
            return None
        choice = os.environ.get("WEB_EXPRESS_MODEL", "xsmall").strip().lower()
        spec = MODELS.get(choice)
        if spec is None:
            _status = "off" if choice == "off" else f"unavailable: unknown model {choice!r}"
            return None
        _status = "loading" if is_installed(spec) else "downloading"
        generation = _generation
    threading.Thread(target=_prepare, args=(spec, generation), name="web_express-model", daemon=True).start()
    return None


def reset() -> None:
    """Forget the loaded model (used by tests and by `web-express setup`)."""
    global _embedder, _status, _generation, _failures, _retry_at
    with _state_lock:
        _generation += 1
        _embedder, _status, _failures, _retry_at = None, "not loaded", 0, 0.0
