"""Split text into passages and pick the ones that answer a query.

Passages are exact slices of the document text, so every highlight is verbatim
and its start/end offsets can be checked against the page.

Ranking here is BM25 (keyword based). It is the dependency-free baseline; the
semantic ranker is a later layer that plugs in at rank_passages().
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

TARGET_CHARS = 500  # merge small blocks up to about this size
MAX_CHARS = 900  # split blocks larger than this

_BLOCK = re.compile(r"[^\n]+(?:\n[^\n]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[^\W_]+(?:[._'+#-][^\W_]+)*")
# Chinese, Japanese and Korean are written without spaces: match them by character pairs.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i in is it its of on or "
    "that the their this to was what when where which who why will with you your".split()
)


@dataclass(frozen=True)
class Passage:
    start: int
    end: int
    text: str


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WORD.findall(text.casefold()):
        position = 0
        for run in _CJK.finditer(word):
            if run.start() > position:
                tokens.append(word[position : run.start()])
            chars = run.group()
            tokens.extend(chars[i : i + 2] for i in range(len(chars) - 1)) if len(chars) > 1 else tokens.append(chars)
            position = run.end()
        if position < len(word):
            tokens.append(word[position:])
    return tokens


def split_passages(text: str) -> list[Passage]:
    """Cut text into passages of roughly TARGET_CHARS, on paragraph boundaries."""
    spans: list[tuple[int, int]] = []
    for block in _BLOCK.finditer(text):
        start, end = block.span()
        if end - start <= MAX_CHARS:
            spans.append((start, end))
            continue
        # An oversized block: cut at sentence ends, or hard-cut if there are none.
        piece_start = start
        cuts = [m.end() + start for m in _SENTENCE_END.finditer(text[start:end])] + [end]
        last_cut = start
        for cut in cuts:
            if cut - piece_start > MAX_CHARS:
                if last_cut > piece_start:
                    spans.append((piece_start, last_cut))
                    piece_start = last_cut
                while cut - piece_start > MAX_CHARS:
                    spans.append((piece_start, piece_start + MAX_CHARS))
                    piece_start += MAX_CHARS
            last_cut = cut
        if end > piece_start:
            spans.append((piece_start, end))

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and end - merged[-1][0] <= TARGET_CHARS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    passages = []
    for start, end in merged:
        chunk = text[start:end]
        stripped = chunk.strip()
        if stripped:
            offset = start + chunk.index(stripped)
            passages.append(Passage(offset, offset + len(stripped), stripped))
    return passages


def rank_passages(query: str, passages: list[Passage], k1: float = 1.4, b: float = 0.75) -> list[float]:
    """BM25 score of each passage against the query (0 means no query word matched)."""
    terms = [t for t in dict.fromkeys(tokenize(query)) if t not in _STOPWORDS]
    if not terms or not passages:
        return [0.0] * len(passages)
    counts = [Counter(tokenize(p.text)) for p in passages]
    lengths = [sum(c.values()) for c in counts]
    average = (sum(lengths) / len(lengths)) or 1.0
    n = len(passages)
    idf = {}
    for term in terms:
        df = sum(1 for c in counts if term in c)
        idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
    scores = []
    for count, length in zip(counts, lengths):
        score = 0.0
        for term in terms:
            tf = count.get(term, 0)
            if tf:
                score += idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


def top_passages(query: str, text: str, limit: int) -> list[Passage]:
    """The passages of text that best match query, best first."""
    passages = split_passages(text)
    scores = rank_passages(query, passages)
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    return [passages[i] for i in order[:limit] if scores[i] > 0]
