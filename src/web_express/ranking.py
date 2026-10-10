"""Ranking that combines keyword match with meaning.

Each function takes an optional embedder. With one, meaning leads and keyword
match adds a bonus; without one, keyword rank is used alone, so everything
works before the model has been downloaded.
"""

from __future__ import annotations

import math

from .passages import Passage, rank_passages, split_passages

EMBED_PER_PAGE = 64  # passages embedded per page; a longer page is sampled
KEYWORD_BONUS = 0.15  # added to similarity for the best keyword match, less for weaker ones
ORDER_BONUS = 0.10  # added to similarity for the top incoming result, less down the list
MIN_SIMILARITY = 0.35  # below this a passage is not about the query


def _candidates(keyword_order: list[int], keyword: list[float]) -> list[int]:
    """Which passages to embed: all of a short page; for a long one, the best keyword
    matches plus an even sample of the rest, from the first passage to the last."""
    total = len(keyword)
    if total <= EMBED_PER_PAGE:
        return list(range(total))
    keep = [i for i in keyword_order[: EMBED_PER_PAGE * 2 // 3] if keyword[i] > 0]
    chosen = set(keep)
    rest = [i for i in range(total) if i not in chosen]
    wanted = EMBED_PER_PAGE - len(keep)
    if wanted >= len(rest):
        sample = rest
    elif wanted == 1:
        sample = rest[:1]
    else:
        sample = [rest[round(j * (len(rest) - 1) / (wanted - 1))] for j in range(wanted)]
    return keep + list(dict.fromkeys(sample))


def _similarities(vectors) -> list[float]:
    """Similarity of each text to the query (row 0); a broken value counts as unrelated."""
    return [s if math.isfinite(s) else -1.0 for s in (vectors[1:] @ vectors[0]).tolist()]


def best_passages(query: str, text: str, limit: int, embedder=None) -> list[Passage]:
    """The passages of text that best answer query, best first."""
    passages = split_passages(text)
    if not passages:
        return []
    keyword = rank_passages(query, passages)
    keyword_order = sorted(range(len(passages)), key=lambda i: (-keyword[i], i))
    if embedder is None:
        return [passages[i] for i in keyword_order[:limit] if keyword[i] > 0]

    candidates = _candidates(keyword_order, keyword)
    similarity = _similarities(embedder.embed([query] + [passages[i].text for i in candidates]))
    top_keyword = max(keyword) or 1.0
    scored = []
    for j, i in enumerate(candidates):
        share = keyword[i] / top_keyword
        if similarity[j] >= MIN_SIMILARITY or share >= 0.5:
            scored.append((similarity[j] + KEYWORD_BONUS * share, -i))
    scored.sort(reverse=True)
    return [passages[-i] for _, i in scored[:limit]]


def rerank(query: str, texts: list[str], embedder=None, bonus: list[float] | None = None) -> list[int]:
    """Order of texts (indexes, best first) after blending their given order with meaning.

    texts arrive in their current rank order. bonus, if given, adds a small amount per
    text (for example for a well-linked site). Without an embedder the order is unchanged.
    """
    count = len(texts)
    if embedder is None or count < 2:
        return list(range(count))
    similarity = _similarities(embedder.embed([query] + texts))
    extra = bonus if bonus is not None and len(bonus) == count else [0.0] * count
    score = [similarity[i] + ORDER_BONUS * (1 - i / (count - 1)) + extra[i] for i in range(count)]
    return sorted(range(count), key=lambda i: (-score[i], i))
