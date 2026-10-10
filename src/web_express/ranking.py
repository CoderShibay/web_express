"""Ranking that combines keyword match with meaning.

Each function takes an optional embedder. With one, meaning leads and keyword
match adds a bonus; without one, keyword rank is used alone, so everything
works before the model has been downloaded.
"""

from __future__ import annotations

from .passages import Passage, rank_passages, split_passages

EMBED_PER_PAGE = 64  # passages embedded per page; a longer page is sampled
KEYWORD_BONUS = 0.15  # added to similarity for the best keyword match, less for weaker ones
ORDER_BONUS = 0.10  # added to similarity for the top incoming result, less down the list
MIN_SIMILARITY = 0.35  # below this a passage is not about the query


def _candidates(keyword_order: list[int], total: int) -> list[int]:
    """Which passages to embed: all of a short page; the best keyword matches plus an even sample of a long one."""
    if total <= EMBED_PER_PAGE:
        return list(range(total))
    keep = keyword_order[: EMBED_PER_PAGE * 2 // 3]
    step = total / (EMBED_PER_PAGE - len(keep))
    sample = [int(i * step) for i in range(EMBED_PER_PAGE - len(keep))]
    return list(dict.fromkeys(keep + sample))


def best_passages(query: str, text: str, limit: int, embedder=None) -> list[Passage]:
    """The passages of text that best answer query, best first."""
    passages = split_passages(text)
    if not passages:
        return []
    keyword = rank_passages(query, passages)
    keyword_order = sorted(range(len(passages)), key=lambda i: (-keyword[i], i))
    if embedder is None:
        return [passages[i] for i in keyword_order[:limit] if keyword[i] > 0]

    candidates = _candidates(keyword_order, len(passages))
    vectors = embedder.embed([query] + [passages[i].text for i in candidates])
    similarity = (vectors[1:] @ vectors[0]).tolist()
    top_keyword = max(keyword) or 1.0
    scored = []
    for j, i in enumerate(candidates):
        share = keyword[i] / top_keyword
        if similarity[j] >= MIN_SIMILARITY or share >= 0.5:
            scored.append((similarity[j] + KEYWORD_BONUS * share, -i))
    scored.sort(reverse=True)
    return [passages[-i] for _, i in scored[:limit]]


def rerank(query: str, texts: list[str], embedder=None) -> list[int]:
    """Order of texts (indexes, best first) after blending their given order with meaning.

    texts arrive in their current rank order. Without an embedder the order is unchanged.
    """
    count = len(texts)
    if embedder is None or count < 2:
        return list(range(count))
    vectors = embedder.embed([query] + texts)
    similarity = (vectors[1:] @ vectors[0]).tolist()
    score = [similarity[i] + ORDER_BONUS * (1 - i / (count - 1)) for i in range(count)]
    return sorted(range(count), key=lambda i: (-score[i], i))
