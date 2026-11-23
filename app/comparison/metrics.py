"""Framework-independent common retrieval metrics.

These calculations are identical for the custom and framework editions so
that ranking quality can be compared without depending on internal scores.

Relevance is binary for metric purposes: a judgment grade greater than 0
counts as relevant; a grade of 0 (or a missing chunk id) is not relevant.
All metrics return 0.0 (never None) when the calculation cannot produce a
positive signal. A score of None at the result level signals that the metric
was unavailable (e.g. implementation failure), distinct from a genuine 0.0.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def _relevance(relevant: Mapping[str, int], chunk_id: str) -> int:
    return int(relevant.get(chunk_id, 0))


def precision_at_k(retrieved: Iterable[str], relevant: Mapping[str, int], k: int) -> float:
    """Precision@k = (relevant items in top-k) / k.

    Returns 0.0 for k <= 0. Divides by k (not by the number of items
    actually returned) so that a system returning fewer than k items is
    penalized proportionally.
    """
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    hits = sum(1 for cid in top if _relevance(relevant, cid) > 0)
    return hits / k


def recall_at_k(retrieved: Iterable[str], relevant: Mapping[str, int], k: int) -> float:
    """Recall@k = (relevant items in top-k) / (total relevant items).

    Returns 0.0 for k <= 0 or when there are no relevant items.
    """
    if k <= 0:
        return 0.0
    total_relevant = sum(1 for v in relevant.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    top = list(retrieved)[:k]
    hits = sum(1 for cid in top if _relevance(relevant, cid) > 0)
    return hits / total_relevant


def reciprocal_rank(retrieved: Iterable[str], relevant: Mapping[str, int]) -> float:
    """Reciprocal rank = 1 / rank of the first relevant item.

    Returns 0.0 when no relevant item appears in the retrieved list.
    """
    for i, cid in enumerate(retrieved, start=1):
        if _relevance(relevant, cid) > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: Iterable[str], relevant: Mapping[str, int], k: int) -> float:
    """Discounted Cumulative Gain at k.

    gain = (2^rel - 1) / log2(rank + 1). Returns 0.0 for k <= 0.
    """
    if k <= 0:
        return 0.0
    score = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        rel = _relevance(relevant, cid)
        if rel > 0:
            score += (2 ** rel - 1) / math.log2(i + 1)
    return score


def idcg_at_k(relevant: Mapping[str, int], k: int) -> float:
    """Ideal DCG at k: DCG of the relevance grades sorted descending.

    Returns 0.0 for k <= 0 or when there are no relevant items.
    """
    if k <= 0:
        return 0.0
    rels = sorted((v for v in relevant.values() if v > 0), reverse=True)
    if not rels:
        return 0.0
    score = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        score += (2 ** rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: Iterable[str], relevant: Mapping[str, int], k: int) -> float:
    """Normalized DCG at k = DCG@k / IDCG@k.

    Returns 0.0 when IDCG@k is 0 (no relevant items) or k <= 0.
    """
    ideal = idcg_at_k(relevant, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def average(values: list[float]) -> float:
    """Mean of values; 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
