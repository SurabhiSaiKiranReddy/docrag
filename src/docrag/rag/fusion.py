"""Reciprocal Rank Fusion (RRF) for combining multiple rankings.

RRF merges ranked lists without needing comparable scores: each item gets
``sum(1 / (k + rank))`` across the lists it appears in. It is the standard,
score-agnostic way to fuse lexical (BM25) and semantic (vector) results.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Fuse ranked id lists into a combined ``id -> score`` mapping.

    Args:
        rankings: each inner sequence is a ranking of item ids, best first.
        k: smoothing constant; larger values flatten the contribution of rank.

    Returns:
        Mapping of item id to fused score (higher is better).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
