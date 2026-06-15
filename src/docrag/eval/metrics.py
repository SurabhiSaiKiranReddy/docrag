"""Pure retrieval-quality metric functions (no I/O, easily testable).

Relevance is evaluated at the *source-document* level: a retrieved chunk counts
as relevant when its source filename is in the gold ``relevant`` set for the
question.
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any of the top-``k`` retrieved sources is relevant, else 0.0."""
    return 1.0 if any(source in relevant for source in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the rank (1-based) of the first relevant source; 0 if none."""
    for index, source in enumerate(retrieved):
        if source in relevant:
            return 1.0 / (index + 1)
    return 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant sources present in the top-``k`` retrieved sources."""
    if not relevant:
        return 0.0
    found = {source for source in retrieved[:k] if source in relevant}
    return len(found) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` retrieved sources that are relevant."""
    if k <= 0:
        return 0.0
    hits = sum(1 for source in retrieved[:k] if source in relevant)
    return hits / k
