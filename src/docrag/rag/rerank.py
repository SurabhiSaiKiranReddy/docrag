"""Cross-encoder reranking.

A bi-encoder (the embedding model) retrieves cheaply but approximately. A
cross-encoder jointly encodes the (query, chunk) pair and scores relevance far
more accurately — too slow for first-stage retrieval, ideal for reordering a
small candidate pool.

A :class:`Reranker` is any object with a ``rerank(query, chunks, top_n)`` method,
so tests can inject a deterministic fake without downloading a model.
"""

from __future__ import annotations

from typing import Protocol

from docrag.models import Chunk, ScoredChunk


class Reranker(Protocol):
    """Reorders candidate chunks by relevance to the query."""

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Reranker backed by a ``sentence-transformers`` CrossEncoder (CPU-friendly)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [ScoredChunk(chunk=chunk, score=float(score)) for chunk, score in ranked[:top_n]]
