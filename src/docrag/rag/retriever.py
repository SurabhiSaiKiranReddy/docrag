"""Retrieval: embed the query and fetch the most similar chunks.

The MVP :class:`Retriever` does dense (vector) retrieval. :class:`HybridRetriever`
adds optional BM25 keyword fusion (via reciprocal rank fusion) and optional
cross-encoder reranking, while preserving the same ``retrieve(query, k)``
interface so the pipeline is unaffected.
"""

from __future__ import annotations

import re

from docrag.embeddings.base import Embeddings
from docrag.models import Chunk, ScoredChunk
from docrag.rag.fusion import reciprocal_rank_fusion
from docrag.rag.rerank import Reranker
from docrag.vectorstore.base import VectorStore

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Retriever:
    """Dense vector retriever over a :class:`VectorStore`."""

    def __init__(self, embeddings: Embeddings, vectorstore: VectorStore) -> None:
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return the top-``k`` chunks most similar to ``query``."""
        if not query.strip():
            return []
        query_vector = self.embeddings.embed_query(query)
        return self.vectorstore.search(query_vector, k)


class HybridRetriever(Retriever):
    """Dense retrieval optionally fused with BM25 and reranked.

    Pipeline: gather a candidate pool from the vector store (and, when enabled,
    a BM25 keyword ranking), fuse the rankings with RRF, then optionally rerank
    the fused candidates with a cross-encoder before truncating to ``k``.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        vectorstore: VectorStore,
        *,
        use_bm25: bool = True,
        reranker: Reranker | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(embeddings, vectorstore)
        self.use_bm25 = use_bm25
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._bm25 = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_count = -1

    def _ensure_bm25(self, chunks: list[Chunk]) -> None:
        # Rebuild only when the corpus size changes (e.g. after new ingestion).
        if self._bm25 is not None and self._bm25_count == len(chunks):
            return
        from rank_bm25 import BM25Okapi

        self._bm25_chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(chunk.text) for chunk in chunks])
        self._bm25_count = len(chunks)

    def _bm25_ranking(self, query: str, chunks: list[Chunk]) -> list[str]:
        self._ensure_bm25(chunks)
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i].id for i in order[: self.candidates]]

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if not query.strip():
            return []

        dense = super().retrieve(query, self.candidates)
        by_id: dict[str, Chunk] = {scored.chunk.id: scored.chunk for scored in dense}

        rankings: list[list[str]] = [[scored.chunk.id for scored in dense]]
        if self.use_bm25:
            all_chunks = self.vectorstore.all_chunks()
            if all_chunks:
                for chunk in all_chunks:
                    by_id.setdefault(chunk.id, chunk)
                rankings.append(self._bm25_ranking(query, all_chunks))

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        fused_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        candidate_chunks = [by_id[cid] for cid in fused_ids[: self.candidates] if cid in by_id]

        if self.reranker is not None:
            return self.reranker.rerank(query, candidate_chunks, k)

        return [
            ScoredChunk(chunk=by_id[cid], score=fused[cid])
            for cid in fused_ids[:k]
            if cid in by_id
        ]
