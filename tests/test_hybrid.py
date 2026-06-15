"""Tests for RRF fusion and the hybrid retriever (with a fake reranker)."""

from __future__ import annotations

from pathlib import Path

from docrag.models import Chunk, ChunkMetadata, ScoredChunk
from docrag.rag.fusion import reciprocal_rank_fusion
from docrag.rag.retriever import HybridRetriever
from docrag.vectorstore.faiss_store import FaissVectorStore

from .conftest import FakeEmbeddings


def test_rrf_rewards_items_ranked_highly_in_multiple_lists() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    # "b" is 1st + 2nd, "a" is 2nd + 1st -> they should top "c"/"d" (single list).
    ranked = sorted(fused, key=lambda key: fused[key], reverse=True)
    assert set(ranked[:2]) == {"a", "b"}
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]


def test_rrf_empty_input_is_empty() -> None:
    assert reciprocal_rank_fusion([]) == {}


def _store(tmp_path: Path, embeddings: FakeEmbeddings) -> FaissVectorStore:
    store = FaissVectorStore(index_dir=tmp_path)
    texts = [
        "Nimbus raw events retention is 90 days.",
        "The collector flush interval is 5 seconds.",
        "Enterprise pricing is custom per region.",
    ]
    chunks = [
        Chunk(id=f"d::{i}", text=t, metadata=ChunkMetadata(source="d.md", chunk_id=i))
        for i, t in enumerate(texts)
    ]
    store.add(chunks, embeddings.embed_documents(texts))
    return store


def test_hybrid_retriever_returns_relevant_chunk_first(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    store = _store(tmp_path, fake_embeddings)
    retriever = HybridRetriever(fake_embeddings, store, use_bm25=True, candidates=10)

    results = retriever.retrieve("events retention days", k=2)

    assert results
    assert results[0].chunk.id == "d::0"


def test_hybrid_retriever_applies_reranker(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    store = _store(tmp_path, fake_embeddings)

    class KeywordReranker:
        """Deterministic fake: ranks chunks by count of the word 'pricing'."""

        def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[ScoredChunk]:
            scored = [
                ScoredChunk(chunk=c, score=float(c.text.lower().count("pricing"))) for c in chunks
            ]
            scored.sort(key=lambda s: s.score, reverse=True)
            return scored[:top_n]

    retriever = HybridRetriever(
        fake_embeddings, store, use_bm25=True, reranker=KeywordReranker(), candidates=10
    )

    results = retriever.retrieve("anything", k=1)
    assert results[0].chunk.id == "d::2"  # the chunk containing "pricing"


def test_hybrid_retriever_rebuilds_bm25_after_new_documents(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    store = _store(tmp_path, fake_embeddings)
    retriever = HybridRetriever(fake_embeddings, store, use_bm25=True, candidates=10)
    retriever.retrieve("events", k=1)  # builds BM25 over 3 chunks

    new_texts = ["A brand new uptime guarantee document."]
    new_chunks = [
        Chunk(id="d::3", text=new_texts[0], metadata=ChunkMetadata(source="d.md", chunk_id=3))
    ]
    store.add(new_chunks, fake_embeddings.embed_documents(new_texts))

    results = retriever.retrieve("uptime guarantee", k=1)
    assert results[0].chunk.id == "d::3"
