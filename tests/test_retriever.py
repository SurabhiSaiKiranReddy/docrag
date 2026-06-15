"""Tests for the dense retriever over a real FAISS store with fake embeddings."""

from __future__ import annotations

from pathlib import Path

from docrag.models import Chunk, ChunkMetadata
from docrag.rag.retriever import Retriever
from docrag.vectorstore.faiss_store import FaissVectorStore

from .conftest import FakeEmbeddings


def _index(tmp_path: Path, embeddings: FakeEmbeddings) -> FaissVectorStore:
    store = FaissVectorStore(index_dir=tmp_path)
    texts = {
        "retention": "Nimbus events retention is 90 days for raw events.",
        "collector": "The collector flush happens every 5 seconds.",
        "uptime": "Enterprise uptime is guaranteed in every region.",
    }
    chunks, vectors = [], []
    for i, (key, text) in enumerate(texts.items()):
        chunks.append(
            Chunk(id=f"doc::{i}", text=text, metadata=ChunkMetadata(source=key, chunk_id=i))
        )
        vectors.append(embeddings.embed_query(text))
    store.add(chunks, vectors)
    return store


def test_retriever_ranks_relevant_chunk_first(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    store = _index(tmp_path, fake_embeddings)
    retriever = Retriever(fake_embeddings, store)

    results = retriever.retrieve("how long is event retention in days", k=3)

    assert results
    assert results[0].chunk.metadata.source == "retention"
    # scores should be in non-increasing order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_respects_k(tmp_path: Path, fake_embeddings: FakeEmbeddings) -> None:
    store = _index(tmp_path, fake_embeddings)
    retriever = Retriever(fake_embeddings, store)

    assert len(retriever.retrieve("events", k=2)) == 2


def test_retriever_empty_query_returns_nothing(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    store = _index(tmp_path, fake_embeddings)
    retriever = Retriever(fake_embeddings, store)

    assert retriever.retrieve("   ", k=3) == []
