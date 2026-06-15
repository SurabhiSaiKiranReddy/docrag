"""Tests for evaluation metrics, dataset loading, and the retrieval harness."""

from __future__ import annotations

from pathlib import Path

from docrag.eval.dataset import load_dataset
from docrag.eval.harness import evaluate_retrieval
from docrag.eval.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from docrag.models import Chunk, ChunkMetadata
from docrag.rag.retriever import Retriever
from docrag.vectorstore.faiss_store import FaissVectorStore

from .conftest import FakeEmbeddings


def test_hit_at_k() -> None:
    assert hit_at_k(["a.md", "b.md"], {"b.md"}, 1) == 0.0
    assert hit_at_k(["a.md", "b.md"], {"b.md"}, 2) == 1.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a.md", "b.md", "c.md"], {"c.md"}) == 1 / 3
    assert reciprocal_rank(["a.md"], {"z.md"}) == 0.0


def test_recall_and_precision_at_k() -> None:
    retrieved = ["a.md", "b.md", "x.md"]
    relevant = {"a.md", "b.md"}
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert recall_at_k(retrieved, relevant, 1) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_load_dataset_real_file() -> None:
    dataset = load_dataset()
    assert len(dataset) >= 5
    for item in dataset:
        assert item.question and item.answer and item.sources


def test_evaluate_retrieval_perfect_when_sources_align(
    tmp_path: Path, fake_embeddings: FakeEmbeddings
) -> None:
    from docrag.eval.dataset import QAItem

    store = FaissVectorStore(index_dir=tmp_path)
    texts = {
        "retention.md": "Nimbus raw events retention is 90 days.",
        "collector.md": "The collector flush interval is 5 seconds.",
    }
    chunks, vectors = [], []
    for source, text in texts.items():
        chunks.append(
            Chunk(id=f"{source}::0", text=text, metadata=ChunkMetadata(source=source, chunk_id=0))
        )
        vectors.append(fake_embeddings.embed_query(text))
    store.add(chunks, vectors)

    dataset = [
        QAItem(question="events retention days", answer="90 days", sources=["retention.md"]),
        QAItem(question="collector flush seconds", answer="5 seconds", sources=["collector.md"]),
    ]
    report = evaluate_retrieval(
        Retriever(fake_embeddings, store), dataset, indexed_chunks=2, ks=(1,)
    )

    assert report.hit_at_k[1] == 1.0
    assert report.mrr == 1.0
    assert "Hit@k" in report.as_markdown()
