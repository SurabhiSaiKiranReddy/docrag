"""Tests for the RAG pipeline orchestration with offline fakes."""

from __future__ import annotations

from pathlib import Path

from docrag.models import Chunk, ChunkMetadata
from docrag.rag.pipeline import RagPipeline
from docrag.rag.prompt import build_prompt
from docrag.vectorstore.faiss_store import FaissVectorStore

from .conftest import FakeEmbeddings, FakeLLM


def _pipeline(tmp_path: Path, embeddings: FakeEmbeddings, llm: FakeLLM) -> RagPipeline:
    store = FaissVectorStore(index_dir=tmp_path)
    texts = [
        "Nimbus events retention is 90 days for raw events.",
        "The collector flush happens every 5 seconds.",
    ]
    chunks = [
        Chunk(id=f"doc::{i}", text=t, metadata=ChunkMetadata(source="nimbus.md", chunk_id=i))
        for i, t in enumerate(texts)
    ]
    store.add(chunks, embeddings.embed_documents(texts))
    return RagPipeline(embeddings, store, llm, top_k=2)


def test_pipeline_streams_answer_and_exposes_citations(
    tmp_path: Path, fake_embeddings: FakeEmbeddings, fake_llm: FakeLLM
) -> None:
    pipeline = _pipeline(tmp_path, fake_embeddings, fake_llm)

    stream = pipeline.run("how long is event retention in days")
    answer = "".join(stream.tokens)

    assert answer == fake_llm.answer
    assert stream.citations
    assert stream.citations[0].source == "nimbus.md"
    assert stream.citations[0].score >= stream.citations[-1].score


def test_pipeline_passes_grounded_prompt_to_llm(
    tmp_path: Path, fake_embeddings: FakeEmbeddings, fake_llm: FakeLLM
) -> None:
    pipeline = _pipeline(tmp_path, fake_embeddings, fake_llm)

    pipeline.answer("how long is event retention")

    # The user prompt must contain the question and citation markers; the system
    # prompt must instruct grounding.
    assert "how long is event retention" in (fake_llm.last_prompt or "")
    assert "[source: nimbus.md #0]" in (fake_llm.last_prompt or "")
    assert "ONLY" in (fake_llm.last_system or "")


def test_build_prompt_handles_braces_in_documents() -> None:
    from docrag.models import ScoredChunk

    chunk = Chunk(
        id="c::0",
        text='config = {"timeout": 30}',  # braces must not break templating
        metadata=ChunkMetadata(source="c", chunk_id=0),
    )
    system, user = build_prompt("what is the timeout?", [ScoredChunk(chunk=chunk, score=1.0)])

    assert '{"timeout": 30}' in user
    assert "what is the timeout?" in user


def test_pipeline_answer_with_no_documents_still_runs(
    tmp_path: Path, fake_embeddings: FakeEmbeddings, fake_llm: FakeLLM
) -> None:
    store = FaissVectorStore(index_dir=tmp_path)
    pipeline = RagPipeline(fake_embeddings, store, fake_llm, top_k=3)

    result = pipeline.answer("anything")

    assert result.answer == fake_llm.answer
    assert result.citations == []
