"""End-to-end API tests using offline fakes injected via dependency override."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docrag.api.deps import Services, get_services
from docrag.api.main import app
from docrag.config import get_settings
from docrag.ingestion.chunker import TokenChunker
from docrag.ingestion.service import IngestionService
from docrag.rag.pipeline import RagPipeline
from docrag.vectorstore.faiss_store import FaissVectorStore

from .conftest import FakeEmbeddings, FakeLLM


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    embeddings = FakeEmbeddings()
    vectorstore = FaissVectorStore(index_dir=tmp_path)
    llm = FakeLLM(answer="Raw events are retained for 90 days. [source: nimbus.md #0]")
    chunker = TokenChunker(chunk_size=128, chunk_overlap=16)
    services = Services(
        settings=get_settings(),
        embeddings=embeddings,
        vectorstore=vectorstore,
        llm=llm,
        ingestion=IngestionService(embeddings, vectorstore, chunker),
        pipeline=RagPipeline(embeddings, vectorstore, llm, top_k=3),
    )
    app.dependency_overrides[get_services] = lambda: services
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == 0


def test_ingest_then_query_json(client: TestClient) -> None:
    content = b"Nimbus retains raw events for 90 days. The collector flushes every 5 seconds."
    ingest = client.post(
        "/ingest",
        files={"file": ("nimbus.md", content, "text/markdown")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] >= 1

    health = client.get("/health").json()
    assert health["indexed_chunks"] >= 1

    response = client.post(
        "/query",
        json={"question": "how long are events retained?", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert "90 days" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["source"] == "nimbus.md"


def test_query_streaming_ndjson(client: TestClient) -> None:
    content = b"Nimbus retains raw events for 90 days."
    client.post("/ingest", files={"file": ("nimbus.md", content, "text/markdown")})

    with client.stream(
        "POST", "/query", json={"question": "retention period?", "stream": True}
    ) as response:
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]

    types = [event["type"] for event in events]
    assert types[0] == "sources"
    assert "token" in types
    assert types[-1] == "done"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "90 days" in answer


def test_ingest_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert response.status_code == 415


def test_ingest_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/ingest",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert response.status_code == 400
