"""Request/response schemas for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from docrag.models import Citation


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str = "ok"
    embeddings_provider: str
    llm_provider: str
    vectorstore_provider: str
    indexed_chunks: int


class IngestResponse(BaseModel):
    """Result of ingesting a single document."""

    source: str
    chunks: int
    pages: int


class QueryRequest(BaseModel):
    """A natural-language question over the indexed documents."""

    question: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(None, ge=1, le=50, description="Override the number of chunks.")
    stream: bool = Field(True, description="Stream NDJSON token events when true.")


class QueryResponse(BaseModel):
    """A non-streaming grounded answer with citations."""

    answer: str
    citations: list[Citation]
