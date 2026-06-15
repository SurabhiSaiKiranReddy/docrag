"""Shared data models used across ingestion, retrieval, and generation.

These are deliberately decoupled from the API schemas (:mod:`docrag.api.schemas`)
so internal logic and the public contract can evolve independently.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Provenance for a single chunk, used to build citations."""

    source: str = Field(..., description="Originating filename, e.g. 'report.pdf'.")
    chunk_id: int = Field(..., description="0-based index of this chunk within its document.")
    page: int | None = Field(None, description="1-based source page number, when known.")


class Chunk(BaseModel):
    """A unit of text indexed in the vector store."""

    id: str = Field(..., description="Stable unique id, e.g. 'report.pdf::3'.")
    text: str
    metadata: ChunkMetadata

    @property
    def citation(self) -> str:
        """Human-readable citation marker, e.g. ``[source: report.pdf #3]``."""
        return f"[source: {self.metadata.source} #{self.metadata.chunk_id}]"


class ScoredChunk(BaseModel):
    """A chunk returned from retrieval together with its similarity score."""

    chunk: Chunk
    score: float = Field(..., description="Similarity score; higher is more relevant.")
