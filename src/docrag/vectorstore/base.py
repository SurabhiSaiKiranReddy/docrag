"""Abstract vector store interface.

A store ingests chunks plus their embeddings and answers similarity queries.
Concrete backends (FAISS now; pgvector/Pinecone are natural future additions)
are selected via :func:`docrag.factory.build_vectorstore`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from docrag.models import Chunk, ScoredChunk


class VectorStore(ABC):
    """Persistent index supporting add + similarity search over chunks."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Index ``chunks`` with their corresponding ``embeddings`` (parallel lists)."""

    @abstractmethod
    def search(self, query_embedding: list[float], k: int) -> list[ScoredChunk]:
        """Return the ``k`` most similar chunks to ``query_embedding``."""

    @abstractmethod
    def persist(self) -> None:
        """Flush the index and metadata to durable storage."""

    @abstractmethod
    def load(self) -> None:
        """Load a previously persisted index, if one exists."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed chunks."""

    @abstractmethod
    def all_chunks(self) -> list[Chunk]:
        """Return every indexed chunk (used by keyword/hybrid retrieval)."""
