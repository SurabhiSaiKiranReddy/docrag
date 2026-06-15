"""Abstract embedding provider interface.

Concrete implementations (local sentence-transformers, OpenAI, ...) are selected
at runtime via :func:`docrag.factory.build_embeddings`. Keeping this an ABC is
what lets the rest of the system depend on the abstraction rather than a vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embeddings(ABC):
    """Turns text into dense vectors for similarity search."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks.

        Implementations should return L2-normalized vectors so that an inner
        product equals cosine similarity downstream.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
