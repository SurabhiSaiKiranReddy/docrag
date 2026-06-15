"""OpenAI embeddings (``text-embedding-3-small`` by default).

Vectors are L2-normalized here so that downstream inner-product search yields
cosine similarity, matching the behavior of :class:`LocalEmbeddings`.
"""

from __future__ import annotations

import math

from .base import Embeddings


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class OpenAIEmbeddings(Embeddings):
    """Embeddings backed by the OpenAI API."""

    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "OpenAI embeddings selected but no API key provided. "
                "Set OPENAI_API_KEY in your environment or .env file."
            )
        self.model = model
        self._client = OpenAI(api_key=api_key)
        self._dimension = self._DIMENSIONS.get(model, 1536)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [_normalize(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model, input=[text])
        return _normalize(response.data[0].embedding)

    @property
    def dimension(self) -> int:
        return self._dimension
