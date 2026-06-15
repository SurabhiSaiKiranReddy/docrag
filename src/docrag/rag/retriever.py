"""Retrieval: embed the query and fetch the most similar chunks.

For the MVP this is dense (vector) retrieval. Hybrid search (BM25 + vector with
reciprocal-rank fusion) and cross-encoder reranking are layered on in a later
phase without changing this interface.
"""

from __future__ import annotations

from docrag.embeddings.base import Embeddings
from docrag.models import ScoredChunk
from docrag.vectorstore.base import VectorStore


class Retriever:
    """Dense vector retriever over a :class:`VectorStore`."""

    def __init__(self, embeddings: Embeddings, vectorstore: VectorStore) -> None:
        self.embeddings = embeddings
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return the top-``k`` chunks most similar to ``query``."""
        if not query.strip():
            return []
        query_vector = self.embeddings.embed_query(query)
        return self.vectorstore.search(query_vector, k)
