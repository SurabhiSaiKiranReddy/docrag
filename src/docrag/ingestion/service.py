"""High-level ingestion service: load -> chunk -> embed -> index.

This orchestrates the ingestion-side providers so the API and scripts share one
code path. It depends only on the abstract :class:`Embeddings` and
:class:`VectorStore` interfaces.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from docrag.embeddings.base import Embeddings
from docrag.ingestion.chunker import TokenChunker
from docrag.ingestion.loaders import load_bytes, load_document
from docrag.vectorstore.base import VectorStore


class IngestResult(BaseModel):
    """Summary of a single document ingestion."""

    source: str
    chunks: int
    pages: int


class IngestionService:
    """Coordinates parsing, chunking, embedding, and indexing of documents."""

    def __init__(
        self,
        embeddings: Embeddings,
        vectorstore: VectorStore,
        chunker: TokenChunker,
        embed_batch_size: int = 64,
    ) -> None:
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.chunker = chunker
        self.embed_batch_size = embed_batch_size

    def _index_document(self, document) -> IngestResult:
        chunks = self.chunker.chunk_document(document)
        for start in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[start : start + self.embed_batch_size]
            vectors = self.embeddings.embed_documents([chunk.text for chunk in batch])
            self.vectorstore.add(batch, vectors)
        self.vectorstore.persist()
        return IngestResult(
            source=document.source,
            chunks=len(chunks),
            pages=len(document.pages),
        )

    def ingest_bytes(self, filename: str, data: bytes) -> IngestResult:
        """Ingest a document provided as raw bytes (e.g. an API upload)."""
        return self._index_document(load_bytes(filename, data))

    def ingest_path(self, path: Path | str) -> IngestResult:
        """Ingest a document from a filesystem path."""
        return self._index_document(load_document(path))
