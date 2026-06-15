"""Composition of runtime services shared across API requests.

Ingestion and querying must share the *same* vector store instance so that
documents indexed via ``/ingest`` are immediately searchable via ``/query``.
This module wires those singletons together from configuration. Tests override
:func:`get_services` to inject offline fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from docrag.config import Settings, get_settings
from docrag.embeddings.base import Embeddings
from docrag.factory import build_embeddings, build_llm, build_vectorstore
from docrag.ingestion.chunker import TokenChunker
from docrag.ingestion.service import IngestionService
from docrag.llm.base import LLM
from docrag.rag.pipeline import RagPipeline
from docrag.vectorstore.base import VectorStore


@dataclass
class Services:
    """Container of long-lived, request-shared service objects."""

    settings: Settings
    embeddings: Embeddings
    vectorstore: VectorStore
    llm: LLM
    ingestion: IngestionService
    pipeline: RagPipeline


def build_services(settings: Settings | None = None) -> Services:
    """Construct all services from configuration, sharing one vector store."""
    settings = settings or get_settings()
    settings.ensure_dirs()

    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore(settings)
    llm = build_llm(settings)
    chunker = TokenChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    ingestion = IngestionService(embeddings, vectorstore, chunker)
    pipeline = RagPipeline(embeddings, vectorstore, llm, top_k=settings.top_k)

    return Services(
        settings=settings,
        embeddings=embeddings,
        vectorstore=vectorstore,
        llm=llm,
        ingestion=ingestion,
        pipeline=pipeline,
    )


@lru_cache
def get_services() -> Services:
    """Return the process-wide :class:`Services` singleton (lazily built)."""
    return build_services()
