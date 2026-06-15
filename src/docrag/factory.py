"""Provider factories — the composition root.

This module is the single place that maps configuration to concrete provider
implementations. Everything else depends only on the abstract interfaces
(:class:`Embeddings`, :class:`VectorStore`, :class:`LLM`), which keeps the system
provider-agnostic and easy to test.
"""

from __future__ import annotations

from docrag.config import (
    EmbeddingsProvider,
    LLMProvider,
    Settings,
    VectorStoreProvider,
    get_settings,
)
from docrag.embeddings.base import Embeddings
from docrag.llm.base import LLM
from docrag.rag.retriever import HybridRetriever, Retriever
from docrag.vectorstore.base import VectorStore


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Construct the configured embeddings provider."""
    settings = settings or get_settings()
    if settings.embeddings_provider is EmbeddingsProvider.LOCAL:
        from docrag.embeddings.local import LocalEmbeddings

        return LocalEmbeddings(model_name=settings.local_embeddings_model)
    if settings.embeddings_provider is EmbeddingsProvider.OPENAI:
        from docrag.embeddings.openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embeddings_model,
            api_key=settings.openai_api_key,
        )
    raise ValueError(f"Unsupported embeddings provider: {settings.embeddings_provider}")


def build_vectorstore(settings: Settings | None = None) -> VectorStore:
    """Construct the configured vector store."""
    settings = settings or get_settings()
    if settings.vectorstore_provider is VectorStoreProvider.FAISS:
        from docrag.vectorstore.faiss_store import FaissVectorStore

        return FaissVectorStore(index_dir=settings.index_dir)
    raise ValueError(f"Unsupported vector store provider: {settings.vectorstore_provider}")


def build_llm(settings: Settings | None = None) -> LLM:
    """Construct the configured LLM provider."""
    settings = settings or get_settings()
    if settings.llm_provider is LLMProvider.OLLAMA:
        from docrag.llm.ollama import OllamaLLM

        return OllamaLLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    if settings.llm_provider is LLMProvider.OPENAI:
        from docrag.llm.openai import OpenAILLM

        return OpenAILLM(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def build_retriever(
    embeddings: Embeddings,
    vectorstore: VectorStore,
    settings: Settings | None = None,
) -> Retriever:
    """Construct the configured retriever (dense, hybrid, and/or reranked)."""
    settings = settings or get_settings()
    if not settings.hybrid_search and not settings.rerank:
        return Retriever(embeddings, vectorstore)

    reranker = None
    if settings.rerank:
        from docrag.rag.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker(settings.rerank_model)

    return HybridRetriever(
        embeddings,
        vectorstore,
        use_bm25=settings.hybrid_search,
        reranker=reranker,
        candidates=settings.hybrid_candidates,
        rrf_k=settings.rrf_k,
    )
