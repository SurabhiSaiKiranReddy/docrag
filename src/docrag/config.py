"""Application configuration via environment variables / ``.env``.

All settings are namespaced with the ``DOCRAG_`` prefix (except the standard
``OPENAI_API_KEY``). Providers are swappable at runtime by changing env vars —
no code changes required.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingsProvider(StrEnum):
    """Supported embedding backends."""

    LOCAL = "local"
    OPENAI = "openai"


class LLMProvider(StrEnum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    OPENAI = "openai"


class VectorStoreProvider(StrEnum):
    """Supported vector store backends."""

    FAISS = "faiss"


class Settings(BaseSettings):
    """Runtime configuration loaded from the environment / ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="DOCRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Provider selection ────────────────────────────────────────────
    embeddings_provider: EmbeddingsProvider = EmbeddingsProvider.LOCAL
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    vectorstore_provider: VectorStoreProvider = VectorStoreProvider.FAISS

    # ── Local embeddings ──────────────────────────────────────────────
    local_embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── OpenAI ────────────────────────────────────────────────────────
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "DOCRAG_OPENAI_API_KEY"),
    )
    openai_embeddings_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    # ── Ollama ────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ── Chunking ──────────────────────────────────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 80

    # ── Retrieval ─────────────────────────────────────────────────────
    top_k: int = 5

    # ── Hybrid search + reranking (Phase 7) ──────────────────────────
    hybrid_search: bool = False  # fuse BM25 keyword ranking with vector search
    rerank: bool = False  # cross-encoder rerank of fused candidates
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_candidates: int = 20  # candidate pool size before fuse/rerank
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # ── Generation ────────────────────────────────────────────────────
    temperature: float = 0.1
    max_tokens: int = 1024

    # ── Storage ───────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")

    @property
    def uploads_dir(self) -> Path:
        """Directory where uploaded source files are persisted."""
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        """Create storage directories if they do not yet exist."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
