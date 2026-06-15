"""Abstract LLM provider interface.

Both methods take an already-assembled prompt (the RAG layer is responsible for
retrieval and grounding). Implementations are selected via
:func:`docrag.factory.build_llm`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLM(ABC):
    """A text-generation backend supporting streaming and blocking calls."""

    @abstractmethod
    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield answer tokens/fragments as they are produced."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the full answer as a single string."""
