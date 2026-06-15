"""Shared pytest fixtures."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

import pytest

from docrag.embeddings.base import Embeddings
from docrag.ingestion.loaders import LoadedDocument, Page
from docrag.llm.base import LLM

_VOCAB = [
    "events",
    "retention",
    "days",
    "collector",
    "flush",
    "seconds",
    "uptime",
    "enterprise",
    "pricing",
    "region",
]
_TOKEN = re.compile(r"[a-z]+")
_TOKEN_WITH_WS = re.compile(r"\S+\s*")


class FakeEmbeddings(Embeddings):
    """Deterministic, offline bag-of-words embeddings over a tiny vocabulary.

    Good enough to make semantically overlapping texts score higher than
    unrelated ones, without downloading any model.
    """

    def _vector(self, text: str) -> list[float]:
        words = _TOKEN.findall(text.lower())
        counts = [float(words.count(term)) for term in _VOCAB]
        norm = math.sqrt(sum(value * value for value in counts)) or 1.0
        return [value / norm for value in counts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return len(_VOCAB)


class FakeLLM(LLM):
    """LLM stub that records the last prompt and streams a canned answer."""

    def __init__(self, answer: str = "Events are retained for 90 days.") -> None:
        self.answer = answer
        self.last_system: str | None = None
        self.last_prompt: str | None = None

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        self.last_prompt = prompt
        self.last_system = system
        # Yield whitespace-preserving fragments so ''.join reconstructs the
        # answer, mirroring how real LLM tokens carry their own spacing.
        yield from _TOKEN_WITH_WS.findall(self.answer)

    def complete(self, prompt: str, system: str | None = None) -> str:
        return "".join(self.stream(prompt, system))


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_document() -> LoadedDocument:
    """A small two-page document for chunking/retrieval tests."""
    return LoadedDocument(
        source="sample.md",
        pages=[
            Page(text="Nimbus retains raw events for 90 days. " * 20, page=1),
            Page(text="The Collector flushes every 5 seconds or 500 events. " * 20, page=2),
        ],
    )
