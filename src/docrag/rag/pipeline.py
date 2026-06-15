"""RAG orchestration: retrieve -> build grounded prompt -> stream the answer.

Retrieval runs eagerly so citations are known before generation starts; the
answer tokens stream lazily. This lets the API send sources immediately and then
forward tokens as they arrive.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from docrag.embeddings.base import Embeddings
from docrag.llm.base import LLM
from docrag.models import Citation, ScoredChunk
from docrag.rag.prompt import build_prompt
from docrag.rag.retriever import Retriever
from docrag.vectorstore.base import VectorStore


@dataclass
class RagStream:
    """A retrieval result plus a lazy token stream for the answer."""

    citations: list[Citation]
    chunks: list[ScoredChunk]
    tokens: Iterator[str]


@dataclass
class RagAnswer:
    """A fully materialized answer with its citations."""

    answer: str
    citations: list[Citation]


class RagPipeline:
    """End-to-end RAG pipeline over the configured providers."""

    def __init__(
        self,
        embeddings: Embeddings,
        vectorstore: VectorStore,
        llm: LLM,
        top_k: int = 5,
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever or Retriever(embeddings, vectorstore)
        self.llm = llm
        self.top_k = top_k

    def run(self, question: str, top_k: int | None = None) -> RagStream:
        """Retrieve context and return a streaming answer handle."""
        scored = self.retriever.retrieve(question, top_k or self.top_k)
        citations = [Citation.from_scored(chunk) for chunk in scored]
        system, user = build_prompt(question, scored)
        tokens = self.llm.stream(user, system=system)
        return RagStream(citations=citations, chunks=scored, tokens=tokens)

    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        """Retrieve context and return the fully materialized answer."""
        stream = self.run(question, top_k)
        text = "".join(stream.tokens)
        return RagAnswer(answer=text, citations=stream.citations)
