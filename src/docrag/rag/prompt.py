"""Grounded prompt construction using LangChain prompt templates.

The retrieved chunks are formatted with explicit ``[source: file #chunk]``
markers and the model is instructed to cite them inline and to refuse when the
answer is not supported by the context. Using ``langchain_core`` here keeps the
orchestration layer aligned with the LangChain-based pipeline it models, while
the LLM/embedding/vector-store providers remain framework-agnostic.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from docrag.models import ScoredChunk

SYSTEM_PROMPT = (
    "You are DocRAG, a precise assistant that answers strictly from the provided "
    "context.\n"
    "Rules:\n"
    "- Use ONLY the information in the context. Do not rely on outside knowledge.\n"
    "- If the context does not contain the answer, say you don't know.\n"
    "- After each claim, cite its source by copying VERBATIM the exact "
    "[source: ...] marker that appears directly above the supporting text.\n"
    "- Never invent, renumber, or modify a citation marker. Only use markers that "
    "appear in the context exactly as written.\n"
    "- Be concise and factual."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer the question using only the context above, with inline citations:",
        ),
    ]
)


def format_context(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks into a citation-annotated context block."""
    blocks = [f"{scored.chunk.citation}\n{scored.chunk.text}" for scored in chunks]
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[ScoredChunk]) -> tuple[str, str]:
    """Return the ``(system, user)`` prompt strings for the LLM.

    Chunk text is passed as a *value* to the template, so any braces inside the
    documents are inserted literally and never re-interpreted as template
    variables.
    """
    messages = _PROMPT.format_messages(
        context=format_context(chunks) or "(no relevant context found)",
        question=question,
    )
    system_text = str(messages[0].content)
    user_text = str(messages[1].content)
    return system_text, user_text
