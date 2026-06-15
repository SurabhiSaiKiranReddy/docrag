"""DocRAG Streamlit UI — upload documents and chat over them with citations.

This is a thin client over the FastAPI service. Set ``DOCRAG_API_URL`` to point
at a non-default backend (defaults to ``http://localhost:8000``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("DOCRAG_API_URL", "http://localhost:8000").rstrip("/")
SUPPORTED_TYPES = ["pdf", "txt", "md", "markdown"]

st.set_page_config(page_title="DocRAG", page_icon="📄", layout="centered")


def fetch_health() -> dict[str, Any] | None:
    """Return backend health, or ``None`` if the service is unreachable."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def ingest_file(name: str, data: bytes, mime: str) -> dict[str, Any]:
    """Upload a single document to the backend for indexing."""
    response = httpx.post(
        f"{API_URL}/ingest",
        files={"file": (name, data, mime or "application/octet-stream")},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def reset_index() -> dict[str, Any]:
    """Delete all indexed documents from the backend vector store."""
    response = httpx.delete(f"{API_URL}/index", timeout=30.0)
    response.raise_for_status()
    return response.json()


def stream_answer(question: str, top_k: int, sink: dict[str, Any]) -> Iterator[str]:
    """Stream answer tokens from the backend, stashing citations/errors in ``sink``."""
    payload = {"question": question, "top_k": top_k, "stream": True}
    with httpx.stream("POST", f"{API_URL}/query", json=payload, timeout=None) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("type")
            if kind == "sources":
                sink["citations"] = event.get("citations", [])
            elif kind == "token":
                yield event.get("text", "")
            elif kind == "error":
                sink["error"] = event.get("message", "unknown error")
                yield f"\n\n⚠️ Generation failed: {sink['error']}"
            elif kind == "done":
                break


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render a compact, expandable list of source citations."""
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} source(s)"):
        for citation in citations:
            page = citation.get("page")
            page_str = f", p.{page}" if page is not None else ""
            st.caption(
                f"`{citation['source']}` · chunk #{citation['chunk_id']}{page_str} "
                f"· score {citation['score']:.3f}"
            )


# ── Sidebar: ingestion + backend status ───────────────────────────────
with st.sidebar:
    st.title("📄 DocRAG")
    st.caption("Retrieval-Augmented Generation with citations")

    health = fetch_health()
    if health is None:
        st.error(f"Backend unreachable at {API_URL}")
        st.code("make run", language="bash")
    else:
        st.success("Backend online")
        col1, col2 = st.columns(2)
        col1.metric("Indexed chunks", health["indexed_chunks"])
        col2.metric("LLM", health["llm_provider"])
        st.caption(
            f"embeddings: `{health['embeddings_provider']}` · "
            f"store: `{health['vectorstore_provider']}`"
        )

    st.divider()
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploads, use_container_width=True):
        for upload in uploads or []:
            try:
                result = ingest_file(upload.name, upload.getvalue(), upload.type)
                st.success(f"{result['source']}: {result['chunks']} chunks")
            except httpx.HTTPError as exc:
                st.error(f"{upload.name}: {exc}")

    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=15, value=5)
    col_a, col_b = st.columns(2)
    if col_a.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col_b.button("Reset index", use_container_width=True):
        try:
            result = reset_index()
            st.session_state.messages = []
            st.success(f"Index cleared ({result['indexed_chunks']} chunks)")
            st.rerun()
        except httpx.HTTPError as exc:
            st.error(f"Reset failed: {exc}")


# ── Main: chat ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(message.get("citations", []))

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sink: dict[str, Any] = {}
        try:
            answer = st.write_stream(stream_answer(prompt, top_k, sink))
        except httpx.HTTPError as exc:
            answer = f"⚠️ Could not reach the backend: {exc}"
            st.error(answer)
        citations = sink.get("citations", [])
        render_citations(citations)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "citations": citations}
    )
