"""Tests for the token-aware chunker and document loaders."""

from __future__ import annotations

import pytest

from docrag.ingestion.chunker import TokenChunker
from docrag.ingestion.loaders import (
    LoadedDocument,
    Page,
    UnsupportedFileTypeError,
    clean_text,
    load_bytes,
)


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TokenChunker(chunk_size=0, chunk_overlap=0)


def test_chunks_respect_max_token_size(sample_document: LoadedDocument) -> None:
    chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(sample_document)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunker.count_tokens(chunk.text) <= 50


def test_chunk_ids_are_unique_and_sequential(sample_document: LoadedDocument) -> None:
    chunker = TokenChunker(chunk_size=40, chunk_overlap=8)
    chunks = chunker.chunk_document(sample_document)

    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))
    assert [c.metadata.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunks_preserve_page_numbers(sample_document: LoadedDocument) -> None:
    chunker = TokenChunker(chunk_size=40, chunk_overlap=8)
    chunks = chunker.chunk_document(sample_document)

    pages = {chunk.metadata.page for chunk in chunks}
    assert pages == {1, 2}


def test_overlap_shares_tokens_between_consecutive_chunks() -> None:
    chunker = TokenChunker(chunk_size=30, chunk_overlap=10)
    doc = LoadedDocument(source="d.txt", pages=[Page(text="word " * 200, page=None)])
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    # Consecutive windows should overlap, so the union of tokens is less than the
    # naive sum of per-chunk token counts.
    total = sum(chunker.count_tokens(c.text) for c in chunks)
    whole = chunker.count_tokens(doc.text)
    assert total > whole


def test_empty_text_produces_no_chunks() -> None:
    chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
    doc = LoadedDocument(source="empty.txt", pages=[Page(text="   ", page=None)])
    assert chunker.chunk_document(doc) == []


def test_clean_text_collapses_blank_lines_and_trailing_ws() -> None:
    raw = "line one   \n\n\n\nline two\r\n"
    assert clean_text(raw) == "line one\n\nline two"


def test_load_bytes_plain_text() -> None:
    doc = load_bytes("notes.txt", b"hello world")
    assert doc.source == "notes.txt"
    assert len(doc.pages) == 1
    assert doc.pages[0].page is None
    assert doc.pages[0].text == "hello world"


def test_load_bytes_markdown() -> None:
    doc = load_bytes("readme.md", b"# Title\n\nBody text")
    assert doc.source == "readme.md"
    assert "Title" in doc.text


def test_load_bytes_rejects_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        load_bytes("data.csv", b"a,b,c")
