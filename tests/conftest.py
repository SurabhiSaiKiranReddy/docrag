"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from docrag.ingestion.loaders import LoadedDocument, Page


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
