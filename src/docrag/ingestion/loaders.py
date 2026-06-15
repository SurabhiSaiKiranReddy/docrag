"""Document loading and text extraction for PDF / TXT / Markdown.

Loaders return a :class:`LoadedDocument` made of one or more :class:`Page`
objects. PDFs preserve 1-based page numbers so citations can point at a page;
plain-text formats use a single page with ``page=None``.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

from pydantic import BaseModel

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}

# Collapse 3+ blank lines and trailing horizontal whitespace.
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+(\n)")


class Page(BaseModel):
    """A single extracted page of text."""

    text: str
    page: int | None = None  # 1-based; None for non-paginated formats


class LoadedDocument(BaseModel):
    """Extracted text for one source file."""

    source: str
    pages: list[Page]

    @property
    def text(self) -> str:
        """All page text joined, for convenience."""
        return "\n\n".join(page.text for page in self.pages)


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is not one of the supported types."""


def clean_text(text: str) -> str:
    """Normalize whitespace without disturbing meaningful structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub(r"\1", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _load_pdf(data: bytes) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        extracted = clean_text(page.extract_text() or "")
        if extracted:
            pages.append(Page(text=extracted, page=number))
    return pages


def _load_plain(data: bytes) -> list[Page]:
    text = clean_text(data.decode("utf-8", errors="replace"))
    return [Page(text=text, page=None)] if text else []


def load_bytes(filename: str, data: bytes) -> LoadedDocument:
    """Load a document from raw bytes (used for API uploads)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type {suffix!r}. Supported: {sorted(_SUPPORTED_SUFFIXES)}"
        )
    pages = _load_pdf(data) if suffix == ".pdf" else _load_plain(data)
    return LoadedDocument(source=Path(filename).name, pages=pages)


def load_document(path: Path | str) -> LoadedDocument:
    """Load a document from a filesystem path."""
    path = Path(path)
    return load_bytes(path.name, path.read_bytes())
