"""Token-aware chunking with a sliding window.

Chunking happens per page so that page-level citation metadata stays accurate.
Token counts use ``tiktoken`` (the ``cl100k_base`` encoding) so chunk sizes line
up with how modern LLMs actually tokenize text, rather than naive character or
word counts.
"""

from __future__ import annotations

from docrag.models import Chunk, ChunkMetadata

from .loaders import LoadedDocument

_DEFAULT_ENCODING = "cl100k_base"


class TokenChunker:
    """Split documents into overlapping, token-bounded chunks."""

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        encoding_name: str = _DEFAULT_ENCODING,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

        import tiktoken

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        return len(self._encoder.encode(text))

    def _split_text(self, text: str) -> list[str]:
        tokens = self._encoder.encode(text)
        if not tokens:
            return []

        step = self.chunk_size - self.chunk_overlap
        pieces: list[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            piece = self._encoder.decode(window).strip()
            if piece:
                pieces.append(piece)
            if start + self.chunk_size >= len(tokens):
                break
        return pieces

    def chunk_document(self, document: LoadedDocument) -> list[Chunk]:
        """Chunk every page of ``document`` into :class:`Chunk` objects."""
        chunks: list[Chunk] = []
        index = 0
        for page in document.pages:
            for piece in self._split_text(page.text):
                chunks.append(
                    Chunk(
                        id=f"{document.source}::{index}",
                        text=piece,
                        metadata=ChunkMetadata(
                            source=document.source,
                            chunk_id=index,
                            page=page.page,
                        ),
                    )
                )
                index += 1
        return chunks
