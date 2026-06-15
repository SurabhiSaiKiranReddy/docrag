"""FAISS-backed vector store (local, zero-cost, CPU).

Uses a flat inner-product index. Because all providers emit L2-normalized
vectors, inner product equals cosine similarity, so scores live in ``[-1, 1]``.
The index and its aligned chunk metadata are persisted side by side under
``index_dir`` so ingestion survives restarts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from docrag.models import Chunk, ScoredChunk

from .base import VectorStore

if TYPE_CHECKING:
    import faiss

_INDEX_FILE = "faiss.index"
_META_FILE = "chunks.json"


class FaissVectorStore(VectorStore):
    """Flat inner-product FAISS index with a JSON metadata sidecar."""

    def __init__(self, index_dir: Path | str) -> None:
        self.index_dir = Path(index_dir)
        self._index: faiss.Index | None = None  # created lazily once we know the dimension
        self._chunks: list[Chunk] = []
        self.load()

    # ── internal helpers ──────────────────────────────────────────────
    def _index_path(self) -> Path:
        return self.index_dir / _INDEX_FILE

    def _meta_path(self) -> Path:
        return self.index_dir / _META_FILE

    def _ensure_index(self, dimension: int) -> None:
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(dimension)

    # ── VectorStore API ───────────────────────────────────────────────
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        import numpy as np

        matrix = np.asarray(embeddings, dtype="float32")
        self._ensure_index(matrix.shape[1])
        assert self._index is not None
        self._index.add(matrix)
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], k: int) -> list[ScoredChunk]:
        if self._index is None or not self._chunks:
            return []

        import numpy as np

        query = np.asarray([query_embedding], dtype="float32")
        top_k = min(k, len(self._chunks))
        scores, indices = self._index.search(query, top_k)

        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:  # FAISS pads with -1 when fewer than k results exist
                continue
            results.append(ScoredChunk(chunk=self._chunks[int(idx)], score=float(score)))
        return results

    def persist(self) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self._index_path()))
        payload = [chunk.model_dump() for chunk in self._chunks]
        self._meta_path().write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        index_path = self._index_path()
        meta_path = self._meta_path()
        if not index_path.exists() or not meta_path.exists():
            return

        import faiss

        self._index = faiss.read_index(str(index_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk.model_validate(item) for item in raw]

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def clear(self) -> None:
        # Reset in place so references held by retrievers/pipeline stay valid.
        self._index = None
        self._chunks = []
        self._index_path().unlink(missing_ok=True)
        self._meta_path().unlink(missing_ok=True)
