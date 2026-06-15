"""Local, zero-cost embeddings via ``sentence-transformers`` (runs on CPU)."""

from __future__ import annotations

from .base import Embeddings


class LocalEmbeddings(Embeddings):
    """CPU-friendly embeddings using a ``sentence-transformers`` model.

    The default ``all-MiniLM-L6-v2`` produces 384-dim vectors and is fast enough
    for interactive use on a laptop with no GPU.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        # Imported lazily so merely importing the package stays cheap and does
        # not pull torch into processes that only need a different provider.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # The accessor was renamed across sentence-transformers versions; support both.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self._dimension = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
