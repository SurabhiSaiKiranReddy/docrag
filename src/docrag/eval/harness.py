"""Run retrieval-quality evaluation over the gold dataset.

The retrieval metrics here need no LLM — they measure whether the right source
documents are retrieved — so they produce real, reproducible numbers on a CPU.
Generation-quality metrics (RAGAS faithfulness / answer relevancy) require a
configured LLM and live in :func:`run_ragas`.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from docrag.config import Settings, get_settings
from docrag.eval.dataset import QAItem
from docrag.eval.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from docrag.factory import build_embeddings, build_retriever
from docrag.ingestion.chunker import TokenChunker
from docrag.ingestion.service import IngestionService
from docrag.rag.retriever import Retriever
from docrag.vectorstore.faiss_store import FaissVectorStore

_SAMPLE_GLOBS = ("*.md", "*.txt", "*.pdf")


@dataclass
class RetrievalReport:
    """Aggregated retrieval metrics plus per-question detail."""

    num_questions: int
    indexed_chunks: int
    ks: tuple[int, ...]
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr: float
    rows: list[dict[str, object]] = field(default_factory=list)

    def as_markdown(self) -> str:
        """Render the aggregate metrics as a Markdown table."""
        lines = [
            f"Indexed {self.indexed_chunks} chunks · {self.num_questions} questions\n",
            "| k | Hit@k | Recall@k | Precision@k |",
            "|---|------:|---------:|------------:|",
        ]
        for k in self.ks:
            lines.append(
                f"| {k} | {self.hit_at_k[k]:.2f} | "
                f"{self.recall_at_k[k]:.2f} | {self.precision_at_k[k]:.2f} |"
            )
        lines.append(f"\n**MRR:** {self.mrr:.3f}")
        return "\n".join(lines)


def build_sample_retriever(
    sample_dir: Path | str,
    settings: Settings | None = None,
    chunk_size: int = 128,
    chunk_overlap: int = 24,
) -> tuple[Retriever, int]:
    """Ingest the sample corpus into an ephemeral index and return a retriever."""
    settings = settings or get_settings()
    embeddings = build_embeddings(settings)
    index_dir = tempfile.mkdtemp(prefix="docrag-eval-")
    store = FaissVectorStore(index_dir=index_dir)
    chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    service = IngestionService(embeddings, store, chunker)

    sample_dir = Path(sample_dir)
    for pattern in _SAMPLE_GLOBS:
        for path in sorted(sample_dir.glob(pattern)):
            service.ingest_path(path)

    retriever = build_retriever(embeddings, store, settings)
    return retriever, store.count()


def evaluate_retrieval(
    retriever: Retriever,
    dataset: Iterable[QAItem],
    indexed_chunks: int,
    ks: tuple[int, ...] = (1, 3, 5),
) -> RetrievalReport:
    """Compute aggregate retrieval metrics over ``dataset``."""
    items = list(dataset)
    max_k = max(ks)
    hits: dict[int, list[float]] = {k: [] for k in ks}
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    precisions: dict[int, list[float]] = {k: [] for k in ks}
    rrs: list[float] = []
    rows: list[dict[str, object]] = []

    for item in items:
        scored = retriever.retrieve(item.question, max_k)
        retrieved = [chunk.chunk.metadata.source for chunk in scored]
        relevant = set(item.sources)
        for k in ks:
            hits[k].append(hit_at_k(retrieved, relevant, k))
            recalls[k].append(recall_at_k(retrieved, relevant, k))
            precisions[k].append(precision_at_k(retrieved, relevant, k))
        rr = reciprocal_rank(retrieved, relevant)
        rrs.append(rr)
        rows.append(
            {
                "question": item.question,
                "relevant": sorted(relevant),
                "retrieved": retrieved,
                "rr": round(rr, 3),
            }
        )

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return RetrievalReport(
        num_questions=len(items),
        indexed_chunks=indexed_chunks,
        ks=ks,
        hit_at_k={k: _avg(hits[k]) for k in ks},
        recall_at_k={k: _avg(recalls[k]) for k in ks},
        precision_at_k={k: _avg(precisions[k]) for k in ks},
        mrr=_avg(rrs),
        rows=rows,
    )


def run_ragas(dataset: Iterable[QAItem], sample_dir: Path | str) -> object:
    """Compute RAGAS generation metrics (requires the ``eval`` extra + an LLM).

    This is intentionally import-guarded: ``ragas`` is a heavy optional
    dependency and the metrics need a working LLM. Raises a clear error when the
    prerequisites are missing.
    """
    try:
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas.metrics import (  # type: ignore[import-not-found]
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - exercised only with extra installed
        raise RuntimeError(
            "RAGAS is not installed. Run `pip install -e \".[eval]\"` and configure "
            "an LLM (OPENAI_API_KEY or a running Ollama) to compute generation metrics."
        ) from exc

    # Building the evaluation dataset requires generating answers with the
    # configured pipeline; wired here for completeness and run on demand.
    from datasets import Dataset  # type: ignore[import-not-found]

    from docrag.api.deps import build_services

    services = build_services()
    records: list[dict[str, object]] = []
    for item in dataset:
        result = services.pipeline.answer(item.question)
        contexts = [c.chunk.text for c in services.pipeline.retriever.retrieve(item.question, 5)]
        records.append(
            {
                "question": item.question,
                "answer": result.answer,
                "contexts": contexts,
                "ground_truth": item.answer,
            }
        )

    return evaluate(
        Dataset.from_list(records),
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
