"""Prometheus metrics for DocRAG.

Exposes domain metrics (ingestion/query latency, retrieved chunks, answer
tokens) alongside the default HTTP metrics from
``prometheus-fastapi-instrumentator`` (which give per-route latency histograms
for p50/p95/p99). If the optional observability dependencies are not installed,
this module degrades to no-op stubs so the app still runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from prometheus_client import Counter, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _PROMETHEUS_AVAILABLE = False

if TYPE_CHECKING:
    from fastapi import FastAPI

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
_CHUNK_BUCKETS = (0, 1, 2, 3, 5, 8, 13, 21)


class _NoOpMetric:
    """Stand-in used when prometheus_client is unavailable."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoOpMetric:
        return self

    def observe(self, _value: float) -> None:
        return None

    def inc(self, _value: float = 1) -> None:
        return None


if _PROMETHEUS_AVAILABLE:
    QUERY_DURATION = Histogram(
        "docrag_query_duration_seconds",
        "End-to-end query duration in seconds.",
        labelnames=("mode",),
        buckets=_LATENCY_BUCKETS,
    )
    INGEST_DURATION = Histogram(
        "docrag_ingest_duration_seconds",
        "Document ingestion duration in seconds.",
        buckets=_LATENCY_BUCKETS,
    )
    RETRIEVED_CHUNKS = Histogram(
        "docrag_retrieved_chunks",
        "Number of chunks retrieved per query.",
        buckets=_CHUNK_BUCKETS,
    )
    ANSWER_TOKENS = Counter(
        "docrag_answer_tokens_total",
        "Total answer tokens/fragments streamed to clients.",
    )
    QUERIES = Counter(
        "docrag_queries_total",
        "Total queries handled.",
        labelnames=("mode", "status"),
    )
    INGESTED_DOCUMENTS = Counter(
        "docrag_ingested_documents_total",
        "Total documents ingested.",
    )
    INGESTED_CHUNKS = Counter(
        "docrag_ingested_chunks_total",
        "Total chunks indexed.",
    )
else:  # pragma: no cover - exercised only without the extra
    QUERY_DURATION = _NoOpMetric()  # type: ignore[assignment]
    INGEST_DURATION = _NoOpMetric()  # type: ignore[assignment]
    RETRIEVED_CHUNKS = _NoOpMetric()  # type: ignore[assignment]
    ANSWER_TOKENS = _NoOpMetric()  # type: ignore[assignment]
    QUERIES = _NoOpMetric()  # type: ignore[assignment]
    INGESTED_DOCUMENTS = _NoOpMetric()  # type: ignore[assignment]
    INGESTED_CHUNKS = _NoOpMetric()  # type: ignore[assignment]


def setup_metrics(app: FastAPI) -> bool:
    """Instrument ``app`` and expose ``/metrics``. Returns False if unavailable."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:  # pragma: no cover - exercised only without the extra
        return False

    Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    return True
