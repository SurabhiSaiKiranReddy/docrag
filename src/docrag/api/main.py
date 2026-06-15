"""DocRAG FastAPI service.

Endpoints:
- ``GET  /health``  — liveness + configured providers + index size
- ``POST /ingest``  — upload a PDF/TXT/MD file to index
- ``POST /query``   — ask a grounded question (streaming NDJSON or JSON)

Heavy, blocking work (embedding, generation) is offloaded to a worker thread so
the event loop stays responsive.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from docrag.api.deps import Services, get_services
from docrag.api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from docrag.ingestion.loaders import UnsupportedFileTypeError
from docrag.observability.logging import configure_logging, get_logger
from docrag.observability.metrics import (
    ANSWER_TOKENS,
    INGEST_DURATION,
    INGESTED_CHUNKS,
    INGESTED_DOCUMENTS,
    QUERIES,
    QUERY_DURATION,
    RETRIEVED_CHUNKS,
    setup_metrics,
)

configure_logging()
logger = get_logger("docrag.api")

app = FastAPI(
    title="DocRAG",
    version="0.1.0",
    summary="Document Intelligence Platform (RAG) with citation-grounded answers.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_metrics(app)

ServicesDep = Annotated[Services, Depends(get_services)]


@app.get("/health", response_model=HealthResponse)
def health(services: ServicesDep) -> HealthResponse:
    """Report liveness, the configured providers, and the index size."""
    settings = services.settings
    return HealthResponse(
        status="ok",
        embeddings_provider=str(settings.embeddings_provider),
        llm_provider=str(settings.llm_provider),
        vectorstore_provider=str(settings.vectorstore_provider),
        indexed_chunks=services.vectorstore.count(),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    services: ServicesDep,
) -> IngestResponse:
    """Ingest one uploaded document into the vector store."""
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename or "upload"
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(services.ingestion.ingest_bytes, filename, data)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    INGEST_DURATION.observe(time.perf_counter() - started)
    INGESTED_DOCUMENTS.inc()
    INGESTED_CHUNKS.inc(result.chunks)
    logger.info(
        "document ingested",
        extra={"event": "ingest", "source": result.source, "chunks": result.chunks},
    )
    return IngestResponse(source=result.source, chunks=result.chunks, pages=result.pages)


def _ndjson_stream(services: Services, question: str, top_k: int | None) -> Iterator[str]:
    """Yield NDJSON events: one ``sources`` event, then ``token`` events, then ``done``."""
    started = time.perf_counter()
    result = services.pipeline.run(question, top_k)
    RETRIEVED_CHUNKS.observe(len(result.citations))
    yield json.dumps(
        {"type": "sources", "citations": [c.model_dump() for c in result.citations]}
    ) + "\n"
    try:
        for token in result.tokens:
            ANSWER_TOKENS.inc()
            yield json.dumps({"type": "token", "text": token}) + "\n"
    except Exception as exc:  # surface provider errors to the client mid-stream
        QUERIES.labels(mode="stream", status="error").inc()
        QUERY_DURATION.labels(mode="stream").observe(time.perf_counter() - started)
        logger.exception("generation failed during streaming")
        yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        return
    QUERIES.labels(mode="stream", status="ok").inc()
    QUERY_DURATION.labels(mode="stream").observe(time.perf_counter() - started)
    yield json.dumps({"type": "done"}) + "\n"


@app.post("/query")
async def query(
    request: QueryRequest,
    services: ServicesDep,
):
    """Answer a question over the indexed documents (streaming or JSON)."""
    logger.info(
        "query received",
        extra={"event": "query", "stream": request.stream, "top_k": request.top_k},
    )
    if request.stream:
        return StreamingResponse(
            _ndjson_stream(services, request.question, request.top_k),
            media_type="application/x-ndjson",
        )

    started = time.perf_counter()
    answer = await run_in_threadpool(
        services.pipeline.answer, request.question, request.top_k
    )
    RETRIEVED_CHUNKS.observe(len(answer.citations))
    QUERIES.labels(mode="json", status="ok").inc()
    QUERY_DURATION.labels(mode="json").observe(time.perf_counter() - started)
    return QueryResponse(answer=answer.answer, citations=answer.citations)
