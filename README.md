# DocRAG — Document Intelligence Platform (RAG)

A portfolio-grade, production-style **Retrieval-Augmented Generation** service.
Upload documents (PDF / TXT / Markdown) and DocRAG **ingests → chunks → embeds →
indexes** them, then answers natural-language questions with an LLM — every
answer **grounded with citations** back to the source chunks.

Designed to run **for free on a CPU** (no GPU required): local embeddings
(`sentence-transformers`) + FAISS + Ollama by default, with one-env-var swaps to
OpenAI for speed.

> 🚧 **Status:** under active construction. See
> [DocRAG-Project-Spec.md](DocRAG-Project-Spec.md) for the full spec and build
> order.

## Quickstart

```bash
make venv       # create the virtual environment (.venv)
make install    # CPU-only torch + project (editable) with dev tools
cp .env.example .env
make run         # FastAPI at http://localhost:8000  (docs at /docs)
make ui          # Streamlit UI at http://localhost:8501  (second terminal)
```

The default configuration needs **no API keys**. To use OpenAI instead, set
`DOCRAG_LLM_PROVIDER=openai` / `DOCRAG_EMBEDDINGS_PROVIDER=openai` and export
`OPENAI_API_KEY` in your `.env`.

## Architecture

See [DocRAG-Project-Spec.md](DocRAG-Project-Spec.md) §5. A rendered diagram will
live at `docs/architecture.png` in a later phase.

## Design Principles

- **Provider-agnostic interfaces** — `Embeddings`, `VectorStore`, and `LLM` are
  abstract base classes; concrete backends are chosen via env vars.
- **Runs free out-of-the-box** — FAISS + sentence-transformers + Ollama require
  no API keys, so any reviewer can run it on a laptop.
- **Measurable quality** — a RAG evaluation harness reports real retrieval and
  answer metrics (added in a later phase).
- **Secure by default** — secrets only via env; `.env` is gitignored; built and
  tested exclusively on public/synthetic data.

## Project Layout

```
src/docrag/        # application package (api, ingestion, embeddings,
                   # vectorstore, rag, llm, observability)
ui/                # Streamlit app
scripts/           # load test + evaluation harness
tests/             # pytest suite
data/sample/       # synthetic documents for local testing
```

## License

MIT — see [LICENSE](LICENSE).
