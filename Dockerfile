# DocRAG image — serves both the FastAPI API and the Streamlit UI
# (the docker-compose file overrides the command per service).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Install dependencies first for better layer caching. CPU-only torch keeps the
# image free of multi-GB CUDA wheels.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[observability]"

COPY ui ./ui
COPY data ./data

EXPOSE 8000 8501

# Default to the API; compose overrides this for the UI service.
CMD ["uvicorn", "docrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
