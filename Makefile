.DEFAULT_GOAL := help
PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install install-all run ui test lint fmt eval clean docker-build docker-up docker-down

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtual environment
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: ## Install CPU torch + project (editable) with dev tools
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev]"

install-all: ## Install everything (dev + eval + observability)
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev,eval,observability]"

run: ## Run the FastAPI service at http://localhost:8000
	$(PYTHON) -m uvicorn docrag.api.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Run the Streamlit UI at http://localhost:8501
	$(PYTHON) -m streamlit run ui/app.py

test: ## Run the test suite
	$(PYTHON) -m pytest -q

lint: ## Lint with ruff and type-check with mypy
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

fmt: ## Auto-format with black and ruff --fix
	$(PYTHON) -m black src tests
	$(PYTHON) -m ruff check --fix src tests

eval: ## Run the RAG evaluation harness
	$(PYTHON) scripts/eval.py

docker-build: ## Build the DocRAG Docker image
	docker build -t docrag:latest .

docker-up: ## Start the full stack (api + ui + ollama + prometheus + grafana)
	docker compose up --build -d

docker-down: ## Stop the full stack
	docker compose down

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
