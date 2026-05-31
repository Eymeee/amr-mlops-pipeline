.PHONY: help sync ingest preprocess dvc-repro dvc-status train train-smoke \
        lint test test-ci check api monitor mlflow docker-build stack-up \
        stack-down stack-ps health

PYTHON := uv run python
DOCKER_COMPOSE ?= sudo docker compose

help:
	@echo "Available targets:"
	@echo "  sync          Install dependencies from uv.lock"
	@echo "  ingest        Validate raw dataset"
	@echo "  preprocess    Generate processed train/val/test data"
	@echo "  dvc-repro     Reproduce DVC preprocessing pipeline"
	@echo "  dvc-status    Show DVC status"
	@echo "  train         Run full training benchmark"
	@echo "  train-smoke   Run 1-trial training smoke test"
	@echo "  lint          Run Ruff"
	@echo "  test          Run all tests"
	@echo "  test-ci       Run CI-safe tests"
	@echo "  check         Run lint + CI-safe tests"
	@echo "  api           Start FastAPI locally"
	@echo "  monitor       Start monitoring script locally"
	@echo "  mlflow        Start MLflow UI"
	@echo "  docker-build  Build Docker image"
	@echo "  stack-up      Start Docker Compose stack"
	@echo "  stack-down    Stop Docker Compose stack"
	@echo "  stack-ps      Show Docker Compose services"
	@echo "  health        Check API health"

sync:
	uv sync --frozen

ingest:
	$(PYTHON) src/ingestion/ingest.py

preprocess:
	$(PYTHON) src/preprocessing/preprocess.py

dvc-repro:
	uv run dvc repro

dvc-status:
	uv run dvc status

train:
	$(PYTHON) src/training/train.py

train-smoke:
	$(PYTHON) src/training/train.py --n-trials 1

lint:
	uv run ruff check src/ tests/

test:
	uv run pytest tests/ -v

test-ci:
	uv run pytest tests/ --ignore=tests/test_api.py -v

check: lint test-ci

api:
	uv run uvicorn src.serving.api:app --reload

monitor:
	$(PYTHON) src/monitoring/monitor.py

mlflow:
	uv run mlflow ui

docker-build:
	docker build -t amr-outcome-api .

stack-up:
	$(DOCKER_COMPOSE) up --build

stack-down:
	$(DOCKER_COMPOSE) down

stack-ps:
	$(DOCKER_COMPOSE) ps

health:
	curl http://127.0.0.1:8000/health
