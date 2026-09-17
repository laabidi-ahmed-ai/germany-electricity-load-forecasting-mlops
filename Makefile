# ---------------------------------------------------------------------------
# Standard commands (CLAUDE.md "Standard commands")
#
#   make setup      install deps into a venv
#   make test       pytest (unit tests; `make test-integration` hits the real APIs)
#   make lint       ruff (lint + format check)
#   make data       run the ingestion / backfill
#   make train      train + evaluate, log to MLflow
#   make serve      run the FastAPI app locally
#   make dashboard  run the Streamlit dashboard
#
# Works on Linux / macOS / Git Bash on Windows. The venv lives in ./.venv.
# ---------------------------------------------------------------------------

PYTHON      ?= python
VENV        := .venv
ifeq ($(OS),Windows_NT)
    BIN := $(VENV)/Scripts
else
    BIN := $(VENV)/bin
endif
PIP         := $(BIN)/pip
PY          := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup test test-integration test-all lint format data data-update features train evaluate mlflow-ui register forecast monitor retrain serve dashboard clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)  ## Create venv and install project + dev deps
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

$(VENV):
	$(PYTHON) -m venv $(VENV)

test:  ## Run the unit test suite (no network)
	$(PY) -m pytest -m "not integration"

test-integration:  ## Run the real-network integration tests (SMARD / Open-Meteo)
	$(PY) -m pytest -m integration

test-all:  ## Run every test
	$(PY) -m pytest

lint:  ## Lint + check formatting with ruff
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Auto-fix lint issues and reformat
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

data:  ## Historical backfill from DATA_START_DATE (SMARD + Open-Meteo, + ENTSO-E if token)
	$(PY) -m src.data.ingest --backfill

data-update:  ## Incremental ingestion: pull only new hours
	$(PY) -m src.data.ingest

features:  ## Build the leakage-safe feature frame -> data/processed/features.parquet (Phase 2)
	$(PY) -m src.features.build_features

train:  ## Expanding-window CV (baselines vs LightGBM) + final model, logged to MLflow (Phase 3)
	$(PY) -m src.models.train

evaluate:  ## CV report only (baselines vs LightGBM), logged to MLflow
	$(PY) -m src.models.evaluate

mlflow-ui:  ## Open the MLflow UI on the local store
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

register:  ## Register the latest training run in the MLflow registry and promote it to champion
	$(PY) -m src.models.registry --promote-latest

forecast:  ## Batch day-ahead forecast with the champion -> load_forecast_model table (Phase 4)
	$(PY) -m src.serving.batch_forecast

monitor:  ## Performance vs official forecast + drift + retrain-trigger check (no retraining)
	$(PY) -m src.monitoring.retrain --check-only

retrain:  ## Evaluate triggers; if fired, train a challenger and promote it only if it wins (Phase 5)
	$(PY) -m src.monitoring.retrain

serve:  ## Run the FastAPI app locally (Phase 4)
	$(BIN)/uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Run the Streamlit dashboard (Phase 7)
	$(BIN)/streamlit run dashboard/app.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
