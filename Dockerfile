# ---------------------------------------------------------------------------
# Serving image: FastAPI day-ahead forecast API (also runs the batch job).
#
#   docker build -t germany-load-api .
#   docker run --rm -p 8000:8000 --env-file .env \
#       -v "$PWD/mlruns:/app/mlruns" -v "$PWD/data:/app/data" germany-load-api
#
# The MLflow store (model registry) and the local SQLite DB live in mounted
# volumes; in the cloud they are replaced by DATABASE_URL / MLFLOW_TRACKING_URI.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

# libgomp: OpenMP runtime required by LightGBM wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config ./config
COPY dashboard ./dashboard
RUN pip install --upgrade pip && pip install .

# Non-root runtime user; data/ and mlruns/ are mounted volumes.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/mlruns \
    && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
