"""MLflow helpers: resolve the tracking URI from settings and select an experiment.

``MLFLOW_TRACKING_URI`` may be:

* ``sqlite:///mlruns/mlflow.db`` (default) - a local SQLite backend, the store
  MLflow recommends. A *relative* path is anchored at the project root so every
  entry point (CLI, tests, GitHub Actions) writes to the same file.
* a plain directory such as ``./mlruns`` - the legacy filesystem store. MLflow 3
  keeps it in maintenance mode behind ``MLFLOW_ALLOW_FILE_STORE=true``; we set that
  flag so the README §12 example keeps working.
* ``postgresql://…`` - a Postgres backend store (the cloud setup: the same
  Neon/Supabase database that holds the data tables; MLflow adds its own tables).
  Re-routed to the psycopg 3 driver like ``DATABASE_URL``.
* any other full URI (``http://…``) - used as-is.

Run artifacts (models, reports) go to ``<project>/mlruns/artifacts`` unless the
experiment already exists with another location.
"""

from __future__ import annotations

import os
from pathlib import Path

from config.settings import PROJECT_ROOT, get_settings

DEFAULT_EXPERIMENT = "germany-load-forecasting"
ARTIFACT_ROOT = PROJECT_ROOT / "mlruns" / "artifacts"

# Silence MLflow's interactive "agent hint" banner in non-interactive runs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

_SQLITE_PREFIX = "sqlite:///"


def resolve_tracking_uri(uri: str | None = None) -> str:
    """Turn ``uri`` (default: ``settings.mlflow_tracking_uri``) into an absolute MLflow URI."""
    raw = uri if uri is not None else get_settings().mlflow_tracking_uri

    if raw.startswith(_SQLITE_PREFIX):
        db_path = Path(raw.removeprefix(_SQLITE_PREFIX))
        if db_path.as_posix() == ":memory:":
            return raw
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        db_path = db_path.resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"{_SQLITE_PREFIX}{db_path.as_posix()}"

    if "://" in raw:
        from src.data.db import normalize_database_url

        return normalize_database_url(raw)

    # Plain directory -> legacy file store (opt in, see module doc).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve().as_uri()


def setup_mlflow(experiment: str = DEFAULT_EXPERIMENT, uri: str | None = None) -> str:
    """Point MLflow at the configured store and experiment; returns the resolved URI."""
    import mlflow

    resolved = resolve_tracking_uri(uri)
    mlflow.set_tracking_uri(resolved)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_root = _artifact_root_for(resolved)
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)
    return resolved


def _artifact_root_for(resolved_uri: str) -> str | None:
    """Keep artifacts next to a local SQLite store; let servers/file stores pick their own."""
    if not resolved_uri.startswith(_SQLITE_PREFIX):
        return None
    db_path = Path(resolved_uri.removeprefix(_SQLITE_PREFIX))
    root = db_path.parent / "artifacts" if db_path.is_absolute() else ARTIFACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root.as_uri()
