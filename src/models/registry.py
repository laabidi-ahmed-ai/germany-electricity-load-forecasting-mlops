"""Model registry: MLflow for lineage, the database for the champion serving needs.

Two stores, one rule
--------------------
* **MLflow Model Registry** (``germany-load-day-ahead``, alias ``champion``) keeps
  lineage: every version links to its training run (params, CV metrics, feature
  list, report). With ``MLFLOW_TRACKING_URI`` pointing at Postgres this survives
  ephemeral runners - but the logged model *files* only live on the runner that
  trained them.
* **``model_artifacts`` table** (``src.data.db.ModelArtifact``) holds, per exported
  version, the zipped MLflow pyfunc bundle (~3-4 MB) plus the metadata serving
  needs (features in model order, horizon, train window, CV metrics). Exactly one
  row per model is flagged ``is_champion``.

``promote(version)`` does both: move the MLflow alias **and** export + flag the
version in the DB. ``load_champion(engine)`` reads only the DB, so the forecast
job, the API and the monitor work on a fresh GitHub Actions runner (or in
Docker) with nothing but ``DATABASE_URL``. Rollback is still one call:
``promote(<old version>)``.

CLI (``make register``)::

    python -m src.models.registry --promote-latest      # latest training run -> champion
    python -m src.models.registry --run-id <id>         # a specific run -> champion
    python -m src.models.registry --promote <version>   # move the alias (+ DB pointer)
    python -m src.models.registry --show
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from config.settings import PROJECT_ROOT, get_settings
from src.data import db
from src.features.horizons import DAY_AHEAD, HORIZONS, Horizon, select_features
from src.models.tracking import DEFAULT_EXPERIMENT, setup_mlflow

log = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "germany-load-day-ahead"
CHAMPION_ALIAS = "champion"
KEEP_EXPORTED_VERSIONS = 5
MODEL_CACHE_DIR = PROJECT_ROOT / "data" / "models"

_METRIC_KEYS = (
    "lightgbm_mae_mean",
    "lightgbm_rmse_mean",
    "lightgbm_mape_mean",
    "seasonal_naive_168_mae_mean",
    "seasonal_naive_24_mae_mean",
    "beats_all_baselines",
)


@dataclass
class LoadedModel:
    """A model ready to predict on a feature frame (from the DB export or MLflow)."""

    name: str
    version: str
    alias: str | None
    run_id: str | None
    features: list[str]
    horizon: Horizon
    pyfunc: Any
    train_start: pd.Timestamp | None = None
    train_end: pd.Timestamp | None = None
    metrics: dict[str, float] | None = None
    source: str = "db"

    @property
    def version_label(self) -> str:
        return f"{self.name}:v{self.version}"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ValueError(f"feature frame is missing model inputs: {missing}")
        return np.asarray(self.pyfunc.predict(X[self.features]), dtype="float64")

    def check_features(self, columns: list[str]) -> None:
        """Assert the served frame exposes exactly the training features for this horizon."""
        expected = select_features(columns, self.horizon)
        if expected != self.features:
            raise ValueError(
                "train/serve feature mismatch:\n"
                f"  model expects : {self.features}\n"
                f"  serving offers: {expected}"
            )


# --------------------------------------------------------------------------- #
# MLflow side: register / alias / load a version
# --------------------------------------------------------------------------- #
def register_run_model(
    run_id: str, *, name: str = REGISTERED_MODEL_NAME, experiment: str = DEFAULT_EXPERIMENT
) -> Any:
    """Create a new registered-model version from ``runs:/<run_id>/model``."""
    import mlflow

    setup_mlflow(experiment)
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    for key in ("horizon", "model"):
        if key in run.data.tags:
            client.set_model_version_tag(name, mv.version, key, run.data.tags[key])
    for key in ("lightgbm_mae_mean", "lightgbm_mape_mean", "beats_all_baselines"):
        if key in run.data.metrics:
            client.set_model_version_tag(name, mv.version, key, f"{run.data.metrics[key]:.4g}")
    log.info("registered %s version %s from run %s", name, mv.version, run_id)
    return mv


def latest_training_run_id(experiment: str = DEFAULT_EXPERIMENT) -> str:
    """Most recent run tagged ``stage=training`` in the experiment."""
    import mlflow

    setup_mlflow(experiment)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment)  # created by setup_mlflow if missing
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.stage = 'training'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError("no training runs found - run `make train` first")
    return runs[0].info.run_id


def champion_version(
    *,
    name: str = REGISTERED_MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
    experiment: str = DEFAULT_EXPERIMENT,
) -> Any | None:
    """The MLflow ModelVersion behind ``name@alias``, or None if nothing is promoted yet."""
    import mlflow
    from mlflow.exceptions import MlflowException

    setup_mlflow(experiment)
    try:
        return mlflow.MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def load_model_version(
    version: str | int,
    *,
    name: str = REGISTERED_MODEL_NAME,
    alias: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> LoadedModel:
    """Load a version from MLflow (needs its artifact files - i.e. the runner that trained it)."""
    import mlflow

    setup_mlflow(experiment)
    client = mlflow.MlflowClient()
    mv = client.get_model_version(name, str(version))
    pyfunc = mlflow.pyfunc.load_model(f"models:/{name}/{mv.version}")
    features = _features_from_signature(pyfunc, f"{name} v{mv.version}")
    horizon = HORIZONS.get(mv.tags.get("horizon", DAY_AHEAD.name), DAY_AHEAD)
    train_start, train_end, metrics = _run_metadata(mv.run_id)
    log.info("loaded %s v%s from MLflow with %d features", name, mv.version, len(features))
    return LoadedModel(
        name=name,
        version=str(mv.version),
        alias=alias,
        run_id=mv.run_id,
        features=features,
        horizon=horizon,
        pyfunc=pyfunc,
        train_start=train_start,
        train_end=train_end,
        metrics=metrics,
        source="mlflow",
    )


# --------------------------------------------------------------------------- #
# DB side: export / champion pointer / load
# --------------------------------------------------------------------------- #
def export_model_version(
    engine: Engine,
    version: str | int,
    *,
    name: str = REGISTERED_MODEL_NAME,
    experiment: str = DEFAULT_EXPERIMENT,
) -> dict[str, Any]:
    """Zip the MLflow pyfunc bundle of ``version`` into ``model_artifacts`` (idempotent)."""
    import mlflow

    setup_mlflow(experiment)
    client = mlflow.MlflowClient()
    mv = client.get_model_version(name, str(version))
    local_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{name}/{mv.version}"))
    pyfunc = mlflow.pyfunc.load_model(str(local_dir))
    features = _features_from_signature(pyfunc, f"{name} v{mv.version}")
    train_start, train_end, metrics = _run_metadata(mv.run_id)

    bundle = _zip_dir(local_dir)
    row = {
        "name": name,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "horizon": mv.tags.get("horizon", DAY_AHEAD.name),
        "features": json.dumps(features),
        "train_start": None if train_start is None else train_start.to_pydatetime(),
        "train_end": None if train_end is None else train_end.to_pydatetime(),
        "metrics": json.dumps(metrics) if metrics else None,
        "bundle": bundle,
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "bundle_bytes": len(bundle),
    }
    db.store_model_artifact(engine, **row)
    log.info("exported %s v%s to the database (%.1f MB)", name, mv.version, len(bundle) / 1e6)
    return row


def promote(
    version: str | int,
    *,
    engine: Engine | None = None,
    alias: str = CHAMPION_ALIAS,
    name: str = REGISTERED_MODEL_NAME,
    experiment: str = DEFAULT_EXPERIMENT,
    keep_versions: int = KEEP_EXPORTED_VERSIONS,
) -> None:
    """Make ``version`` the champion: MLflow alias + DB export + DB pointer. Rollback = same call."""
    import mlflow

    engine = engine or db.get_engine()
    db.init_db(engine)
    setup_mlflow(experiment)
    if db.get_model_artifact(engine, name, str(version)) is None:
        export_model_version(engine, version, name=name, experiment=experiment)
    db.set_champion(engine, name, str(version))
    mlflow.MlflowClient().set_registered_model_alias(name, alias, str(version))
    pruned = db.prune_model_artifacts(engine, name, keep=keep_versions)
    log.info("%s@%s -> version %s (pruned %d old exports)", name, alias, version, pruned)


def champion_record(engine: Engine, *, name: str = REGISTERED_MODEL_NAME) -> dict[str, Any] | None:
    """The DB row of the current champion without loading the model (None if not promoted)."""
    return db.get_model_artifact(engine, name, champion=True)


def load_champion(
    engine: Engine | None = None,
    *,
    name: str = REGISTERED_MODEL_NAME,
    cache_dir: Path = MODEL_CACHE_DIR,
) -> LoadedModel:
    """Load the champion from the database export (works on any machine with DATABASE_URL)."""
    import mlflow

    engine = engine or db.get_engine()
    db.init_db(engine)
    row = champion_record(engine, name=name)
    if row is None:
        raise LookupError(
            f"no champion exported for {name!r} - "
            "run `python -m src.models.registry --promote-latest` first"
        )
    if hashlib.sha256(row["bundle"]).hexdigest() != row["bundle_sha256"]:
        raise ValueError(f"corrupt model bundle for {name} v{row['version']}")

    target = Path(cache_dir) / name / f"v{row['version']}"
    if not (target / "MLmodel").exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(row["bundle"])) as zf:
            zf.extractall(target)
    pyfunc = mlflow.pyfunc.load_model(str(target))
    features = json.loads(row["features"])
    log.info(
        "loaded champion %s v%s from the database (%d features)",
        name,
        row["version"],
        len(features),
    )
    return LoadedModel(
        name=name,
        version=str(row["version"]),
        alias=CHAMPION_ALIAS,
        run_id=row["run_id"],
        features=features,
        horizon=HORIZONS.get(row["horizon"], DAY_AHEAD),
        pyfunc=pyfunc,
        train_start=_utc_or_none(row["train_start"]),
        train_end=_utc_or_none(row["train_end"]),
        metrics=json.loads(row["metrics"]) if row["metrics"] else None,
        source="db",
    )


def training_window(
    model: LoadedModel, *, experiment: str = DEFAULT_EXPERIMENT
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """``(train_start, train_end)`` of the model - from its export metadata, else its MLflow run."""
    if model.train_end is not None:
        return model.train_start, model.train_end
    if model.run_id is None:
        return None, None
    try:
        start, end, _ = _run_metadata(model.run_id, experiment=experiment)
    except Exception as err:  # the run may live in another tracking store
        log.warning("could not read run %s from MLflow: %s", model.run_id, err)
        return None, None
    return start, end


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _features_from_signature(pyfunc: Any, label: str) -> list[str]:
    schema = pyfunc.metadata.get_input_schema()
    if schema is None or not schema.input_names():
        raise ValueError(f"{label} has no input signature - cannot recover features")
    return list(schema.input_names())


def _run_metadata(
    run_id: str | None, *, experiment: str = DEFAULT_EXPERIMENT
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, dict[str, float]]:
    if run_id is None:
        return None, None, {}
    import mlflow

    setup_mlflow(experiment)
    data = mlflow.MlflowClient().get_run(run_id).data
    metrics = {k: float(v) for k, v in data.metrics.items() if k in _METRIC_KEYS}
    return (
        _utc_or_none(data.params.get("train_start")),
        _utc_or_none(data.params.get("train_end")),
        metrics,
    )


def _utc_or_none(value: Any) -> pd.Timestamp | None:
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _zip_dir(directory: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(directory.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(directory).as_posix())
    return buf.getvalue()


def _unzip_to_temp(bundle: bytes) -> Path:  # used by tests / ad-hoc inspection
    target = Path(tempfile.mkdtemp(prefix="champion-"))
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        zf.extractall(target)
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.models.registry")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--promote-latest", action="store_true", help="register latest training run + promote"
    )
    g.add_argument("--run-id", help="register this run's model + promote")
    g.add_argument("--promote", metavar="VERSION", help="move the champion to VERSION")
    g.add_argument("--show", action="store_true", help="print the current champion")
    p.add_argument("--name", default=REGISTERED_MODEL_NAME)
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    engine = db.get_engine(args.database_url)
    db.init_db(engine)
    if args.show:
        row = champion_record(engine, name=args.name)
        if row is None:
            print("no champion")
        else:
            print(
                f"{args.name}@champion = v{row['version']} (run {row['run_id']}, "
                f"trained to {row['train_end']}, {row['bundle_bytes'] / 1e6:.1f} MB, "
                f"promoted {row['promoted_at']})"
            )
        return 0
    if args.promote:
        promote(args.promote, engine=engine, name=args.name)
    else:
        run_id = args.run_id or latest_training_run_id()
        mv = register_run_model(run_id, name=args.name)
        promote(mv.version, engine=engine, name=args.name)
    row = champion_record(engine, name=args.name)
    print(f"{args.name}@champion = v{row['version']} (run {row['run_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
