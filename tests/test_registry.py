"""Model registry tests: MLflow lineage + DB champion export (temporary SQLite stores, offline)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from config.settings import get_settings
from src.data import db
from src.features.horizons import DAY_AHEAD, select_features
from src.models import evaluate, registry, train
from src.models.baselines import BASELINE_FACTORIES
from tests.test_models import FAST_LGBM, synthetic_frame


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """One MLflow store + one data DB with one logged training run, shared by the tests."""
    root = tmp_path_factory.mktemp("registry")
    return {
        "uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        "db_url": f"sqlite:///{(root / 'data.db').as_posix()}",
        "cache": root / "model-cache",
        "frame": synthetic_frame(days=40),
    }


@pytest.fixture
def env(store, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store["uri"])
    monkeypatch.setenv("DATABASE_URL", store["db_url"])
    get_settings.cache_clear()
    if "engine" not in store:
        store["engine"] = db.get_engine(store["db_url"])
        db.init_db(store["engine"])
    yield store
    get_settings.cache_clear()


@pytest.fixture
def training_run(env):
    if "run_id" not in env:
        df = env["frame"]
        cv = evaluate.cross_validate(
            df,
            {**BASELINE_FACTORIES, "lightgbm": train.lightgbm_factory(FAST_LGBM)},
            DAY_AHEAD,
            n_splits=2,
            val_hours=24,
            min_train_hours=200,
        )
        model, cols = train.train_final_model(df, DAY_AHEAD, FAST_LGBM)
        env["run_id"] = train.log_training_run(model, cols, cv, DAY_AHEAD, df)
        env["cols"] = cols
        env["model"] = model
    return env


def test_no_champion_before_promotion(training_run) -> None:
    engine = training_run["engine"]
    assert registry.champion_version(name="fresh-model-name") is None
    assert registry.champion_record(engine, name="fresh-model-name") is None
    with pytest.raises(LookupError, match="promote-latest"):
        registry.load_champion(engine, name="fresh-model-name")


def test_register_promote_export_and_load_from_db(training_run) -> None:
    engine = training_run["engine"]
    mv = registry.register_run_model(training_run["run_id"])
    assert mv.name == registry.REGISTERED_MODEL_NAME and mv.run_id == training_run["run_id"]

    registry.promote(mv.version, engine=engine)

    # MLflow alias moved ...
    assert registry.champion_version().version == mv.version
    # ... and the version is exported + flagged in the DB with its metadata.
    row = registry.champion_record(engine)
    assert row["version"] == str(mv.version) and row["run_id"] == training_run["run_id"]
    assert json.loads(row["features"]) == training_run["cols"]
    assert row["horizon"] == "day_ahead"
    assert row["train_end"] is not None and row["bundle_bytes"] > 10_000
    assert "lightgbm_mae_mean" in json.loads(row["metrics"])

    # Load from the DB into a fresh cache dir (= a fresh runner) and compare predictions.
    champ = registry.load_champion(engine, cache_dir=training_run["cache"])
    assert champ.source == "db" and champ.version == str(mv.version)
    assert champ.features == select_features(training_run["frame"].columns, DAY_AHEAD)
    assert "load_lag_1" not in champ.features
    assert champ.train_end is not None and champ.train_start < champ.train_end
    assert champ.metrics["beats_all_baselines"] == 1.0

    df = training_run["frame"]
    np.testing.assert_allclose(
        champ.predict(df.head(20)), training_run["model"].predict(df.head(20)), rtol=1e-6
    )
    via_mlflow = registry.load_model_version(mv.version)
    np.testing.assert_allclose(champ.predict(df.tail(20)), via_mlflow.predict(df.tail(20)))
    assert via_mlflow.source == "mlflow" and via_mlflow.features == champ.features

    champ.check_features(list(df.columns))
    with pytest.raises(ValueError, match="mismatch"):
        champ.check_features([c for c in df.columns if c != "hour"])
    with pytest.raises(ValueError, match="missing model inputs"):
        champ.predict(df.drop(columns=["load_lag_24"]))

    # training_window comes straight from the export metadata (no MLflow needed).
    assert registry.training_window(champ) == (champ.train_start, champ.train_end)


def test_loading_uses_the_cache_and_detects_corruption(training_run) -> None:
    engine = training_run["engine"]
    cache = training_run["cache"]
    row = registry.champion_record(engine)
    assert (cache / registry.REGISTERED_MODEL_NAME / f"v{row['version']}" / "MLmodel").exists()
    registry.load_champion(engine, cache_dir=cache)  # second load: cache hit, no error

    from sqlalchemy import update

    with engine.begin() as conn:
        conn.execute(update(db.ModelArtifact).values(bundle_sha256="0" * 64))
    with pytest.raises(ValueError, match="corrupt"):
        registry.load_champion(engine, cache_dir=cache)
    with engine.begin() as conn:  # restore
        import hashlib

        conn.execute(
            update(db.ModelArtifact).values(bundle_sha256=hashlib.sha256(row["bundle"]).hexdigest())
        )


def test_version_tags_carry_cv_metrics(training_run) -> None:
    import mlflow

    mv = registry.champion_version()
    tags = mlflow.MlflowClient().get_model_version(mv.name, mv.version).tags
    assert tags["horizon"] == "day_ahead" and tags["model"] == "lightgbm"
    assert "lightgbm_mae_mean" in tags and "beats_all_baselines" in tags


def test_promote_moves_both_pointers_rollback(training_run) -> None:
    engine = training_run["engine"]
    v1 = registry.champion_version().version
    mv2 = registry.register_run_model(training_run["run_id"])  # a second version, same run
    assert int(mv2.version) == int(v1) + 1

    registry.promote(mv2.version, engine=engine)
    assert registry.champion_version().version == mv2.version
    assert registry.champion_record(engine)["version"] == str(mv2.version)
    listing = db.list_model_artifacts(engine, registry.REGISTERED_MODEL_NAME)
    assert len(listing) == 2 and listing["is_champion"].sum() == 1

    registry.promote(v1, engine=engine)  # rollback = one call, both pointers move
    assert registry.champion_version().version == v1
    assert registry.champion_record(engine)["version"] == str(v1)
    assert registry.load_champion(engine, cache_dir=training_run["cache"]).version == str(v1)


def test_promote_prunes_old_exports_but_never_the_champion(training_run) -> None:
    engine = training_run["engine"]
    champ = registry.champion_record(engine)["version"]
    for _ in range(3):
        mv = registry.register_run_model(training_run["run_id"])
        registry.promote(mv.version, engine=engine, keep_versions=2)
    listing = db.list_model_artifacts(engine, registry.REGISTERED_MODEL_NAME)
    assert len(listing) == 2
    assert listing["is_champion"].sum() == 1 and str(mv.version) in set(listing["version"])
    registry.promote(champ, engine=engine, keep_versions=2)  # old champion re-exported on demand
    assert registry.champion_record(engine)["version"] == str(champ)


def test_latest_training_run_id(training_run) -> None:
    assert registry.latest_training_run_id() == training_run["run_id"]
    with pytest.raises(LookupError, match="no training runs"):
        registry.latest_training_run_id(experiment="empty-experiment")


def test_cli(training_run, capsys) -> None:
    url = training_run["db_url"]
    assert registry.main(["--show", "--database-url", url]) == 0
    assert "@champion = v" in capsys.readouterr().out
    before = int(registry.champion_version().version)
    assert registry.main(["--promote-latest", "--database-url", url]) == 0
    assert int(registry.champion_version().version) > before
    assert registry.main(["--promote", str(before), "--database-url", url]) == 0
    assert int(registry.champion_version().version) == before
    assert registry.champion_record(training_run["engine"])["version"] == str(before)
    assert registry.main(["--show", "--name", "unknown-model", "--database-url", url]) == 0
    assert "no champion" in capsys.readouterr().out
