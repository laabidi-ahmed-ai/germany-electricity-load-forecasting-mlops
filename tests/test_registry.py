"""Model registry tests: MLflow lineage + DB champion export (temporary SQLite stores, offline).

One training run, one registered + promoted version and one data DB are built once per
module (training is the slow part). Every test starts from at least that state and asserts
only relative facts, so the tests pass in any order and in isolation.
"""

from __future__ import annotations

import hashlib
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
    root = tmp_path_factory.mktemp("registry")
    return {
        "uri": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        "db_url": f"sqlite:///{(root / 'data.db').as_posix()}",
        "cache": root / "model-cache",
        "frame": synthetic_frame(days=40),
    }


@pytest.fixture
def env(store, monkeypatch):
    """Point settings at the module store and make sure the base state exists."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store["uri"])
    monkeypatch.setenv("DATABASE_URL", store["db_url"])
    get_settings.cache_clear()
    if "engine" not in store:
        store["engine"] = db.get_engine(store["db_url"])
        db.init_db(store["engine"])
        df = store["frame"]
        cv = evaluate.cross_validate(
            df,
            {**BASELINE_FACTORIES, "lightgbm": train.lightgbm_factory(FAST_LGBM)},
            DAY_AHEAD,
            n_splits=2,
            val_hours=24,
            min_train_hours=200,
        )
        model, cols = train.train_final_model(df, DAY_AHEAD, FAST_LGBM)
        store["run_id"] = train.log_training_run(model, cols, cv, DAY_AHEAD, df)
        store["cols"] = cols
        store["model"] = model
        mv = registry.register_run_model(store["run_id"])
        registry.promote(mv.version, engine=store["engine"])
    yield store
    get_settings.cache_clear()


def alias_version(name: str = registry.REGISTERED_MODEL_NAME) -> str:
    """The MLflow version behind ``name@champion`` (read straight from MLflow)."""
    import mlflow

    return mlflow.MlflowClient().get_model_version_by_alias(name, registry.CHAMPION_ALIAS).version


def test_no_champion_before_promotion(env) -> None:
    import mlflow
    from mlflow.exceptions import MlflowException

    engine = env["engine"]
    with pytest.raises(MlflowException):
        mlflow.MlflowClient().get_model_version_by_alias("fresh-model-name", "champion")
    assert registry.champion_record(engine, name="fresh-model-name") is None
    with pytest.raises(LookupError, match="promote-latest"):
        registry.load_champion(engine, name="fresh-model-name")


def test_register_promote_export_and_load_from_db(env) -> None:
    engine = env["engine"]
    mv = registry.register_run_model(env["run_id"])
    assert mv.name == registry.REGISTERED_MODEL_NAME and mv.run_id == env["run_id"]

    registry.promote(mv.version, engine=engine)

    # MLflow alias moved ...
    assert alias_version() == mv.version
    # ... and the version is exported + flagged in the DB with its metadata.
    row = registry.champion_record(engine)
    assert row["version"] == str(mv.version) and row["run_id"] == env["run_id"]
    assert json.loads(row["features"]) == env["cols"]
    assert row["horizon"] == "day_ahead"
    assert row["train_end"] is not None and row["bundle_bytes"] > 10_000
    assert "lightgbm_mae_mean" in json.loads(row["metrics"])

    # Load from the DB into a fresh cache dir (= a fresh runner) and compare predictions.
    champ = registry.load_champion(engine, cache_dir=env["cache"])
    assert champ.source == "db" and champ.version == str(mv.version)
    assert champ.features == select_features(env["frame"].columns, DAY_AHEAD)
    assert "load_lag_1" not in champ.features
    assert champ.train_end is not None and champ.train_start < champ.train_end
    assert champ.metrics["beats_all_baselines"] == 1.0

    df = env["frame"]
    np.testing.assert_allclose(
        champ.predict(df.head(20)), env["model"].predict(df.head(20)), rtol=1e-6
    )
    # The exported bundle is the very model MLflow holds for that version.
    import mlflow

    via_mlflow = mlflow.pyfunc.load_model(f"models:/{registry.REGISTERED_MODEL_NAME}/{mv.version}")
    np.testing.assert_allclose(
        champ.predict(df.tail(20)), via_mlflow.predict(df.tail(20)[champ.features])
    )

    champ.check_features(list(df.columns))
    with pytest.raises(ValueError, match="mismatch"):
        champ.check_features([c for c in df.columns if c != "hour"])
    with pytest.raises(ValueError, match="missing model inputs"):
        champ.predict(df.drop(columns=["load_lag_24"]))

    # training_window comes straight from the export metadata (no MLflow needed).
    assert registry.training_window(champ) == (champ.train_start, champ.train_end)


def test_loading_uses_the_cache_and_detects_corruption(env) -> None:
    engine = env["engine"]
    cache = env["cache"]
    row = registry.champion_record(engine)
    registry.load_champion(engine, cache_dir=cache)
    assert (cache / registry.REGISTERED_MODEL_NAME / f"v{row['version']}" / "MLmodel").exists()
    registry.load_champion(engine, cache_dir=cache)  # second load: cache hit, no error

    from sqlalchemy import update

    where = (db.ModelArtifact.name == row["name"], db.ModelArtifact.version == row["version"])
    with engine.begin() as conn:
        conn.execute(update(db.ModelArtifact).where(*where).values(bundle_sha256="0" * 64))
    with pytest.raises(ValueError, match="corrupt"):
        registry.load_champion(engine, cache_dir=cache)
    with engine.begin() as conn:  # restore
        conn.execute(
            update(db.ModelArtifact)
            .where(*where)
            .values(bundle_sha256=hashlib.sha256(row["bundle"]).hexdigest())
        )


def test_version_tags_carry_cv_metrics(env) -> None:
    import mlflow

    tags = (
        mlflow.MlflowClient()
        .get_model_version(registry.REGISTERED_MODEL_NAME, alias_version())
        .tags
    )
    assert tags["horizon"] == "day_ahead" and tags["model"] == "lightgbm"
    assert "lightgbm_mae_mean" in tags and "beats_all_baselines" in tags


def test_promote_moves_both_pointers_rollback(env) -> None:
    engine = env["engine"]
    v1 = alias_version()
    mv2 = registry.register_run_model(env["run_id"])  # a newer version, same run
    assert int(mv2.version) > int(v1)

    registry.promote(mv2.version, engine=engine)
    assert alias_version() == mv2.version
    assert registry.champion_record(engine)["version"] == str(mv2.version)
    listing = db.list_model_artifacts(engine, registry.REGISTERED_MODEL_NAME)
    assert listing["is_champion"].sum() == 1 and str(mv2.version) in set(listing["version"])

    registry.promote(v1, engine=engine)  # rollback = one call, both pointers move
    assert alias_version() == v1
    assert registry.champion_record(engine)["version"] == str(v1)
    assert registry.load_champion(engine, cache_dir=env["cache"]).version == str(v1)


def test_promote_prunes_old_exports_but_never_the_champion(env) -> None:
    engine = env["engine"]
    champ = registry.champion_record(engine)["version"]
    for _ in range(3):
        mv = registry.register_run_model(env["run_id"])
        registry.promote(mv.version, engine=engine, keep_versions=2)
    listing = db.list_model_artifacts(engine, registry.REGISTERED_MODEL_NAME)
    assert len(listing) == 2
    assert listing["is_champion"].sum() == 1 and str(mv.version) in set(listing["version"])
    registry.promote(champ, engine=engine, keep_versions=2)  # old champion re-exported on demand
    assert registry.champion_record(engine)["version"] == str(champ)


def test_latest_training_run_id(env) -> None:
    assert registry.latest_training_run_id() == env["run_id"]
    with pytest.raises(LookupError, match="no training runs"):
        registry.latest_training_run_id(experiment="empty-experiment")


def test_cli(env, capsys) -> None:
    url = env["db_url"]
    assert registry.main(["--show", "--database-url", url]) == 0
    assert "@champion = v" in capsys.readouterr().out
    before = int(alias_version())
    assert registry.main(["--promote-latest", "--database-url", url]) == 0
    assert int(alias_version()) > before
    assert registry.main(["--promote", str(before), "--database-url", url]) == 0
    assert int(alias_version()) == before
    assert registry.champion_record(env["engine"])["version"] == str(before)
    assert registry.main(["--show", "--name", "unknown-model", "--database-url", url]) == 0
    assert "no champion" in capsys.readouterr().out
