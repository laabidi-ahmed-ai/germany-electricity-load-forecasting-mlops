"""Baselines, metrics, expanding-window CV, LightGBM wrapper and MLflow logging - all offline."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import TARGET, WEATHER_COLUMNS, build_feature_frame
from src.features.horizons import DAY_AHEAD, NOWCAST, select_features
from src.models import evaluate, tracking, train
from src.models.baselines import BASELINE_FACTORIES, SeasonalNaive, seasonal_naive_168
from src.models.evaluate import compute_metrics, cross_validate

FAST_LGBM = {"n_estimators": 60, "learning_rate": 0.2, "num_leaves": 15, "min_child_samples": 5}


def synthetic_frame(days: int = 45, seed: int = 0) -> pd.DataFrame:
    """Load with daily + weekly cycles and a temperature effect, so a model can learn."""
    ts = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    h = np.arange(len(ts))
    temp = 5 + 10 * np.sin(2 * np.pi * h / (24 * 30)) + rng.normal(0, 1, len(ts))
    load = (
        55_000
        + 10_000 * np.sin(2 * np.pi * (h - 6) / 24)
        + 3_000 * np.sin(2 * np.pi * h / 168)
        - 300 * temp
        + rng.normal(0, 400, len(ts))
    )
    load_df = pd.DataFrame({"timestamp_utc": ts, TARGET: load})
    weather = pd.DataFrame({"timestamp_utc": ts})
    for col in WEATHER_COLUMNS:
        weather[col] = temp if col.startswith("temperature") else rng.normal(10, 5, len(ts))
    return build_feature_frame(load_df, weather)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


# --- Baselines ---
def test_seasonal_naive_predicts_the_lag_column(frame) -> None:
    X = frame[select_features(frame.columns, DAY_AHEAD)]
    for lag in (24, 168):
        model = SeasonalNaive(lag).fit(X, frame[TARGET])
        pred = model.predict(X)
        np.testing.assert_allclose(pred, frame[f"load_lag_{lag}"].to_numpy())
        # i.e. exactly the target shifted by `lag` hours
        shifted = frame[TARGET].shift(freq=pd.Timedelta(hours=lag)).reindex(frame.index)
        mask = shifted.notna()
        np.testing.assert_allclose(pred[mask.to_numpy()], shifted[mask].to_numpy())


def test_seasonal_naive_is_day_ahead_valid_by_construction() -> None:
    with pytest.raises(ValueError, match="lag_hours >= 24"):
        SeasonalNaive(1)
    assert seasonal_naive_168().column in select_features(["load_lag_168", "load_lag_1"], DAY_AHEAD)
    assert set(BASELINE_FACTORIES) == {"seasonal_naive_24", "seasonal_naive_168"}


def test_seasonal_naive_requires_its_column() -> None:
    with pytest.raises(KeyError, match="load_lag_168"):
        SeasonalNaive(168).fit(pd.DataFrame({"hour": [1, 2]}))


# --- Metrics ---
def test_metrics_exact_values() -> None:
    y = np.array([100.0, 200.0, 400.0])
    p = np.array([110.0, 180.0, 400.0])
    m = compute_metrics(y, p)
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert m["mape"] == pytest.approx((10 / 100 + 20 / 200 + 0) / 3 * 100)


def test_metrics_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_metrics([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([], [])
    with pytest.raises(ValueError, match="NaN"):
        compute_metrics([1.0], [np.nan])


# --- LightGBM wrapper ---
def test_lightgbm_forecaster_fits_predicts_and_reports_importances(frame) -> None:
    cols = select_features(frame.columns, DAY_AHEAD)
    model = train.LightGBMForecaster(FAST_LGBM).fit(frame[cols], frame[TARGET])
    pred = model.predict(frame[cols])
    assert pred.shape == (len(frame),)
    assert not np.isnan(pred).any()
    assert 1 <= model.best_iteration_ <= FAST_LGBM["n_estimators"]
    imp = model.feature_importances()
    assert list(imp.columns) == ["feature", "gain", "split"]
    assert set(imp["feature"]) == set(cols)
    assert imp["gain"].is_monotonic_decreasing
    assert model.get_params()["n_estimators"] == FAST_LGBM["n_estimators"]


def test_lightgbm_predict_uses_training_column_order(frame) -> None:
    cols = select_features(frame.columns, DAY_AHEAD)
    model = train.LightGBMForecaster(FAST_LGBM).fit(frame[cols], frame[TARGET])
    shuffled = frame[[*reversed(cols), "load_lag_1"]]  # extra column + reversed order
    np.testing.assert_allclose(model.predict(shuffled), model.predict(frame[cols]))


def test_lightgbm_early_stopping_uses_chronological_tail(frame, monkeypatch) -> None:
    """The early-stopping set must be the *last* rows in time, never a random sample."""
    import lightgbm as lgb

    seen: dict[str, pd.DataFrame] = {}
    original_fit = lgb.LGBMRegressor.fit

    def spy(self, X, y, *args, **kwargs):
        seen["fit"] = X
        seen["es"] = kwargs["eval_set"][0][0]
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", spy)
    cols = select_features(frame.columns, DAY_AHEAD)
    train.LightGBMForecaster(FAST_LGBM, early_stopping_frac=0.2).fit(frame[cols], frame[TARGET])

    assert seen["fit"].index.max() < seen["es"].index.min()
    assert len(seen["es"]) == int(len(frame) * 0.2)
    assert seen["es"].index.max() == frame.index.max()


def test_lightgbm_refuses_unsorted_input(frame) -> None:
    cols = select_features(frame.columns, DAY_AHEAD)
    with pytest.raises(ValueError, match="sorted in time"):
        train.LightGBMForecaster(FAST_LGBM).fit(frame[cols].iloc[::-1], frame[TARGET].iloc[::-1])


def test_train_final_model_uses_only_horizon_valid_features(frame) -> None:
    model, cols = train.train_final_model(frame, DAY_AHEAD, FAST_LGBM)
    assert cols == select_features(frame.columns, DAY_AHEAD)
    assert "load_lag_1" not in model.feature_names_
    assert not any(c.startswith("load_roll_") for c in model.feature_names_)


# --- Cross-validation ---
@pytest.fixture(scope="module")
def cv(frame) -> evaluate.CVResult:
    factories = {**BASELINE_FACTORIES, "lightgbm": train.lightgbm_factory(FAST_LGBM)}
    return cross_validate(
        frame, factories, DAY_AHEAD, n_splits=3, val_hours=48, min_train_hours=24 * 20
    )


def test_cv_folds_are_expanding_and_chronological(cv, frame) -> None:
    f = cv.folds
    assert len(f) == 3 * 3  # folds x models
    assert set(f["model"]) == {"seasonal_naive_24", "seasonal_naive_168", "lightgbm"}
    assert (f["n_val"] == 48).all()
    per_fold = f.drop_duplicates("fold").sort_values("fold")
    assert per_fold["n_train"].is_monotonic_increasing  # expanding
    assert (per_fold["val_start"].shift(-1).dropna() > per_fold["val_end"].iloc[:-1]).all()
    assert per_fold["val_end"].iloc[-1] == frame.index.max()
    assert cv.features == select_features(frame.columns, DAY_AHEAD)
    assert cv.horizon == "day_ahead"


def test_cv_summary_and_verdict(cv) -> None:
    s = cv.summary
    assert list(s.index) == ["seasonal_naive_24", "seasonal_naive_168", "lightgbm"]
    assert {"mae_mean", "mae_std", "rmse_mean", "rmse_std", "mape_mean", "mape_std"} <= set(
        s.columns
    )
    assert (s["rmse_mean"] >= s["mae_mean"]).all()
    # Per-fold numbers average to the summary.
    lgb_mae = cv.folds.loc[cv.folds["model"] == "lightgbm", "mae"]
    assert cv.mean("lightgbm") == pytest.approx(lgb_mae.mean())

    comp = cv.beats_baselines()
    assert set(comp) == {"seasonal_naive_24", "seasonal_naive_168"}
    for b, v in comp.items():
        assert v["beats"] == (cv.mean("lightgbm") < cv.mean(b))
        assert v["improvement_pct"] == pytest.approx(
            (cv.mean(b) - cv.mean("lightgbm")) / cv.mean(b) * 100
        )
    # On this synthetic series LightGBM should clearly win.
    assert cv.beats_all_baselines()


def test_cv_report_mentions_every_fold_model_and_the_verdict(cv) -> None:
    text = evaluate.format_report(cv)
    for name in ("seasonal_naive_24", "seasonal_naive_168", "lightgbm"):
        assert name in text
    assert "beats both seasonal-naive baselines: YES" in text
    assert text.count("| 0 |") == 3 and text.count("| 2 |") == 3


def test_cv_every_model_sees_the_same_horizon_valid_columns(frame) -> None:
    seen: dict[str, list[str]] = {}

    class Spy:
        name = "spy"

        def fit(self, X, y):
            seen["cols"] = list(X.columns)
            return self

        def predict(self, X):
            return X["load_lag_24"].to_numpy()

        def get_params(self):
            return {}

    cross_validate(frame, {"spy": Spy}, DAY_AHEAD, n_splits=2, val_hours=24, min_train_hours=100)
    assert seen["cols"] == select_features(frame.columns, DAY_AHEAD)
    assert "load_lag_1" not in seen["cols"]

    cross_validate(frame, {"spy": Spy}, NOWCAST, n_splits=2, val_hours=24, min_train_hours=100)
    assert "load_lag_1" in seen["cols"]


# --- MLflow ---
def test_tracking_uri_resolution(monkeypatch, tmp_path) -> None:
    from config.settings import PROJECT_ROOT

    # default: relative sqlite path anchored at the project root
    assert tracking.resolve_tracking_uri() == (
        f"sqlite:///{(PROJECT_ROOT / 'mlruns' / 'mlflow.db').resolve().as_posix()}"
    )
    abs_db = tmp_path / "x.db"
    assert tracking.resolve_tracking_uri(f"sqlite:///{abs_db.as_posix()}") == (
        f"sqlite:///{abs_db.resolve().as_posix()}"
    )
    # legacy file store: plain paths -> file:// URI + opt-in flag
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    assert tracking.resolve_tracking_uri("./mlruns") == (PROJECT_ROOT / "mlruns").resolve().as_uri()
    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
    assert tracking.resolve_tracking_uri(str(tmp_path)) == tmp_path.resolve().as_uri()
    # remote / full URIs untouched
    assert tracking.resolve_tracking_uri("http://mlflow:5000") == "http://mlflow:5000"
    # Postgres backends are routed to the psycopg 3 driver we ship (like DATABASE_URL).
    assert tracking.resolve_tracking_uri("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert (
        tracking.resolve_tracking_uri("postgresql+psycopg://u:p@h/db")
        == "postgresql+psycopg://u:p@h/db"
    )
    # settings override via env
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 's.db').as_posix()}")
    from config.settings import get_settings

    get_settings.cache_clear()
    assert tracking.resolve_tracking_uri().endswith("/s.db")


def test_training_run_is_logged_to_mlflow(frame, cv, tmp_path, monkeypatch) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    from config.settings import get_settings

    get_settings.cache_clear()

    model, cols = train.train_final_model(frame, DAY_AHEAD, FAST_LGBM)
    run_id = train.log_training_run(model, cols, cv, DAY_AHEAD, frame, experiment="test-exp")

    client = mlflow.MlflowClient(tracking_uri=tracking.resolve_tracking_uri())
    run = client.get_run(run_id)
    assert run.data.tags["horizon"] == "day_ahead"
    assert run.data.params["n_features"] == str(len(cols))
    assert run.data.params["lgbm_n_estimators"] == str(FAST_LGBM["n_estimators"])
    assert "lightgbm_mae_mean" in run.data.metrics
    assert "seasonal_naive_168_mae_mean" in run.data.metrics
    assert run.data.metrics["beats_all_baselines"] == 1.0
    # per-fold metrics are logged as steps
    hist = client.get_metric_history(run_id, "lightgbm_mae")
    assert sorted(m.step for m in hist) == [0, 1, 2]
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"features.json", "cv_report.md", "cv_folds.csv", "feature_importance.csv"} <= artifacts

    # MLflow 3 stores the model as a LoggedModel linked to the run; it must load back
    # via the run URI and reproduce the in-memory predictions.
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    np.testing.assert_allclose(
        loaded.predict(frame[cols].head(10)), model.predict(frame.head(10)), rtol=1e-6
    )


def test_standalone_cv_run_is_logged(cv, tmp_path, monkeypatch) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    from config.settings import get_settings

    get_settings.cache_clear()
    run_id = evaluate.log_cv_run(cv, experiment="test-cv")
    run = mlflow.MlflowClient(tracking_uri=tracking.resolve_tracking_uri()).get_run(run_id)
    assert run.data.tags["stage"] == "evaluation"
    assert run.data.params["n_splits"] == "3"
    assert "lightgbm_vs_seasonal_naive_168_improvement_pct" in run.data.metrics


# --- CLI ---
def test_train_cli_end_to_end(frame, tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "features.parquet"
    frame.to_parquet(path)
    original_factory = train.lightgbm_factory
    monkeypatch.setattr(train, "lightgbm_factory", lambda params=None: original_factory(FAST_LGBM))
    monkeypatch.setattr(train, "LightGBMForecaster", _FastForecaster)
    rc = train.main(
        [
            "--features",
            str(path),
            "--n-splits",
            "2",
            "--val-hours",
            "24",
            "--min-train-hours",
            "200",
            "--no-mlflow",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "beats both seasonal-naive baselines" in out
    assert "Top features by gain" in out


def test_evaluate_cli_end_to_end(frame, tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "features.parquet"
    frame.to_parquet(path)
    original_factory = train.lightgbm_factory
    monkeypatch.setattr(train, "lightgbm_factory", lambda params=None: original_factory(FAST_LGBM))
    rc = evaluate.main(
        [
            "--features",
            str(path),
            "--n-splits",
            "2",
            "--val-hours",
            "24",
            "--min-train-hours",
            "200",
            "--no-mlflow",
        ]
    )
    assert rc == 0
    assert "Mean over folds" in capsys.readouterr().out


class _FastForecaster(train.LightGBMForecaster):
    def __init__(self, params=None):
        super().__init__({**FAST_LGBM, **(params or {})})
