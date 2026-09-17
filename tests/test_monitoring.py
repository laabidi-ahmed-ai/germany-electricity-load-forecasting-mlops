"""Monitoring tests (offline): performance head-to-head, drift summaries, triggers, champion/challenger."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from config.settings import get_settings
from src.data import db
from src.features.build_features import TARGET, WEATHER_COLUMNS, build_feature_frame
from src.features.horizons import DAY_AHEAD, select_features
from src.models.registry import LoadedModel
from src.monitoring import drift as drift_mod
from src.monitoring import performance as perf
from src.monitoring import retrain
from tests.test_models import FAST_LGBM

N_DAYS = 40
START = pd.Timestamp("2024-01-01T00:00Z")
AS_OF = START + pd.Timedelta(days=N_DAYS) - pd.Timedelta(hours=1)  # last hour of the data


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def synthetic_load(n_hours: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = np.arange(n_hours)
    return (
        55_000
        + 10_000 * np.sin(2 * np.pi * (h - 6) / 24)
        + 3_000 * np.sin(2 * np.pi * h / 168)
        + rng.normal(0, 300, n_hours)
    )


def seed_db(
    engine,
    *,
    days: int = N_DAYS,
    model_noise: float = 500.0,
    model_bias: float = 0.0,
    official_noise: float = 1_000.0,
    model_version: str = "1",
    with_model: bool = True,
) -> pd.DataFrame:
    ts = pd.date_range(START, periods=24 * days, freq="h", tz="UTC")
    load = synthetic_load(len(ts))
    rng = np.random.default_rng(1)
    db.upsert_dataframe(
        engine, db.LoadActual, pd.DataFrame({"timestamp_utc": ts, TARGET: load, "source": "smard"})
    )
    db.upsert_dataframe(
        engine,
        db.LoadForecastOfficial,
        pd.DataFrame(
            {
                "timestamp_utc": ts,
                "forecast_mw": load + rng.normal(0, official_noise, len(ts)),
                "source": "smard",
            }
        ),
    )
    if with_model:
        db.upsert_dataframe(
            engine,
            db.LoadForecastModel,
            pd.DataFrame(
                {
                    "timestamp_utc": ts,
                    "model_version": model_version,
                    "model_name": "germany-load-day-ahead",
                    "forecast_mw": load + model_bias + rng.normal(0, model_noise, len(ts)),
                    "issued_at": datetime(2024, 1, 1, tzinfo=UTC),
                }
            ),
        )
    return pd.DataFrame({"timestamp_utc": ts, TARGET: load})


def feature_frame(load: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weather = pd.DataFrame({"timestamp_utc": load["timestamp_utc"]})
    for col in WEATHER_COLUMNS:
        weather[col] = rng.normal(10, 5, len(load))
    return build_feature_frame(load, weather)


class StubPyfunc:
    """Predicts last week's load (a strong-ish but beatable stand-in for the champion)."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["load_lag_168"].to_numpy() + self.bias


def fake_champion(features: list[str], *, bias: float = 0.0, version: str = "1") -> LoadedModel:
    return LoadedModel(
        name="germany-load-day-ahead",
        version=version,
        alias="champion",
        run_id="run-champ",
        features=features,
        horizon=DAY_AHEAD,
        pyfunc=StubPyfunc(bias),
    )


def fake_runner(drifted: set[str], *, score_ok: float = 0.02, score_bad: float = 0.5):
    """A stand-in for Evidently returning its 0.7 metrics structure."""

    def _run(reference, current, columns, threshold=0.25):
        out = [
            {
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount", "columns": columns},
                "value": {"count": float(len(drifted & set(columns))), "share": 0.0},
            }
        ]
        for c in columns:
            out.append(
                {
                    "config": {
                        "type": "evidently:metric_v2:ValueDrift",
                        "column": c,
                        "method": "Wasserstein distance (normed)",
                        "threshold": threshold,
                    },
                    "value": score_bad if c in drifted else score_ok,
                }
            )
        return out

    return _run


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def test_aligned_frame_inner_joins_on_common_hours(engine) -> None:
    seed_db(engine)
    # remove some official rows -> those hours must vanish from the alignment
    from sqlalchemy import delete

    cut = START + pd.Timedelta(days=5)
    with engine.begin() as conn:
        conn.execute(
            delete(db.LoadForecastOfficial).where(
                db.LoadForecastOfficial.timestamp_utc < cut.to_pydatetime()
            )
        )
    a = perf.aligned_frame(engine)
    assert list(a.columns) == [
        "timestamp_utc",
        "actual_mw",
        "model_mw",
        "official_mw",
        "model_version",
    ]
    assert a["timestamp_utc"].min() == cut
    assert len(a) == 24 * (N_DAYS - 5)
    assert a["timestamp_utc"].is_monotonic_increasing


def test_aligned_frame_uses_latest_issued_forecast_per_hour(engine) -> None:
    seed_db(engine)
    ts = START + pd.Timedelta(days=3)
    later = pd.DataFrame(
        {
            "timestamp_utc": [ts],
            "model_version": ["2"],
            "model_name": ["germany-load-day-ahead"],
            "forecast_mw": [99_999.0],
            "issued_at": [datetime(2024, 2, 1, tzinfo=UTC)],
        }
    )
    db.upsert_dataframe(engine, db.LoadForecastModel, later)
    a = perf.aligned_frame(engine)
    row = a[a["timestamp_utc"] == ts].iloc[0]
    assert row["model_mw"] == 99_999.0 and row["model_version"] == "2"
    assert len(a) == 24 * N_DAYS  # still one row per hour
    only_v1 = perf.aligned_frame(engine, model_version="1")
    assert only_v1[only_v1["timestamp_utc"] == ts]["model_mw"].iloc[0] != 99_999.0


def test_report_head_to_head_on_same_hours(engine) -> None:
    seed_db(engine, model_noise=500, official_noise=1_000)
    report = perf.compute_report(engine, as_of=AS_OF)
    assert report.has_data
    assert report.model_versions == ["1"]
    for days in (7, 30):
        w = report.window(days)
        assert w.judged and w.n_hours == 24 * days
        assert w.model_mae < w.official_mae
        assert w.model_beats_official is True
        assert w.improvement_pct > 0
        assert 0 < w.model_mape < w.official_mape < 5
    assert len(report.daily) == 30
    assert set(report.official_only) == {7, 30}
    assert report.official_only[30]["n_hours"] == 720
    text = report.summary()
    assert "BEATS" in text and "official forecast alone" in text
    d = report.to_dict()
    assert d["windows"][0]["window_days"] == 7


def test_report_when_model_is_worse(engine) -> None:
    seed_db(engine, model_noise=3_000, official_noise=500)
    w = perf.compute_report(engine, as_of=AS_OF).window(7)
    assert w.model_beats_official is False and w.improvement_pct < 0
    assert "LOSES TO" in perf.compute_report(engine, as_of=AS_OF).summary()


def test_report_without_model_forecasts_is_honest(engine) -> None:
    seed_db(engine, with_model=False)
    report = perf.compute_report(engine, as_of=AS_OF)
    assert not report.has_data
    assert report.n_aligned_hours == 0
    assert all(not w.judged for w in report.windows)
    assert report.official_only[7]["n_hours"] == 168  # the benchmark alone is still scored
    assert "no scorable model forecasts yet" in report.summary()


def test_window_needs_minimum_hours(engine) -> None:
    seed_db(engine, days=2)
    as_of = START + pd.Timedelta(hours=10)
    w = perf.window_metrics(perf.aligned_frame(engine), as_of, 7)
    assert w.n_hours == 11 and not w.judged


def test_backtest_vs_official_is_out_of_sample(engine) -> None:
    load = seed_db(engine, days=400, with_model=False, official_noise=1_500)
    frame = feature_frame(load)
    bt = perf.backtest_vs_official(engine, days=14, frame=frame, params=FAST_LGBM)
    assert bt.n_hours == 24 * 14
    assert bt.train_end < bt.start  # trained strictly before the window
    assert bt.train_rows == len(frame[frame.index <= bt.start - pd.Timedelta(hours=1)])
    assert bt.model_mae > 0 and bt.official_mae > 0
    assert len(bt.daily) == 14 and len(bt.by_hour) == 24
    assert bt.model_beats_official == (bt.model_mae < bt.official_mae)
    assert "Out-of-sample backtest" in bt.summary()


def test_backtest_needs_a_year_of_training(engine) -> None:
    load = seed_db(engine, days=60, with_model=False)
    with pytest.raises(ValueError, match="1 year"):
        perf.backtest_vs_official(engine, days=7, frame=feature_frame(load), params=FAST_LGBM)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def test_drift_columns_exclude_calendar_and_nowcast_features() -> None:
    frame = feature_frame(
        pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(START, periods=24 * 10, freq="h", tz="UTC"),
                TARGET: synthetic_load(240),
            }
        )
    )
    cols = drift_mod.drift_columns(list(frame.columns))
    assert cols == ["load_lag_24", "load_lag_48", "load_lag_168", *WEATHER_COLUMNS]
    assert not any(c in cols for c in ("hour", "day_of_week", "month", "is_holiday", "load_lag_1"))


def test_seasonal_reference_selects_same_season_of_past_years() -> None:
    idx = pd.date_range("2021-01-01", "2024-12-31T23:00", freq="h", tz="UTC")
    training = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    current = pd.date_range("2024-09-01", "2024-09-30T23:00", freq="h", tz="UTC")
    ref = drift_mod.seasonal_reference(training, current, pad_days=7)
    assert {2021, 2022, 2023} <= set(ref.index.year)
    assert (ref.index < current.min()).all()  # strictly before the current window (2024 pad ok)
    assert ref.index.month.min() == 8 and ref.index.month.max() == 10
    assert (ref.index.dayofyear >= current.min().dayofyear - 7).all()
    assert (ref.index.dayofyear <= current.max().dayofyear + 7).all()
    # year-boundary window (Dec -> Jan) wraps correctly
    cur2 = pd.date_range("2024-12-20", "2024-12-31T23:00", freq="h", tz="UTC")
    ref2 = drift_mod.seasonal_reference(training, cur2, pad_days=7)
    assert {12, 1} <= set(ref2.index.month)
    assert not ((ref2.index.month == 1) & (ref2.index.day > 7)).any()


def test_summarize_metrics_reads_evidently_structure() -> None:
    metrics = fake_runner({"a"})(None, None, ["a", "b"], 0.25)
    out = drift_mod.summarize_metrics(metrics)
    assert set(out) == {"a", "b"}
    assert out["a"].drifted and out["a"].score == 0.5 and out["a"].threshold == 0.25
    assert not out["b"].drifted and out["b"].method.startswith("Wasserstein")
    # tolerate the dict-valued variant some metrics use
    metrics.append(
        {
            "config": {"type": "x:ValueDrift", "column": "c", "threshold": 0.2},
            "value": {"drift_score": 0.3},
        }
    )
    assert drift_mod.summarize_metrics(metrics)["c"].drifted


def test_compute_drift_flags_and_prediction_drift(engine) -> None:
    load = seed_db(engine, days=120, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    training = frame[frame.index < frame.index.max() - pd.Timedelta(days=40)]

    # no drift
    r = drift_mod.compute_drift(
        engine, champ, window_days=30, frame=frame, training=training, runner=fake_runner(set())
    )
    assert r.n_monitored == 8 and r.n_drifted == 0 and not r.flag
    assert r.prediction is not None and not r.prediction_drift
    assert r.target is not None and not r.target_drift
    assert r.current_rows == 24 * 30 and r.reference_rows > 0
    assert "DRIFT FLAG: no" in r.summary()

    # 4 of 8 features drift (50% >= 50%) -> dataset drift; 3 of 8 does not
    r = drift_mod.compute_drift(
        engine,
        champ,
        window_days=30,
        frame=frame,
        training=training,
        runner=fake_runner(
            {"load_lag_24", "load_lag_168", "temperature_2m_de_avg", "cloud_cover_de_avg"}
        ),
    )
    assert r.dataset_drift and r.flag
    assert r.drifted_columns == [
        "load_lag_24",
        "load_lag_168",
        "temperature_2m_de_avg",
        "cloud_cover_de_avg",
    ]
    assert r.to_dict()["share_drifted"] == pytest.approx(4 / 8)
    r = drift_mod.compute_drift(
        engine,
        champ,
        window_days=30,
        frame=frame,
        training=training,
        runner=fake_runner({"load_lag_24", "load_lag_168", "temperature_2m_de_avg"}),
    )
    assert not r.dataset_drift and not r.flag
    # ... unless the share threshold is lowered
    r = drift_mod.compute_drift(
        engine,
        champ,
        window_days=30,
        frame=frame,
        training=training,
        drift_share_threshold=0.3,
        runner=fake_runner({"load_lag_24", "load_lag_168", "temperature_2m_de_avg"}),
    )
    assert r.dataset_drift

    # only the prediction drifts -> flag, but no dataset drift
    r = drift_mod.compute_drift(
        engine,
        champ,
        window_days=30,
        frame=frame,
        training=training,
        runner=fake_runner({"prediction"}),
    )
    assert not r.dataset_drift and r.prediction_drift and r.flag

    # only the target drifts -> informational, no flag
    r = drift_mod.compute_drift(
        engine, champ, window_days=30, frame=frame, training=training, runner=fake_runner({TARGET})
    )
    assert r.target_drift and not r.flag

    # without a model there is no prediction column
    r = drift_mod.compute_drift(
        engine, None, window_days=30, frame=frame, training=training, runner=fake_runner(set())
    )
    assert r.prediction is None


def test_compute_drift_with_real_evidently(engine) -> None:
    """Pin the Evidently 0.7 API contract on a tiny frame (offline)."""
    load = seed_db(engine, days=60, with_model=False)
    frame = feature_frame(load)
    training = frame[frame.index < frame.index.max() - pd.Timedelta(days=10)]
    shifted = frame.copy()
    # make the temperature of the current window drift hard
    recent = shifted.index > shifted.index.max() - pd.Timedelta(days=7)
    shifted.loc[recent, "temperature_2m_de_avg"] += 40
    r = drift_mod.compute_drift(
        engine, None, window_days=7, frame=shifted, training=training, pad_days=60
    )
    assert r.n_monitored == 8
    assert all(c.threshold == drift_mod.DEFAULT_COLUMN_THRESHOLD for c in r.columns)
    assert all(c.method.lower().startswith("wasserstein") for c in r.columns)
    temp = next(c for c in r.columns if c.column == "temperature_2m_de_avg")
    assert temp.drifted and temp.score > temp.threshold
    assert not next(c for c in r.columns if c.column == "wind_speed_10m_de_avg").drifted
    assert r.target is not None


# --------------------------------------------------------------------------- #
# Triggers
# --------------------------------------------------------------------------- #
@pytest.fixture
def cfg() -> retrain.RetrainConfig:
    return retrain.RetrainConfig(
        min_holdout_hours=24 * 3,
        holdout_days=5,
        lgbm_params=FAST_LGBM,
        cv_n_splits=2,
        cv_val_hours=24,
        cv_min_train_hours=24 * 30,
    )


def test_triggers_insufficient_data_do_not_fire(engine, cfg) -> None:
    load = seed_db(engine, days=60, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine, champ, cfg, as_of=AS_OF, frame=frame, drift_runner=fake_runner(set())
    )
    assert not d.triggered
    assert [c.name for c in d.checks] == ["error", "official", "drift"]
    assert "insufficient data" in d.checks[0].detail and "insufficient data" in d.checks[1].detail
    assert "not triggered" in d.summary()


def test_error_trigger_fires_on_high_mape(engine, cfg) -> None:
    load = seed_db(engine, days=60, model_noise=100, model_bias=4_000)  # ~7% MAPE
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    as_of = load["timestamp_utc"].max()
    d = retrain.evaluate_triggers(
        engine, champ, cfg, as_of=as_of, frame=frame, drift_runner=fake_runner(set())
    )
    err = d.checks[0]
    assert err.fired and err.value > 3.5
    assert d.triggered and any("MAPE" in r for r in d.reasons)


def test_official_trigger_fires_when_benchmark_is_much_better(engine, cfg) -> None:
    load = seed_db(engine, days=60, model_noise=1_500, official_noise=300)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine,
        champ,
        cfg,
        as_of=load["timestamp_utc"].max(),
        frame=frame,
        drift_runner=fake_runner(set()),
    )
    off = d.checks[1]
    assert off.fired and off.value > 15
    assert not d.checks[0].fired  # 1.5 GW noise ~ 2.5% MAPE, under the error threshold


def test_drift_trigger_fires_and_drift_errors_are_contained(engine, cfg) -> None:
    load = seed_db(engine, days=120, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine,
        champ,
        cfg,
        as_of=load["timestamp_utc"].max(),
        frame=frame,
        drift_runner=fake_runner(
            {"load_lag_24", "load_lag_48", "load_lag_168", "temperature_2m_de_avg"}
        ),
    )
    assert d.checks[2].fired and d.triggered and d.drift is not None

    def boom(*a, **k):
        raise RuntimeError("evidently exploded")

    d = retrain.evaluate_triggers(
        engine, champ, cfg, as_of=load["timestamp_utc"].max(), frame=frame, drift_runner=boom
    )
    assert not d.checks[2].fired and "error: evidently exploded" in d.checks[2].detail
    assert d.drift is None and d.drift_error == "evidently exploded"


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
def test_challenger_skipped_without_fresh_data(engine, cfg, monkeypatch) -> None:
    load = seed_db(engine, days=60, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    monkeypatch.setattr(retrain.registry, "training_window", lambda m: (None, frame.index.max()))
    out = retrain.run_champion_challenger(engine, champ, cfg, frame=frame)
    assert out.decision == "skipped" and "need 72h" in out.reason
    assert out.champion_mae is None


def test_challenger_skipped_without_train_end(engine, cfg, monkeypatch) -> None:
    load = seed_db(engine, days=60, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    monkeypatch.setattr(retrain.registry, "training_window", lambda m: (None, None))
    assert retrain.run_champion_challenger(engine, champ, cfg, frame=frame).decision == "skipped"


def test_challenger_promoted_when_it_beats_champion_by_margin(engine, cfg, monkeypatch) -> None:
    load = seed_db(engine, days=90, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD), bias=3_000)  # a weak champion
    train_end = frame.index.max() - pd.Timedelta(days=5)
    monkeypatch.setattr(retrain.registry, "training_window", lambda m: (None, train_end))

    dry = retrain.run_champion_challenger(
        engine, champ, cfg, frame=frame, dry_run=True, triggers=["x"]
    )
    assert dry.decision == "promoted (dry-run)"
    assert dry.holdout_start > train_end and dry.holdout_hours == 24 * 5
    assert dry.challenger_mae < dry.champion_mae and dry.improvement_pct >= cfg.min_improvement_pct
    assert dry.challenger_version is None and dry.triggers == ["x"]

    calls: dict[str, object] = {}
    monkeypatch.setattr(
        retrain.train, "log_training_run", lambda *a, **k: calls.setdefault("run", "run-new")
    )
    monkeypatch.setattr(
        retrain.registry,
        "register_run_model",
        lambda run_id: calls.setdefault("mv", type("MV", (), {"version": "9"})()),
    )
    monkeypatch.setattr(
        retrain.registry, "promote", lambda v, engine=None: calls.setdefault("promoted", v)
    )
    out = retrain.run_champion_challenger(engine, champ, cfg, frame=frame)
    assert out.decision == "promoted" and out.challenger_version == "9"
    assert out.training_run_id == "run-new" and calls["promoted"] == "9"
    assert "promoted challenger as v9" in out.summary()


def test_challenger_kept_when_margin_not_met(engine, cfg, monkeypatch) -> None:
    """The anti-churn margin: a champion as good as the candidate stays."""
    load = seed_db(engine, days=90, with_model=False)
    frame = feature_frame(load)
    cols = select_features(frame.columns, DAY_AHEAD)
    train_end = frame.index.max() - pd.Timedelta(days=5)
    monkeypatch.setattr(retrain.registry, "training_window", lambda m: (None, train_end))

    # Champion = the very same LightGBM the challenger would be -> ~0% improvement.
    from src.models.train import LightGBMForecaster

    fit = frame[frame.index <= train_end]
    twin = LightGBMForecaster(FAST_LGBM).fit(fit[cols], fit[TARGET])

    class TwinPyfunc:
        def predict(self, X):
            return twin.predict(X)

    champ = LoadedModel(
        "germany-load-day-ahead", "1", "champion", "r", cols, DAY_AHEAD, TwinPyfunc()
    )
    strict = retrain.RetrainConfig(
        min_holdout_hours=24 * 3, holdout_days=5, lgbm_params=FAST_LGBM, min_improvement_pct=50.0
    )
    out = retrain.run_champion_challenger(engine, champ, strict, frame=frame)
    assert out.decision == "kept" and "< required 50.0% margin" in out.reason
    assert out.challenger_version is None


# --------------------------------------------------------------------------- #
# Logging + CLI
# --------------------------------------------------------------------------- #
def test_decisions_are_logged_to_mlflow_and_db(engine, cfg, tmp_path, monkeypatch) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    get_settings.cache_clear()
    load = seed_db(engine, days=60, model_noise=100, model_bias=4_000)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine,
        champ,
        cfg,
        as_of=load["timestamp_utc"].max(),
        frame=frame,
        drift_runner=fake_runner(set()),
    )
    out = retrain.ChallengerOutcome(
        "kept",
        "margin",
        "1",
        holdout_hours=100,
        champion_mae=1.0,
        challenger_mae=0.99,
        champion_mape=1.0,
        challenger_mape=0.99,
        improvement_pct=1.0,
        holdout_start=AS_OF,
        holdout_end=AS_OF,
    )

    run_id = retrain.log_decision(d, out, cfg, experiment="test-retrain")
    run = mlflow.MlflowClient().get_run(run_id)
    assert run.data.tags["decision"] == "kept" and run.data.tags["triggered"] == "True"
    assert run.data.metrics["check_error_fired"] == 1.0
    assert run.data.metrics["promoted"] == 0.0
    assert run.data.params["cfg_min_improvement_pct"] == "3.0"
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run_id)}
    assert {"trigger_decision.json", "challenger_outcome.json"} <= artifacts

    retrain.record_event(engine, d, out)
    retrain.record_event(engine, d, None)
    events = db.read_monitoring_events(engine)
    assert len(events) == 2
    assert list(events["kind"]) == ["check", "retrain"]  # newest first, id breaks ties
    retrain_row = events[events["kind"] == "retrain"].iloc[0]
    assert retrain_row["decision"] == "kept" and bool(retrain_row["triggered"])
    assert events["model_mape_7d"].notna().all()
    get_settings.cache_clear()


def test_cli_check_only_and_force_dry_run(engine, cfg, monkeypatch, capsys) -> None:
    load = seed_db(engine, days=90, with_model=False)
    frame = feature_frame(load)
    champ = fake_champion(select_features(frame.columns, DAY_AHEAD), bias=3_000)
    monkeypatch.setattr(retrain.registry, "load_champion", lambda engine: champ)
    monkeypatch.setattr(
        retrain.registry,
        "training_window",
        lambda m: (None, frame.index.max() - pd.Timedelta(days=10)),
    )
    monkeypatch.setattr(retrain, "build_features", lambda *a, **k: frame)
    monkeypatch.setattr(retrain.drift_mod, "run_evidently", fake_runner(set()))
    url = str(engine.url)

    assert retrain.main(["--check-only", "--no-mlflow", "--database-url", url]) == 0
    out = capsys.readouterr().out
    assert "RETRAIN not triggered" in out and "Champion/challenger" not in out
    assert db.read_monitoring_events(engine).iloc[0]["kind"] == "check"

    assert retrain.main(["--no-mlflow", "--database-url", url]) == 0
    assert "No trigger fired" in capsys.readouterr().out

    assert (
        retrain.main(
            [
                "--force",
                "--dry-run",
                "--no-mlflow",
                "--database-url",
                url,
                "--holdout-days",
                "7",
                "--as-of",
                str(frame.index.max()),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "PROMOTED (DRY-RUN)" in out
    events = db.read_monitoring_events(engine)
    assert (
        events.iloc[0]["kind"] == "retrain" and events.iloc[0]["decision"] == "promoted (dry-run)"
    )
