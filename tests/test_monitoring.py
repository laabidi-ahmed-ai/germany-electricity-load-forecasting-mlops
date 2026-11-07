"""Monitoring tests (offline): performance head-to-head, drift summaries, triggers, champion/challenger."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from config.settings import get_settings
from src.data import db
from src.features.build_features import TARGET, WEATHER_COLUMNS, build_feature_frame
from src.features.horizons import D_PLUS_1, DAY_AHEAD, select_features
from src.monitoring import drift as drift_mod
from src.monitoring import metrics, retrain
from src.monitoring import performance as perf
from tests.conftest import FAST_LGBM, fake_loaded_model, load_curve, random_weather

N_DAYS = 40
START = pd.Timestamp("2024-01-01T00:00Z")
AS_OF = START + pd.Timedelta(days=N_DAYS) - pd.Timedelta(hours=1)  # last hour of the data
MODEL_NAME = "germany-load-day-ahead"
# The daily job issues at 05:40 UTC; in winter a D+1 forecast is due at 11:00 UTC.
ISSUE_LEAD = pd.Timedelta(hours=5, minutes=20)


# --- Helpers ---
def issue(
    hours: pd.Series | pd.DatetimeIndex,
    forecast_mw: np.ndarray | float,
    *,
    version: str = "1",
    issued_at: pd.Series | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """D+1 forecast rows; by default each Berlin day issued at 05:40 UTC the day before."""
    hours = pd.Series(hours)
    if issued_at is None:
        issued_at = D_PLUS_1.issue_deadline(hours) - ISSUE_LEAD
    return pd.DataFrame(
        {
            "timestamp_utc": hours,
            "model_version": version,
            "model_name": MODEL_NAME,
            "horizon": D_PLUS_1.name,
            "forecast_mw": forecast_mw,
            "issued_at": issued_at,
        }
    )


def local_day(day: str) -> pd.DatetimeIndex:
    """The UTC hours of Berlin calendar ``day`` (winter: 23:00 UTC the evening before on)."""
    return pd.date_range(
        pd.Timestamp(day).tz_localize("Europe/Berlin"), periods=24, freq="h"
    ).tz_convert("UTC")


def delete_forecasts(engine, hours: pd.DatetimeIndex) -> None:
    from sqlalchemy import delete

    with engine.begin() as conn:
        conn.execute(
            delete(db.LoadForecastModel).where(
                db.LoadForecastModel.timestamp_utc.in_(list(hours.to_pydatetime()))
            )
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
    load = load_curve(len(ts))
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
    if with_model:  # one on-time D+1 issue per day, as the daily job makes them
        db.upsert_dataframe(
            engine,
            db.LoadForecastModel,
            issue(
                ts,
                load + model_bias + rng.normal(0, model_noise, len(ts)),
                version=model_version,
            ),
        )
    return pd.DataFrame({"timestamp_utc": ts, TARGET: load})


def feature_frame(load: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    return build_feature_frame(load, random_weather(load["timestamp_utc"], seed=seed))


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


def align(engine, **kwargs) -> pd.DataFrame:
    """Actual x official x model forecast over everything stored."""
    return metrics.with_model_forecast(engine, metrics.actual_vs_official(engine), **kwargs)


# --- Performance ---
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
    a = align(engine)
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


def test_the_latest_on_time_issue_wins_and_a_late_one_never_does(engine) -> None:
    seed_db(engine)
    ts = START + pd.Timedelta(days=3)  # 01:00 Berlin on 4 Jan: due 3 Jan 11:00 UTC
    due = D_PLUS_1.issue_deadline(ts)
    assert due == pd.Timestamp("2024-01-03T11:00Z")
    on_time = issue([ts], 77_777.0, version="2", issued_at=due)  # at the deadline still counts
    late = issue([ts], 99_999.0, version="2", issued_at=due + pd.Timedelta(minutes=1))
    db.upsert_dataframe(engine, db.LoadForecastModel, on_time)
    db.upsert_dataframe(engine, db.LoadForecastModel, late)

    stored = db.read_table(engine, db.LoadForecastModel, start=ts, end=ts)
    assert len(stored) == 3  # every issue is kept: v1's, and both re-issues
    a = align(engine)
    row = a[a["timestamp_utc"] == ts].iloc[0]
    assert row["model_mw"] == 77_777.0 and row["model_version"] == "2"
    assert len(a) == 24 * N_DAYS  # still one row per hour
    only_v1 = align(engine, model_version="1")
    assert only_v1[only_v1["timestamp_utc"] == ts]["model_mw"].iloc[0] not in (77_777.0, 99_999.0)


def test_on_time_late_and_missing_hours_are_counted_across_versions(engine) -> None:
    seed_db(engine)
    late_day, missing_day, v2_day = (
        local_day("2024-01-20"),
        local_day("2024-01-22"),
        local_day("2024-01-25"),
    )
    for day in (late_day, missing_day, v2_day):
        delete_forecasts(engine, day)
    late_at = D_PLUS_1.issue_deadline(late_day[0]) + pd.Timedelta(hours=2)  # a late backup run
    db.upsert_dataframe(engine, db.LoadForecastModel, issue(late_day, 50_000.0, issued_at=late_at))
    db.upsert_dataframe(engine, db.LoadForecastModel, issue(v2_day, 50_000.0, version="2"))

    status = metrics.forecast_status(engine, metrics.actual_vs_official(engine)).set_index(
        "timestamp_utc"
    )["status"]
    assert (status[late_day] == metrics.LATE).all()
    assert (status[missing_day] == metrics.MISSING).all()
    assert (status[v2_day] == metrics.ON_TIME).all()

    report = perf.compute_report(engine, as_of=AS_OF, model_version="1")
    w30 = report.window(30)  # (10 Jan 23:00, 9 Feb 23:00]: the three days are inside
    assert w30.n_hours == 24 * 30 - 3 * 24  # v1 is scored on its own on-time hours only
    assert (w30.n_on_time, w30.n_late, w30.n_missing) == (24 * 30 - 48, 24, 24)
    assert w30.on_time_pct == pytest.approx(100 * (24 * 30 - 48) / (24 * 30))
    w7 = report.window(7)
    assert (w7.n_late, w7.n_missing, w7.on_time_pct) == (0, 0, 100.0)
    assert "on time 93% (24 late, 24 missing)" in report.summary()
    assert report.to_dict()["windows"][1]["on_time_pct"] == w30.on_time_pct
    aligned = align(engine)
    assert not aligned["timestamp_utc"].isin(late_day).any()  # a late-only hour is not scored


def test_hours_before_the_first_forecast_are_not_missing(engine) -> None:
    seed_db(engine)
    delete_forecasts(engine, pd.date_range(START, AS_OF - pd.Timedelta(days=10), freq="h"))
    w30 = perf.compute_report(engine, as_of=AS_OF).window(30)
    assert w30.n_hours == w30.n_on_time == 24 * 10  # the forecasts start 31 Jan 00:00 UTC
    assert w30.n_missing == 0 and w30.on_time_pct == 100.0


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
        assert (w.n_on_time, w.n_late, w.n_missing, w.on_time_pct) == (24 * days, 0, 0, 100.0)
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
    assert all(w.on_time_pct is None and w.n_missing == 0 for w in report.windows)  # not live
    assert report.official_only[7]["n_hours"] == 168  # the benchmark alone is still scored
    assert "no scorable model forecasts yet" in report.summary()


def test_report_reads_each_table_once(engine, monkeypatch) -> None:
    """Actual and official load are read once and shared by every window and score."""
    seed_db(engine)
    reads: list[str] = []
    original = db.read_table

    def counting(eng, model, **kwargs):
        reads.append(model.__tablename__)
        return original(eng, model, **kwargs)

    monkeypatch.setattr(db, "read_table", counting)
    report = perf.compute_report(engine, as_of=AS_OF)
    assert report.has_data and report.official_only
    assert sorted(reads) == ["load_actual", "load_forecast_model", "load_forecast_official"]


def test_window_needs_minimum_hours(engine) -> None:
    seed_db(engine, days=2)
    as_of = START + pd.Timedelta(hours=10)
    w = perf.window_metrics(align(engine), as_of, 7)
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


def test_performance_cli_defaults_to_the_champion_horizon(monkeypatch) -> None:
    """No database: the engine, report and backtest are stubbed; only the parsed horizon matters."""
    seen = {}

    class _Stub:
        daily = pd.DataFrame()

        def summary(self) -> str:
            return ""

    def fake_backtest(engine, *, horizon, **kwargs):
        seen["horizon"] = horizon.name
        return _Stub()

    monkeypatch.setattr(perf.db, "get_engine", lambda url=None: None)
    monkeypatch.setattr(perf, "compute_report", lambda *a, **k: _Stub())
    monkeypatch.setattr(perf, "backtest_vs_official", fake_backtest)
    assert perf.main(["--backtest-days", "1"]) == 0
    assert seen["horizon"] == "d_plus_1"


def test_performance_cli_fair_weather_runs_the_fair_backtest(monkeypatch, capsys) -> None:
    """Stubbed like the test above: --fair-weather replaces the measured-only backtest."""
    from src.monitoring import fair_backtest

    seen = {}

    class _Stub:
        def summary(self) -> str:
            return "fair summary"

        def monthly(self) -> pd.DataFrame:
            return pd.DataFrame({"month": ["2026-01"], "hours": [744]})

    def fake_fair(engine, *, days, horizon, progress, **kwargs):
        seen.update(days=days, horizon=horizon.name)
        progress("measured row")
        return _Stub()

    def not_this_one(*args, **kwargs):
        raise AssertionError("--fair-weather must not run the measured-only backtest")

    monkeypatch.setattr(perf.db, "get_engine", lambda url=None: None)

    class _Live:
        def summary(self) -> str:
            return "live report"

    monkeypatch.setattr(perf, "compute_report", lambda *a, **k: _Live())
    monkeypatch.setattr(perf, "backtest_vs_official", not_this_one)
    monkeypatch.setattr(fair_backtest, "fair_backtest_vs_official", fake_fair)
    assert perf.main(["--backtest-days", "365", "--fair-weather"]) == 0
    assert seen == {"days": 365, "horizon": "d_plus_1"}
    out = capsys.readouterr().out
    order = [out.index(s) for s in ("live report", "measured row", "fair summary", "Per month")]
    assert order == sorted(order)


@pytest.mark.parametrize(
    "argv",
    [["--fair-weather"], ["--fair-weather", "--backtest-days", "7", "--horizon", "day_ahead"]],
)
def test_performance_cli_refuses_fair_weather_without_a_d_plus_1_backtest(
    argv, monkeypatch
) -> None:
    def no_database(url=None):
        raise AssertionError("arguments are checked before any database is opened")

    monkeypatch.setattr(perf.db, "get_engine", no_database)
    with pytest.raises(SystemExit):
        perf.main(argv)


# --- Drift ---
def test_drift_columns_exclude_calendar_and_nowcast_features() -> None:
    frame = feature_frame(
        pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(START, periods=24 * 10, freq="h", tz="UTC"),
                TARGET: load_curve(240),
            }
        )
    )
    cols = drift_mod.drift_columns(list(frame.columns))
    assert cols == ["load_lag_24", "load_lag_48", "load_lag_168", *WEATHER_COLUMNS]
    assert not any(c in cols for c in ("hour", "day_of_week", "month", "is_holiday", "load_lag_1"))
    # A D+1 champion is monitored on what it actually uses.
    d1 = drift_mod.drift_columns(list(frame.columns), D_PLUS_1)
    assert d1 == ["load_lag_48", "load_lag_168", *WEATHER_COLUMNS]


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


def test_seasonal_reference_current_window_spanning_new_year() -> None:
    """Dec 11 -> Jan 10 (lo=338 > hi=17 before the fix) must select Dec + Jan, not nothing."""
    idx = pd.date_range("2021-01-01", "2025-01-10T23:00", freq="h", tz="UTC")
    training = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    current = pd.date_range("2024-12-11", "2025-01-10T23:00", freq="h", tz="UTC")
    ref = drift_mod.seasonal_reference(training, current, pad_days=7)
    assert len(ref) > 0
    assert set(ref.index.month) == {12, 1}
    assert (ref.index < current.min()).all()  # past years only (+ the pad before the window)
    assert {2021, 2022, 2023} <= set(ref.index[ref.index.month == 12].year)
    assert {2021, 2022, 2023, 2024} <= set(ref.index[ref.index.month == 1].year)
    assert not ((ref.index.month == 12) & (ref.index.day < 3)).any()  # Dec 11 - 7 d (+ leap)
    assert not ((ref.index.month == 1) & (ref.index.day > 17)).any()  # Jan 10 + 7 d
    # a window plus pad covering a whole year keeps every past row
    long = pd.date_range("2023-06-01", "2024-05-31T23:00", freq="h", tz="UTC")
    full = drift_mod.seasonal_reference(training, long, pad_days=7)
    assert len(full) == (training.index < long.min()).sum()


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
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
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


def test_drift_reference_defaults_to_the_champions_training_window(engine) -> None:
    load = seed_db(engine, days=120, with_model=False)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
    champ.train_end = frame.index.max() - pd.Timedelta(days=40)
    seen: dict[str, pd.DataFrame] = {}

    def spy(reference, current, columns, threshold=0.25):
        seen["reference"] = reference
        return fake_runner(set())(reference, current, columns, threshold)

    r = drift_mod.compute_drift(engine, champ, window_days=30, frame=frame, runner=spy)
    assert seen["reference"].index.max() <= champ.train_end
    assert r.reference_rows == len(seen["reference"])
    assert not any("train_end" in n for n in r.notes)

    champ.train_end = None  # nothing recorded -> whole frame, and say so
    r = drift_mod.compute_drift(engine, champ, window_days=30, frame=frame, runner=spy)
    assert any("no recorded train_end" in n for n in r.notes)


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


# --- Triggers ---
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
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine, champ, cfg, as_of=AS_OF, frame=frame, drift_runner=fake_runner(set())
    )
    assert not d.triggered
    assert [c.name for c in d.checks] == ["error", "official", "drift"]
    assert "insufficient data" in d.checks[0].detail and "insufficient data" in d.checks[1].detail
    assert "not triggered" in d.summary()


def _error_check(engine, cfg, **seed_kwargs) -> tuple[retrain.TriggerDecision, retrain.Check]:
    load = seed_db(engine, days=60, **seed_kwargs)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine,
        champ,
        cfg,
        as_of=load["timestamp_utc"].max(),
        frame=frame,
        drift_runner=fake_runner(set()),
    )
    return d, d.checks[0]


def test_error_trigger_fires_when_official_is_better(engine, cfg) -> None:
    d, err = _error_check(engine, cfg, model_noise=100, model_bias=3_000)  # ~5.6% vs ~1.5%
    w = d.performance.window(cfg.error_window_days)
    assert err.value > err.threshold and w.official_mape < w.model_mape
    assert err.fired and "(official as good or better)" in err.detail
    assert d.triggered and any("MAPE" in r for r in d.reasons)


def test_error_trigger_holds_when_official_is_worse_on_the_same_hours(engine, cfg) -> None:
    """A week that is hard for both forecasts is weather, not model decay."""
    d, err = _error_check(engine, cfg, model_noise=100, model_bias=3_000, official_noise=7_000)
    w = d.performance.window(cfg.error_window_days)
    assert err.threshold < err.value < cfg.error_ceiling_factor * err.threshold
    assert w.official_mape > w.model_mape
    assert not err.fired and "(model still ahead)" in err.detail
    assert f"official {w.official_mape:.2f}% on the same hours" in err.detail
    assert not d.triggered


def test_error_trigger_fires_over_twice_the_limit_even_if_official_is_worse(engine, cfg) -> None:
    """The backstop: a broken model in a week that is hard for both still fires."""
    d, err = _error_check(engine, cfg, model_noise=100, model_bias=6_000, official_noise=12_000)
    w = d.performance.window(cfg.error_window_days)
    assert err.value > cfg.error_ceiling_factor * err.threshold
    assert w.official_mape > w.model_mape  # the official forecast is even worse
    assert err.fired and "(over twice the limit)" in err.detail
    assert d.triggered


def test_error_trigger_does_not_fire_below_the_limit(engine, cfg) -> None:
    d, err = _error_check(engine, cfg)  # model ~0.7% MAPE
    assert err.value < err.threshold
    assert not err.fired and "(under the limit)" in err.detail
    assert not d.triggered


def test_error_threshold_follows_the_champions_horizon() -> None:
    """A harder task has a higher normal error, so D+1 gets its own limit."""
    default = retrain.RetrainConfig()
    assert retrain.mape_threshold(default, "day_ahead") == 3.5
    assert retrain.mape_threshold(default, "d_plus_1") == 4.0
    assert retrain.mape_threshold(default, "unknown") == 3.5
    explicit = retrain.RetrainConfig(mape_threshold_pct=5.0)  # a CLI override always wins
    assert retrain.mape_threshold(explicit, "d_plus_1") == 5.0


def test_official_trigger_fires_when_benchmark_is_much_better(engine, cfg) -> None:
    load = seed_db(engine, days=60, model_noise=1_500, official_noise=300)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
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


def test_official_detail_states_the_hours_it_covers(engine, cfg) -> None:
    """The 30-day label alone hides how few champion hours were scored."""
    load = seed_db(engine, days=60, model_noise=1_500, official_noise=300)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
    d = retrain.evaluate_triggers(
        engine,
        champ,
        cfg,
        as_of=load["timestamp_utc"].max(),
        frame=frame,
        drift_runner=fake_runner(set()),
    )
    m = re.search(r"MW over (\d+)h \((official|model) better by", d.checks[1].detail)
    assert m is not None, d.checks[1].detail
    n_hours = d.performance.window(cfg.official_window_days).n_hours
    assert int(m.group(1)) == n_hours > 0


def test_drift_trigger_fires_and_drift_errors_are_contained(engine, cfg) -> None:
    load = seed_db(engine, days=120, with_model=False)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
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


def test_empty_drift_result_is_unavailable_not_no_drift(engine, cfg) -> None:
    """A runner that scores no monitored column must not read as a clean 0/0 report."""
    load = seed_db(engine, days=120, with_model=False)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))

    def empty(*a, **k):
        return []

    with pytest.raises(RuntimeError, match="no result"):
        drift_mod.compute_drift(engine, champ, window_days=30, frame=frame, runner=empty)
    d = retrain.evaluate_triggers(
        engine, champ, cfg, as_of=load["timestamp_utc"].max(), frame=frame, drift_runner=empty
    )
    assert d.drift is None and "no result" in d.drift_error  # -> drift.unavailable = 1
    drift_check = next(c for c in d.checks if c.name == "drift")
    assert not drift_check.fired and drift_check.detail.startswith("error:")


# --- Champion / challenger ---
def test_challenger_skipped_without_fresh_data(engine, cfg) -> None:
    load = seed_db(engine, days=60, with_model=False)
    frame = feature_frame(load)
    champ = fake_loaded_model(
        select_features(frame.columns, DAY_AHEAD), train_end=frame.index.max()
    )
    out = retrain.run_champion_challenger(engine, champ, cfg, frame=frame)
    assert out.decision == "skipped" and "need 72h" in out.reason
    assert out.champion_mae is None


def test_challenger_skipped_without_train_end(engine, cfg) -> None:
    load = seed_db(engine, days=60, with_model=False)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))  # no train_end recorded
    assert retrain.run_champion_challenger(engine, champ, cfg, frame=frame).decision == "skipped"


def test_challenger_promoted_when_it_beats_champion_by_margin(engine, cfg, monkeypatch) -> None:
    load = seed_db(engine, days=90, with_model=False)
    frame = feature_frame(load)
    train_end = frame.index.max() - pd.Timedelta(days=5)
    champ = fake_loaded_model(  # a weak champion
        select_features(frame.columns, DAY_AHEAD), bias=3_000, train_end=train_end
    )

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


def test_challenger_kept_when_margin_not_met(engine, cfg) -> None:
    """The anti-churn margin: a champion as good as the candidate stays."""
    load = seed_db(engine, days=90, with_model=False)
    frame = feature_frame(load)
    cols = select_features(frame.columns, DAY_AHEAD)
    train_end = frame.index.max() - pd.Timedelta(days=5)

    # Champion = the very same LightGBM the challenger would be -> ~0% improvement.
    from src.models.train import LightGBMForecaster

    fit = frame[frame.index <= train_end]
    twin = LightGBMForecaster(FAST_LGBM).fit(fit[cols], fit[TARGET])

    class TwinPyfunc:
        def predict(self, X):
            return twin.predict(X)

    champ = fake_loaded_model(cols, pyfunc=TwinPyfunc(), train_end=train_end)
    strict = retrain.RetrainConfig(
        min_holdout_hours=24 * 3, holdout_days=5, lgbm_params=FAST_LGBM, min_improvement_pct=50.0
    )
    out = retrain.run_champion_challenger(engine, champ, strict, frame=frame)
    assert out.decision == "kept" and "< required 50.0% margin" in out.reason
    assert out.challenger_version is None


def test_challenger_competes_on_the_champions_horizon(engine, cfg, monkeypatch) -> None:
    """A D+1 champion is challenged by a D+1 model - never by one that sees load_lag_24."""
    load = seed_db(engine, days=90, with_model=False)
    frame = feature_frame(load)
    cols = select_features(frame.columns, D_PLUS_1)
    champ = fake_loaded_model(
        cols, horizon=D_PLUS_1, train_end=frame.index.max() - pd.Timedelta(days=10)
    )
    seen: list[str] = []
    real_fit = retrain.train.fit_model

    def spy_fit(df, horizon, params=None):
        seen.append(horizon.name)
        return real_fit(df, horizon, params)

    monkeypatch.setattr(retrain.train, "fit_model", spy_fit)
    out = retrain.run_champion_challenger(engine, champ, cfg, frame=frame)
    assert seen and set(seen) == {"d_plus_1"}
    assert out.decision in {"kept", "promoted (dry-run)", "promoted"}


# --- Logging + CLI ---
def test_decisions_are_logged_to_mlflow_and_db(engine, cfg, tmp_path, monkeypatch) -> None:
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    get_settings.cache_clear()
    load = seed_db(engine, days=60, model_noise=100, model_bias=4_000)
    frame = feature_frame(load)
    champ = fake_loaded_model(select_features(frame.columns, DAY_AHEAD))
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

    retrain.record_event(engine, d, out, cfg)
    retrain.record_event(engine, d, None, cfg)
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
    champ = fake_loaded_model(
        select_features(frame.columns, DAY_AHEAD),
        bias=3_000,
        train_end=frame.index.max() - pd.Timedelta(days=10),
    )
    monkeypatch.setattr(retrain.registry, "load_champion", lambda engine: champ)
    monkeypatch.setattr(retrain, "build_features", lambda *a, **k: frame)
    monkeypatch.setattr(retrain.drift_mod, "run_evidently", fake_runner(set()))
    url = str(engine.url)

    # --once-per-day only skips when a pass was already recorded today (none yet).
    assert (
        retrain.main(["--check-only", "--once-per-day", "--no-mlflow", "--database-url", url]) == 0
    )
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

    # An automatic (backup / chained) run on the same UTC day records nothing new.
    assert retrain.main(["--once-per-day", "--no-mlflow", "--database-url", url]) == 0
    assert "already ran today" in capsys.readouterr().out
    assert len(db.read_monitoring_events(engine)) == len(events)
