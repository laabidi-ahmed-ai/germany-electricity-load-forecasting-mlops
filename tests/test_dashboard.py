"""Dashboard: read-only queries, metrics, empty-database behaviour and a headless render."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.engine import Engine

from config.settings import PROJECT_ROOT
from dashboard import queries as q
from src.data import db
from src.models.evaluate import compute_metrics

START = pd.Timestamp("2024-01-01T00:00Z")
DAYS = 10
RETRAIN_DETAILS = {
    "decision": {
        "as_of": "2024-01-11T00:00:00+00:00",
        "champion_version": "2",
        "triggered": True,
        "reasons": ["drift"],
        "checks": [
            {"name": "error", "fired": False, "value": 2.1, "threshold": 3.5, "detail": "ok"},
            {"name": "official", "fired": False, "value": -4.0, "threshold": 15.0, "detail": "ok"},
            {
                "name": "drift",
                "fired": True,
                "value": 0.5,
                "threshold": 0.3,
                "detail": "4/8 drifted",
            },
        ],
        "drift": {
            "n_monitored": 2,
            "n_drifted": 1,
            "share_drifted": 0.5,
            "drift_share_threshold": 0.3,
            "dataset_drift": True,
            "prediction_drift": False,
            "target_drift": False,
            "columns": [
                {
                    "column": "load_lag_24",
                    "score": 0.4,
                    "threshold": 0.25,
                    "drifted": True,
                    "method": "w",
                },
                {
                    "column": "temp",
                    "score": 0.1,
                    "threshold": 0.25,
                    "drifted": False,
                    "method": "w",
                },
            ],
            "prediction": {
                "column": "prediction",
                "score": 0.05,
                "threshold": 0.25,
                "drifted": False,
                "method": "w",
            },
            "target": None,
        },
        "drift_error": None,
    },
    "outcome": None,
}


def synthetic_load(n: int) -> np.ndarray:
    hours = np.arange(n)
    return 55_000 + 8_000 * np.sin((hours - 6) / 24 * 2 * np.pi)


@pytest.fixture
def seeded(engine: Engine) -> Engine:
    """Ten days of actuals + official forecast, two forecast issues, two model versions, events."""
    ts = pd.date_range(START, periods=24 * DAYS, freq="h", tz="UTC")
    load = synthetic_load(len(ts))
    rng = np.random.default_rng(0)
    db.upsert_dataframe(
        engine,
        db.LoadActual,
        pd.DataFrame({"timestamp_utc": ts, "load_mw": load, "source": "smard"}),
    )
    # Official forecast covers one extra day beyond the actuals (published day-ahead).
    ts_off = pd.date_range(START, periods=24 * (DAYS + 1), freq="h", tz="UTC")
    db.upsert_dataframe(
        engine,
        db.LoadForecastOfficial,
        pd.DataFrame(
            {
                "timestamp_utc": ts_off,
                "forecast_mw": synthetic_load(len(ts_off)) + rng.normal(0, 1_000, len(ts_off)),
                "source": "smard",
            }
        ),
    )
    # Model forecasts: v1 on days 3-9 (issued early), then v2 re-issues the last 2 days
    # (later issue wins) plus the not-yet-observed day 10.
    ts_v1 = ts[24 * 3 :]
    db.upsert_dataframe(
        engine,
        db.LoadForecastModel,
        pd.DataFrame(
            {
                "timestamp_utc": ts_v1,
                "model_version": "1",
                "model_name": q.MODEL_NAME,
                "forecast_mw": load[24 * 3 :] + 3_000,  # deliberately worse than the official
                "issued_at": datetime(2024, 1, 3, 5, tzinfo=UTC),
            }
        ),
    )
    ts_v2 = pd.date_range(START + pd.Timedelta(days=DAYS - 2), periods=24 * 3, freq="h", tz="UTC")
    db.upsert_dataframe(
        engine,
        db.LoadForecastModel,
        pd.DataFrame(
            {
                "timestamp_utc": ts_v2,
                "model_version": "2",
                "model_name": q.MODEL_NAME,
                "forecast_mw": synthetic_load(len(ts_v2)) + rng.normal(0, 300, len(ts_v2)),
                "issued_at": datetime(2024, 1, 10, 5, tzinfo=UTC),
            }
        ),
    )
    for version, cv_mape in (("1", 3.1), ("2", 2.4)):
        db.store_model_artifact(
            engine,
            name=q.MODEL_NAME,
            version=version,
            horizon="day_ahead",
            features="[]",
            train_start=START.to_pydatetime(),
            train_end=(START + pd.Timedelta(days=5)).to_pydatetime(),
            metrics=json.dumps({"lightgbm_mape_mean": cv_mape, "lightgbm_mae_mean": 1000.0}),
            bundle=b"zip",
            bundle_sha256="0" * 64,
            bundle_bytes=3,
            created_at=datetime(2024, 1, int(version) * 3, tzinfo=UTC),
        )
    db.set_champion(engine, q.MODEL_NAME, "2")
    db.insert_monitoring_event(
        engine,
        created_at=datetime(2024, 1, 10, 6, tzinfo=UTC),
        as_of=datetime(2024, 1, 10, tzinfo=UTC),
        kind="check",
        triggered=False,
        model_version="1",
        model_mape_7d=2.5,
        official_mape_7d=2.8,
        drift_share=0.0,
        details=json.dumps({"decision": {"checks": [], "drift": None}, "outcome": None}),
    )
    db.insert_monitoring_event(
        engine,
        created_at=datetime(2024, 1, 11, 6, tzinfo=UTC),
        as_of=datetime(2024, 1, 11, tzinfo=UTC),
        kind="check",
        triggered=True,
        reason="drift",
        model_version="2",
        drift_share=0.5,
        details=json.dumps(RETRAIN_DETAILS),
    )
    db.insert_monitoring_event(
        engine,
        created_at=datetime(2024, 1, 11, 6, 30, tzinfo=UTC),
        as_of=datetime(2024, 1, 11, tzinfo=UTC),
        kind="retrain",
        triggered=True,
        decision="promoted",
        reason="challenger MAE 12% lower",
        model_version="2",
        details=json.dumps(
            {"decision": RETRAIN_DETAILS["decision"], "outcome": {"decision": "promoted"}}
        ),
    )
    return engine


# --- Headline ---
def test_champion_and_versions(seeded: Engine) -> None:
    champ = q.champion(seeded)
    assert champ is not None
    assert champ["version"] == "2" and champ["is_champion"] and champ["n_versions"] == 2
    assert champ["metrics"]["lightgbm_mape_mean"] == 2.4
    assert champ["promoted_at"] is not None

    versions = q.model_versions(seeded)
    assert list(versions["version"]) == ["2", "1"]  # newest first
    assert list(versions["cv_mape"]) == [2.4, 3.1]
    assert "metrics" not in versions.columns and "bundle" not in versions.columns


def test_coverage(seeded: Engine) -> None:
    cov = q.coverage(seeded)
    assert cov["first_actual"] == START
    assert cov["last_actual"] == START + pd.Timedelta(hours=24 * DAYS - 1)
    assert cov["n_hours"] == cov["expected_hours"] == 24 * DAYS
    assert cov["completeness_pct"] == 100.0
    assert cov["last_official"] == START + pd.Timedelta(hours=24 * (DAYS + 1) - 1)
    assert cov["last_model_forecast"] == START + pd.Timedelta(hours=24 * (DAYS + 1) - 1)


def test_coverage_counts_gaps(seeded: Engine) -> None:
    from sqlalchemy import delete

    with seeded.begin() as conn:
        conn.execute(
            delete(db.LoadActual).where(
                db.LoadActual.timestamp_utc == (START + pd.Timedelta(hours=50)).to_pydatetime()
            )
        )
    cov = q.coverage(seeded)
    assert cov["n_hours"] == 24 * DAYS - 1 and cov["expected_hours"] == 24 * DAYS
    assert cov["completeness_pct"] == pytest.approx(100 * (24 * DAYS - 1) / (24 * DAYS))


# --- Latest forecast ---
def test_latest_forecast_frame_is_the_latest_issue_with_context(seeded: Engine) -> None:
    frame = q.latest_forecast_frame(seeded, context_hours=48)
    assert len(frame) == 48 + 72  # context + the v2 issue (3 days)
    window = frame[frame["model_mw"].notna()]
    assert len(window) == 72 and set(window["model_version"]) == {"2"}
    assert frame["timestamp_utc"].is_monotonic_increasing
    # Actuals exist for the context and the first two forecast days, not the third.
    assert frame["actual_mw"].notna().sum() == 48 + 48
    assert frame["official_mw"].notna().all()
    assert frame["issued_at"].notna().all()


# --- Accuracy ---
def test_aligned_frame_scores_only_observed_hours_with_the_latest_issue(seeded: Engine) -> None:
    as_of = START + pd.Timedelta(days=DAYS)
    aligned = q.aligned_frame(seeded, days=30, as_of=as_of)
    # v1 covered days 3-9 (7 days), v2 re-issued days 8-9; day 10 has no actuals yet.
    assert len(aligned) == 24 * 7
    by_version = aligned.groupby("model_version").size().to_dict()
    assert by_version == {"1": 24 * 5, "2": 24 * 2}
    assert aligned["actual_mw"].notna().all() and aligned["official_mw"].notna().all()

    only_two_days = q.aligned_frame(seeded, days=2, as_of=as_of)
    assert set(only_two_days["model_version"]) == {"2"}


def test_window_summary_matches_the_training_metrics(seeded: Engine) -> None:
    as_of = START + pd.Timedelta(days=DAYS, hours=-1)  # "now" = the last observed hour
    aligned = q.aligned_frame(seeded, days=30, as_of=as_of)
    w = q.window_summary(aligned, days=7, as_of=as_of)
    assert w["judged"] and w["n_hours"] == 24 * 7
    ref = compute_metrics(aligned["actual_mw"], aligned["model_mw"])
    assert w["model_mae"] == pytest.approx(ref["mae"]) and w["model_mape"] == pytest.approx(
        ref["mape"]
    )
    ref_off = compute_metrics(aligned["actual_mw"], aligned["official_mw"])
    assert w["official_mape"] == pytest.approx(ref_off["mape"])
    assert w["improvement_pct"] == pytest.approx(
        (ref_off["mae"] - ref["mae"]) / ref_off["mae"] * 100
    )
    assert w["model_beats_official"] is False  # v1's +3 GW bias dominates the week

    w2 = q.window_summary(aligned, days=2, as_of=as_of)  # only the accurate v2 days
    assert w2["judged"] and w2["model_beats_official"] is True


def test_window_summary_refuses_to_judge_on_too_few_hours() -> None:
    small = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(START, periods=5, freq="h", tz="UTC"),
            "actual_mw": [50_000.0] * 5,
            "model_mw": [50_100.0] * 5,
            "official_mw": [49_000.0] * 5,
            "model_version": "1",
        }
    )
    as_of = START + pd.Timedelta(hours=4)
    w = q.window_summary(small, days=7, as_of=as_of)
    assert w == {"days": 7, "n_hours": 5, "judged": False}
    assert q.window_summary(small.iloc[0:0], days=7, as_of=as_of) == {
        "days": 7,
        "n_hours": 0,
        "judged": False,
    }


def test_daily_accuracy(seeded: Engine) -> None:
    aligned = q.aligned_frame(seeded, days=30, as_of=START + pd.Timedelta(days=DAYS))
    daily = q.daily_accuracy(aligned)
    assert len(daily) == 7 and (daily["n_hours"] == 24).all()
    assert daily["date"].is_monotonic_increasing
    # the last two days are v2 (accurate), the earlier five are v1 (biased by 3 GW)
    assert (daily["model_mae"].iloc[:5] > daily["official_mae"].iloc[:5]).all()
    assert (daily["model_mae"].iloc[5:] < daily["official_mae"].iloc[5:]).all()
    assert q.daily_accuracy(aligned.iloc[0:0]).empty


def test_official_only_summary_works_before_any_model_forecast(engine: Engine) -> None:
    ts = pd.date_range(START, periods=48, freq="h", tz="UTC")
    load = synthetic_load(48)
    db.upsert_dataframe(
        engine,
        db.LoadActual,
        pd.DataFrame({"timestamp_utc": ts, "load_mw": load, "source": "smard"}),
    )
    db.upsert_dataframe(
        engine,
        db.LoadForecastOfficial,
        pd.DataFrame({"timestamp_utc": ts, "forecast_mw": load * 1.02, "source": "smard"}),
    )
    s = q.official_only_summary(engine, days=30, as_of=ts[-1])
    assert s["n_hours"] == 48 and s["mape"] == pytest.approx(2.0)
    assert q.aligned_frame(engine, days=30, as_of=ts[-1]).empty


# --- Monitoring ---
def test_latest_check_parses_triggers_and_drift(seeded: Engine) -> None:
    events = q.monitoring_events(seeded)
    assert len(events) == 3 and events["created_at"].is_monotonic_decreasing
    check = q.latest_check(events)
    assert check is not None and check["triggered"] and check["model_version"] == "2"
    assert check["as_of"] == pd.Timestamp(
        "2024-01-11T00:00Z"
    )  # the latest *check*, not the retrain
    assert list(check["checks"]["name"]) == ["error", "official", "drift"]
    assert list(check["checks"]["fired"]) == [False, False, True]
    assert check["drift"]["share_drifted"] == 0.5
    # per-column table = monitored columns + the prediction column (target absent)
    assert list(check["drift_columns"]["column"]) == ["load_lag_24", "temp", "prediction"]
    assert list(check["drift_columns"]["drifted"]) == [True, False, False]


def test_event_timeline_status(seeded: Engine) -> None:
    timeline = q.event_timeline(q.monitoring_events(seeded))
    assert list(timeline["kind"]) == ["retrain", "check", "check"]
    assert list(timeline["status"]) == ["promoted", "triggered", "ok"]
    assert timeline.loc[2, "model_mape_7d"] == 2.5


def test_latest_check_survives_unparseable_details(engine: Engine) -> None:
    db.insert_monitoring_event(
        engine,
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        kind="check",
        triggered=False,
        details="{oops",
    )
    check = q.latest_check(q.monitoring_events(engine))
    assert check is not None and check["checks"].empty and check["drift"] is None


# --- Empty database + engine ---
def test_everything_handles_an_empty_database(engine: Engine) -> None:
    assert q.champion(engine) is None
    assert q.model_versions(engine).empty
    cov = q.coverage(engine)
    assert cov["first_actual"] is None and cov["n_hours"] == 0 and cov["completeness_pct"] is None
    assert q.latest_forecast_frame(engine).empty
    assert q.aligned_frame(engine, days=30).empty
    assert q.official_only_summary(engine, days=30)["mape"] is None
    events = q.monitoring_events(engine)
    assert events.empty and q.latest_check(events) is None and q.event_timeline(events).empty


def test_make_engine_opens_postgres_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_create_engine(url, **kwargs):
        captured["url"], captured["kwargs"] = url, kwargs
        return object()

    monkeypatch.setattr(db, "create_engine", fake_create_engine)
    q.make_engine("postgresql://u:p@host/dbname?sslmode=require")
    assert captured["url"].startswith("postgresql+psycopg://")
    assert captured["kwargs"]["connect_args"] == {"options": "-c default_transaction_read_only=on"}
    assert captured["kwargs"]["pool_pre_ping"] is True

    q.make_engine("sqlite:///:memory:")
    assert "connect_args" not in captured["kwargs"]


def test_describe_database_never_leaks_credentials() -> None:
    text = q.describe_database("postgresql://alice:s3cret@ep-x.neon.tech/neondb?sslmode=require")
    assert "s3cret" not in text and "alice" not in text
    assert "ep-x.neon.tech" in text and "neondb" in text
    assert q.describe_database("sqlite:///data/x.db").startswith("SQLite")


# --- The page itself ---
def test_app_renders_every_panel(seeded: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATABASE_URL", str(seeded.url))
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error

    labels = [m.label for m in at.metric]
    assert labels[:5] == [
        "Champion model",
        "Model MAPE · last 7 days",
        "Official forecast MAPE · 30 days",
        "Data coverage",
        "Collecting since",
    ]
    assert at.metric[0].value == "v2"
    assert at.metric[4].value == "2024-01-01"
    subheaders = [s.value for s in at.subheader]
    assert subheaders == [
        "Day-ahead forecast vs. actuals",
        "Accuracy over time: model vs. official forecast",
        "Monitoring status",
    ]


def test_app_renders_an_empty_database(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("DATABASE_URL", str(engine.url))
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].value == "none yet"
    assert any("No model forecast" in i.value for i in at.info)


def test_app_reports_an_unreachable_database(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(
        "DATABASE_URL", f"sqlite:///{(tmp_path / 'missing' / 'x.db').as_posix()}?mode=ro&uri=true"
    )
    at = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.error and "Could not read the database" in at.error[0].value


def test_dashboard_requirements_cover_its_imports() -> None:
    """Import the dashboard in a fresh interpreter: no training/serving library may be pulled in."""
    code = (
        "import sys, json; import dashboard.app; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    heavy = [
        m
        for m in ("lightgbm", "mlflow", "evidently", "sklearn", "torch", "fastapi", "optuna")
        if m in loaded
    ]
    assert not heavy, f"dashboard imports heavy modules: {heavy}"

    listed = {
        line.split(">=")[0].split("[")[0].strip().lower()
        for line in (PROJECT_ROOT / "dashboard" / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {"streamlit", "plotly", "pandas", "sqlalchemy", "psycopg", "pydantic-settings"} <= listed
    assert not ({"lightgbm", "mlflow", "evidently", "torch", "fastapi"} & listed)
