"""Read-only queries and metrics behind the Streamlit dashboard.

Everything the dashboard shows is derived here from the database tables through
``src.data.db`` readers, with pandas only - no LightGBM, MLflow or Evidently, so
the dashboard deploys with the lean ``dashboard/requirements.txt``. Keeping this
module free of Streamlit calls makes it importable and testable on its own.

Metric definitions (MAE in MW, MAPE in %) match ``src.models.evaluate`` so the
numbers on the dashboard agree with the training reports and monitoring events.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src.data import db
from src.data.db import LOAD_SOURCE, OFFICIAL_SOURCE
from src.models.registry import REGISTERED_MODEL_NAME as MODEL_NAME
from src.monitoring import metrics
from src.monitoring.metrics import MIN_WINDOW_HOURS, compute_metrics

__all__ = ["MIN_WINDOW_HOURS", "MODEL_NAME", "compute_metrics"]


# --- Engine ---
def make_engine(database_url: str | None = None) -> Engine:
    """Engine for the dashboard: on Postgres every transaction is opened read-only."""
    url = db.normalize_database_url(database_url) if database_url else None
    kwargs: dict[str, Any] = {}
    if url is not None and url.startswith("postgresql"):
        kwargs["connect_args"] = {"options": "-c default_transaction_read_only=on"}
    return db.get_engine(url, **kwargs)


def describe_database(database_url: str) -> str:
    """Human-readable, secret-free description of where the data comes from."""
    from sqlalchemy.engine import make_url

    u = make_url(db.normalize_database_url(database_url))
    if u.drivername.startswith("sqlite"):
        return f"SQLite · {PurePosixPath(u.database or '').name or u.database}"
    host = u.host or "?"
    return f"{u.drivername.split('+')[0]} · {host}/{u.database}"


# --- Champion + coverage (headline KPIs) ---
def champion(engine: Engine, name: str = MODEL_NAME) -> dict[str, Any] | None:
    """The exported champion version with its parsed CV metrics, or None before bootstrap."""
    versions = db.list_model_artifacts(engine, name)
    if versions.empty:
        return None
    flagged = versions[versions["is_champion"]]
    if flagged.empty:
        return None
    row = flagged.iloc[0].to_dict()
    row["metrics"] = json.loads(row["metrics"]) if row.get("metrics") else {}
    row["n_versions"] = len(versions)
    return row


def model_versions(engine: Engine, name: str = MODEL_NAME) -> pd.DataFrame:
    """All exported versions, newest first, with the CV MAPE pulled out of the metrics JSON."""
    df = db.list_model_artifacts(engine, name)
    if df.empty:
        return df

    def cv_mape(text: str | None) -> float | None:
        if not text:
            return None
        return json.loads(text).get("lightgbm_mape_mean")

    df = df.assign(cv_mape=df["metrics"].map(cv_mape))
    return df.drop(columns=["metrics", "features"])


def coverage(engine: Engine) -> dict[str, Any]:
    """How much of the hourly history is in the database and how fresh it is."""
    first = _min_timestamp(engine, db.LoadActual, source=LOAD_SOURCE)
    last = db.latest_timestamp(engine, db.LoadActual, source=LOAD_SOURCE)
    n_hours = db.count_rows(engine, db.LoadActual, source=LOAD_SOURCE)
    expected = int((last - first) / pd.Timedelta(hours=1)) + 1 if first is not None else 0
    return {
        "first_actual": first,
        "last_actual": last,
        "n_hours": n_hours,
        "expected_hours": expected,
        "completeness_pct": (100.0 * n_hours / expected) if expected else None,
        "last_official": db.latest_timestamp(
            engine, db.LoadForecastOfficial, source=OFFICIAL_SOURCE
        ),
        "last_model_forecast": db.latest_timestamp(engine, db.LoadForecastModel),
    }


def _min_timestamp(engine: Engine, model: type[db.Base], **filters: Any) -> pd.Timestamp | None:
    from sqlalchemy import func, select

    stmt = select(func.min(model.timestamp_utc))
    for col, val in filters.items():
        stmt = stmt.where(getattr(model, col) == val)
    with engine.connect() as conn:
        value = conn.execute(stmt).scalar()
    if value is None:
        return None
    return db.to_utc(value)


# --- Latest day-ahead forecast vs actuals ---
def latest_forecast_frame(engine: Engine, *, context_hours: int = 48) -> pd.DataFrame:
    """The most recently issued forecast with actuals and the official forecast on the same hours.

    Includes ``context_hours`` of history before the forecast window so the chart
    shows where the forecast picks up from. Columns: ``timestamp_utc, model_mw,
    official_mw, actual_mw, issued_at, model_version``; rows outside the forecast
    window have ``model_mw`` NaN.
    """
    forecast = db.read_latest_model_forecast(engine)
    cols = ["timestamp_utc", "model_mw", "official_mw", "actual_mw", "issued_at", "model_version"]
    if forecast.empty:
        return pd.DataFrame(columns=cols)
    forecast = forecast.rename(columns={"forecast_mw": "model_mw"})
    start = forecast["timestamp_utc"].min() - pd.Timedelta(hours=context_hours)
    end = forecast["timestamp_utc"].max()

    hours = pd.DataFrame({"timestamp_utc": pd.date_range(start, end, freq="h", tz="UTC")})
    actual = db.read_load(engine, source=LOAD_SOURCE, start=start, end=end).rename(
        columns={"load_mw": "actual_mw"}
    )
    official = db.read_table(
        engine, db.LoadForecastOfficial, start=start, end=end, source=OFFICIAL_SOURCE
    ).rename(columns={"forecast_mw": "official_mw"})[["timestamp_utc", "official_mw"]]
    out = (
        hours.merge(
            forecast[["timestamp_utc", "model_mw", "issued_at", "model_version"]], how="left"
        )
        .merge(official, how="left")
        .merge(actual, how="left")
    )
    out["issued_at"] = out["issued_at"].ffill().bfill()
    out["model_version"] = out["model_version"].ffill().bfill()
    return out[cols]


# --- Accuracy over time: model vs the official forecast (shared with monitoring) ---
def aligned_frame(engine: Engine, *, days: int, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Hours in the last ``days`` where actual, model and official forecast all exist."""
    as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    return metrics.aligned_frame(engine, start=as_of - pd.Timedelta(days=days), end=as_of)


def window_summary(
    aligned: pd.DataFrame, *, days: int, as_of: pd.Timestamp | None = None
) -> dict[str, Any]:
    """Head-to-head over ``(as_of - days, as_of]`` - the window the monitoring job scores."""
    as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    wm = metrics.window_metrics(aligned, as_of, days)
    out: dict[str, Any] = {"days": days, "n_hours": wm.n_hours, "judged": wm.judged}
    if wm.judged:
        skip = {"window_days", "start", "end", "n_hours"}
        out.update({k: v for k, v in asdict(wm).items() if k not in skip})
    return out


def daily_accuracy(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: MAE / MAPE of model and official forecast plus hours scored."""
    return metrics.daily_metrics(aligned)


def official_only_summary(
    engine: Engine, *, days: int, as_of: pd.Timestamp | None = None
) -> dict[str, Any]:
    """The official forecast scored alone over the last ``days`` - the bar to beat."""
    as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    return {"days": days, **metrics.official_accuracy(engine, as_of, days)}


# --- Monitoring events: drift signals + retraining timeline ---
def monitoring_events(engine: Engine, *, limit: int = 200) -> pd.DataFrame:
    """Monitoring / retraining events, newest first, with the JSON ``details`` parsed."""
    events = db.read_monitoring_events(engine, limit=limit)
    if events.empty:
        return events.assign(details_parsed=pd.Series(dtype=object))
    events["details_parsed"] = events["details"].map(_parse_details)
    return events


def _parse_details(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {}


def latest_check(events: pd.DataFrame) -> dict[str, Any] | None:
    """Trigger checks and drift report of the most recent monitoring check.

    Returns ``{"as_of", "model_version", "checks": DataFrame, "drift": dict | None,
    "drift_error": str | None, "drift_columns": DataFrame}`` or None.
    """
    if events.empty:
        return None
    checks_only = events[events["kind"] == "check"]
    row = (checks_only if not checks_only.empty else events).iloc[0]
    decision = (row["details_parsed"] or {}).get("decision") or {}
    checks = pd.DataFrame(
        decision.get("checks") or [], columns=["name", "fired", "value", "threshold", "detail"]
    )
    drift = decision.get("drift")
    columns = pd.DataFrame(
        (drift or {}).get("columns") or [],
        columns=["column", "score", "threshold", "drifted", "method"],
    )
    for extra in ("prediction", "target"):
        item = (drift or {}).get(extra)
        if item:
            columns = pd.concat([columns, pd.DataFrame([item])], ignore_index=True)
    return {
        "as_of": row["as_of"],
        "created_at": row["created_at"],
        "model_version": row["model_version"],
        "triggered": bool(row["triggered"]),
        "checks": checks,
        "drift": drift,
        "drift_error": decision.get("drift_error"),
        "drift_columns": columns,
    }


def event_timeline(events: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per event for the timeline: when, what happened, one-line reason."""
    cols = [
        "created_at",
        "as_of",
        "kind",
        "status",
        "model_version",
        "reason",
        "model_mape_7d",
        "official_mape_7d",
        "drift_share",
    ]
    if events.empty:
        return pd.DataFrame(columns=cols)

    def status(row: pd.Series) -> str:
        if row["kind"] == "retrain":
            return str(row["decision"] or "retrain")
        return "triggered" if row["triggered"] else "ok"

    out = events.assign(status=events.apply(status, axis=1))
    return out[cols].sort_values("created_at", ascending=False).reset_index(drop=True)
