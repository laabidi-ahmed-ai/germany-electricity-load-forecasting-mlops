"""Read-only queries and metrics behind the Streamlit dashboard.

Everything the dashboard shows is derived here from the database tables through
``src.data.db`` readers, with pandas only - no LightGBM, MLflow or Evidently, so
the dashboard deploys with the lean ``dashboard/requirements.txt``. Keeping this
module free of Streamlit calls makes it importable and testable on its own.

Metric definitions (MAE in MW, MAPE in %) and the scoring rule (only forecasts issued by
their deadline count) come from ``src.monitoring.metrics``, the same code training and
monitoring use, so the dashboard agrees with both.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from src.data import db
from src.data.db import SOURCE_SMARD
from src.features.horizons import HORIZONS
from src.models.registry import REGISTERED_MODEL_NAME as MODEL_NAME
from src.monitoring import metrics
from src.monitoring.metrics import MIN_WINDOW_HOURS, WINDOWS_DAYS, compute_metrics

__all__ = ["MIN_WINDOW_HOURS", "MODEL_NAME", "WINDOWS_DAYS", "compute_metrics"]


# --- Engine ---
def make_engine(database_url: str | None = None) -> Engine:
    """Engine for the dashboard: on Postgres every transaction is opened read-only."""
    url = db.normalize_database_url(database_url) if database_url else None
    kwargs: dict[str, Any] = {}
    if url is not None and url.startswith("postgresql"):
        kwargs["connect_args"] = {"options": "-c default_transaction_read_only=on"}
    return db.get_engine(url, **kwargs)


def describe_database(database_url: str) -> str:
    """What kind of database the data comes from - no credentials, no server address."""
    from sqlalchemy.engine import make_url

    u = make_url(db.normalize_database_url(database_url))
    if u.drivername.startswith("sqlite"):
        return f"SQLite · {PurePosixPath(u.database or '').name or u.database}"
    return f"PostgreSQL · {u.database}"


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
    first = db.earliest_timestamp(engine, db.LoadActual, source=SOURCE_SMARD)
    last = db.latest_timestamp(engine, db.LoadActual, source=SOURCE_SMARD)
    n_hours = db.count_rows(engine, db.LoadActual, source=SOURCE_SMARD)
    expected = int((last - first) / pd.Timedelta(hours=1)) + 1 if first is not None else 0
    return {
        "first_actual": first,
        "last_actual": last,
        "n_hours": n_hours,
        "expected_hours": expected,
        "completeness_pct": (100.0 * n_hours / expected) if expected else None,
        "last_official": db.latest_timestamp(engine, db.LoadForecastOfficial, source=SOURCE_SMARD),
        "last_model_forecast": db.latest_timestamp(engine, db.LoadForecastModel),
    }


# --- Latest day-ahead forecast vs actuals ---
def latest_forecast_frame(engine: Engine, *, context_hours: int = 48) -> pd.DataFrame:
    """The most recently issued forecast with actuals and the official forecast on the same hours.

    Includes ``context_hours`` of history before the forecast window so the chart
    shows where the forecast picks up from. Columns: ``timestamp_utc, model_mw,
    official_mw, actual_mw, issued_at, model_version, horizon``; rows outside the forecast
    window have ``model_mw`` NaN.
    """
    forecast = db.read_latest_model_forecast(engine)
    cols = [
        "timestamp_utc",
        "model_mw",
        "official_mw",
        "actual_mw",
        "issued_at",
        "model_version",
        "horizon",
    ]
    if forecast.empty:
        return pd.DataFrame(columns=cols)
    forecast = forecast.rename(columns={"forecast_mw": "model_mw"})
    start = forecast["timestamp_utc"].min() - pd.Timedelta(hours=context_hours)
    end = forecast["timestamp_utc"].max()

    hours = pd.DataFrame({"timestamp_utc": pd.date_range(start, end, freq="h", tz="UTC")})
    actual = db.read_load(engine, start=start, end=end).rename(columns={"load_mw": "actual_mw"})
    official = db.read_table(
        engine, db.LoadForecastOfficial, start=start, end=end, source=SOURCE_SMARD
    ).rename(columns={"forecast_mw": "official_mw"})[["timestamp_utc", "official_mw"]]
    out = (
        hours.merge(
            forecast[["timestamp_utc", "model_mw", "issued_at", "model_version", "horizon"]],
            how="left",
        )
        .merge(official, how="left")
        .merge(actual, how="left")
    )
    for col in ("issued_at", "model_version", "horizon"):
        out[col] = out[col].ffill().bfill()
    return out[cols]


def deadline_status(window: pd.DataFrame) -> dict[str, Any]:
    """When the shown issue was due, and how many of its hours it made in time.

    ``window`` is the forecast hours of ``latest_forecast_frame`` (one issue). For D+1
    every hour shares one deadline; for a rolling horizon each hour is its own deadline.
    """
    on_time = _on_time(window)
    deadline = HORIZONS[window["horizon"].iloc[0]].issue_deadline(window["timestamp_utc"])
    return {
        "deadline": deadline.min(),
        "n_hours": len(window),
        "n_on_time": int(on_time.sum()),
    }


def scored_hours(window: pd.DataFrame) -> pd.DataFrame:
    """The shown issue's hours that are scored: issued by their deadline, with an actual (and
    the official forecast to compare with) - the hours the accuracy panel counts for it."""
    observed = window["actual_mw"].notna() & window["official_mw"].notna()
    return window[_on_time(window) & observed]


def _on_time(window: pd.DataFrame) -> pd.Series:
    deadline = HORIZONS[window["horizon"].iloc[0]].issue_deadline(window["timestamp_utc"])
    return window["issued_at"] <= deadline


# --- Accuracy over time: model vs the official forecast (shared with monitoring) ---
def actual_vs_official(
    engine: Engine, *, days: int, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Hours in the last ``days`` with both an actual and an official forecast."""
    as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    return metrics.actual_vs_official(engine, start=as_of - pd.Timedelta(days=days), end=as_of)


def model_issues(engine: Engine, base: pd.DataFrame) -> pd.DataFrame:
    """Every stored forecast issue for ``base``'s hours with its deadline (read once)."""
    return metrics.model_issues(engine, base)


def aligned_frame(
    engine: Engine, base: pd.DataFrame, issues: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The ``base`` hours that also have a model forecast (latest on-time issue per hour)."""
    return metrics.with_model_forecast(engine, base, issues=issues)


def forecast_status(
    engine: Engine, base: pd.DataFrame, issues: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per ``base`` hour: was its forecast ``on_time``, ``late`` or ``missing``."""
    return metrics.forecast_status(engine, base, issues)


PUNCTUALITY_KEYS = ("on_time_pct", "n_on_time", "n_late", "n_missing")


def window_summary(
    aligned: pd.DataFrame,
    *,
    days: int,
    as_of: pd.Timestamp | None = None,
    status: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Head-to-head over ``(as_of - days, as_of]`` - the window the monitoring job scores.

    With ``status`` (``forecast_status``) the on-time counts are included, judged or not.
    """
    as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    wm = metrics.window_metrics(aligned, as_of, days, status)
    out: dict[str, Any] = {"days": days, "n_hours": wm.n_hours, "judged": wm.judged}
    if status is not None:
        out.update({k: getattr(wm, k) for k in PUNCTUALITY_KEYS})
    if wm.judged:
        skip = {"window_days", "start", "end", "n_hours", *PUNCTUALITY_KEYS}
        out.update({k: v for k, v in asdict(wm).items() if k not in skip})
    return out


def daily_accuracy(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: MAE / MAPE of model and official forecast plus hours scored."""
    return metrics.daily_metrics(aligned)


def official_only_summary(
    base: pd.DataFrame, *, days: int, as_of: pd.Timestamp | None = None
) -> dict[str, Any]:
    """The official forecast scored alone over the last ``days`` of ``base`` - the bar to beat."""
    as_of = as_of if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    return {"days": days, **metrics.official_accuracy(base, as_of, days)}


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
    """Trigger checks, drift report and outcome of the most recent monitoring pass.

    A clear pass is stored as an ``EVENT_CHECK`` event; a pass that ran the champion/
    challenger (triggered or forced) as a single ``EVENT_RETRAIN`` event. Both carry
    ``details.decision.checks``, so the newest of either kind wins.

    Returns ``{"as_of", "created_at", "kind", "model_version" (champion the pass checked),
    "champion_version" (champion after the pass), "triggered", "forced", "outcome"
    (promoted | kept | skipped | cooldown | promoted (dry-run) | kept (dry-run) | None),
    "cv_gate" (passed | failed | None when the CV did not run), "cooldown" (none | held |
    bypassed), "post_promotion" (the pass's post-promotion report: ``label`` =
    not_applicable | pending | confirmed | rolled_back | inconclusive, each of the last three
    possibly " (report only)", plus ``reason``, ``new_version``, ``previous_version``; None for
    passes recorded before the check existed), "checks": DataFrame, "drift": dict | None,
    "drift_error": str | None, "drift_columns": DataFrame}`` or None.

    Post-promotion verdicts are events of their own (``EVENT_POST_PROMOTION``); they are not
    passes, so the newest pass still wins here.
    """
    passes = events[events["kind"].isin([db.EVENT_CHECK, db.EVENT_RETRAIN])]
    if passes.empty:
        return None
    row = passes.iloc[0]
    details = row["details_parsed"] or {}
    decision = details.get("decision") or {}
    outcome = details.get("outcome") or {}
    is_retrain = row["kind"] == db.EVENT_RETRAIN
    promoted = is_retrain and outcome.get("decision") == "promoted"
    # A forced run is the only retrain that no trigger fired.
    forced = is_retrain and (outcome.get("triggers") == ["forced"] or not bool(row["triggered"]))
    champion_after = (promoted and outcome.get("challenger_version")) or row["model_version"]
    post = details.get("post_promotion") or None
    if post and post.get("label") == db.DECISION_ROLLED_BACK:  # this pass rolled it back
        champion_after = post.get("previous_version") or champion_after
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
    cv_beats = outcome.get("cv_beats_all_baselines") if is_retrain else None
    return {
        "as_of": row["as_of"],
        "created_at": row["created_at"],
        "kind": row["kind"],
        "model_version": row["model_version"],
        "champion_version": champion_after,
        "triggered": bool(row["triggered"]),
        "forced": forced,
        "outcome": (outcome.get("decision") or None) if is_retrain else None,
        "cv_gate": None if cv_beats is None else ("passed" if cv_beats else "failed"),
        "cooldown": (outcome.get("cooldown") or "none") if is_retrain else "none",
        "post_promotion": post,
        "checks": checks,
        "drift": drift,
        "drift_error": decision.get("drift_error"),
        "drift_columns": columns,
    }


OUTCOME_TEXT = {
    "kept": "champion kept",
    "skipped": "challenger skipped",
    db.DECISION_COOLDOWN: "cooldown after a rejected challenger: nothing trained",
    "promoted (dry-run)": "dry run: challenger would be promoted",
    "kept (dry-run)": "dry run: champion would be kept",
}


def post_promotion_note(check: dict[str, Any]) -> str:
    """What the pass's post-promotion check said, if it said anything worth a line."""
    post = check.get("post_promotion") or {}
    label, new, previous = post.get("label"), post.get("new_version"), post.get("previous_version")
    if label == db.DECISION_ROLLED_BACK:
        return f" · ⏪ v{new} rolled back to v{previous}"
    if label == "pending":
        return f" · post-promotion check of v{new} pending ({post.get('reason')})"
    if label == db.DECISION_CONFIRMED:
        return f" · v{new} confirmed against v{previous}"
    if label == db.DECISION_INCONCLUSIVE:
        return f" · post-promotion check of v{new} closed without a verdict"
    if label and label.endswith("(report only)"):
        return f" · post-promotion check of v{new}: {label}"
    return ""


def pass_headline(check: dict[str, Any]) -> str:
    """One status line for the latest pass: what fired, what the retrain did, and what the
    post-promotion check said."""
    return _retrain_headline(check) + post_promotion_note(check)


def _retrain_headline(check: dict[str, Any]) -> str:
    if check["kind"] != db.EVENT_RETRAIN:  # e.g. --check-only: no champion/challenger ran
        if check["triggered"]:
            return "🔴 triggers fired (check only, no retrain)"
        return "🟢 all triggers clear"
    state = "🟠 retrain forced" if check["forced"] else "🔴 retraining triggered"
    outcome = check["outcome"]
    if outcome == "promoted":
        result = f"challenger promoted → v{check['champion_version']}"
    else:
        result = OUTCOME_TEXT.get(outcome, outcome or "no outcome recorded")
        if check.get("cv_gate") == "failed":
            result += " (its CV did not beat every baseline)"
    if check.get("cooldown") == "bypassed":
        result += " · cooldown bypassed"
    return f"{state} · {result}"


def event_timeline(events: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per event for the timeline: when, what happened, one-line reason."""
    cols = [
        "created_at",
        "as_of",
        "kind",
        "status",
        "model_version",
        "version_label",
        "reason",
        "model_mape_7d",
        "official_mape_7d",
        "on_time_pct_7d",
        "drift_share",
    ]
    if events.empty:
        return pd.DataFrame(columns=cols)

    def version_label(row: pd.Series) -> str:
        """Whose version the event names: a pass's champion, or the version a verdict judged."""
        version = row["model_version"]
        if version is None or (isinstance(version, float) and pd.isna(version)):
            return ""
        if row["kind"] == db.EVENT_POST_PROMOTION:
            role = "rolled-back" if row["decision"] == db.DECISION_ROLLED_BACK else "checked"
            return f"{role} v{version}"
        return f"champion v{version}"

    def status(row: pd.Series) -> str:
        if row["kind"] in (db.EVENT_RETRAIN, db.EVENT_POST_PROMOTION):
            return str(row["decision"] or row["kind"])
        return "triggered" if row["triggered"] else "ok"

    out = events.assign(
        status=events.apply(status, axis=1),
        version_label=events.apply(version_label, axis=1),
        on_time_pct_7d=events["details_parsed"].map(_on_time_pct_7d),
    )
    return out[cols].sort_values("created_at", ascending=False).reset_index(drop=True)


def _on_time_pct_7d(details: dict[str, Any]) -> float | None:
    """The 7-day on-time share a pass recorded in ``details`` (None before Step 8)."""
    performance = (details.get("decision") or {}).get("performance") or {}
    windows = performance.get("windows") or []
    return next((w.get("on_time_pct") for w in windows if w.get("window_days") == 7), None)
