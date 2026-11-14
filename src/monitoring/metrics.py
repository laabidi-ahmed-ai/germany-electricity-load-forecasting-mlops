"""Point-forecast metrics and the model-vs-official alignment.

Shared by training (``src.models.evaluate``), monitoring (``src.monitoring.performance``)
and the dashboard, so all three report the same numbers. Pure numpy/pandas on top of
the ``src.data.db`` readers - no LightGBM, MLflow or Evidently - because the dashboard
deploys with its lean requirements file.

The scoring rule: every issue of a model forecast is stored, and an hour is scored with
the latest issue made by that issue's deadline (``Horizon.issue_deadline``; for D+1,
12:00 Berlin the day before). A late issue is kept but never scored. Each hour of a
window is ``on_time``, ``late`` (issues exist, all after the deadline) or ``missing``
(none); ``on_time_pct`` is the on-time share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.data import db
from src.data.db import SOURCE_SMARD
from src.features.horizons import HORIZONS

MIN_WINDOW_HOURS = 24  # fewer aligned hours than this -> a window is not judged
WINDOWS_DAYS: tuple[int, ...] = (7, 30)  # the head-to-head windows monitoring and dashboard show
ALIGNED_COLUMNS = ["timestamp_utc", "actual_mw", "model_mw", "official_mw", "model_version"]
DAILY_COLUMNS = ["date", "n_hours", "model_mae", "model_mape", "official_mae", "official_mape"]
ON_TIME, LATE, MISSING = "on_time", "late", "missing"
STATUS_COLUMNS = ["timestamp_utc", "status"]


def compute_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """MAE, RMSE (MW) and MAPE (%) of a point forecast."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if np.isnan(yp).any():
        raise ValueError("predictions contain NaN")
    err = yp - yt
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.abs(yt)) * 100.0),
    }


def actual_vs_official(
    engine: Engine, *, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Hours with both an actual and an official forecast: ``timestamp_utc, actual_mw, official_mw``.

    Read once per report: the official-only score comes straight from it, and
    ``with_model_forecast`` adds the model's forecast for the head-to-head.
    """
    actual = db.read_load(engine, start=start, end=end).rename(columns={"load_mw": "actual_mw"})
    official = db.read_table(
        engine, db.LoadForecastOfficial, start=start, end=end, source=SOURCE_SMARD
    ).rename(columns={"forecast_mw": "official_mw"})[["timestamp_utc", "official_mw"]]
    out = actual.merge(official, on="timestamp_utc")
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def model_issues(engine: Engine, base: pd.DataFrame) -> pd.DataFrame:
    """Every stored model forecast issue for ``base``'s hours, all versions, with its deadline.

    The ``load_forecast_model`` rows plus ``deadline`` (from each row's horizon) and
    ``on_time`` (``issued_at <= deadline``). Read once per report and shared by the scores
    (``with_model_forecast``) and the on-time counts (``forecast_status``).
    """
    if base.empty:
        model = pd.DataFrame(columns=[c.name for c in db.LoadForecastModel.__table__.columns])
    else:
        model = db.read_table(
            engine,
            db.LoadForecastModel,
            start=base["timestamp_utc"].min(),
            end=base["timestamp_utc"].max(),
        )
    return _with_deadlines(model)


def _with_deadlines(model: pd.DataFrame) -> pd.DataFrame:
    if model.empty:
        return model.assign(
            deadline=pd.Series(dtype="datetime64[ns, UTC]"), on_time=pd.Series(dtype=bool)
        )
    unknown = set(model["horizon"]) - set(HORIZONS)
    if unknown:
        raise ValueError(f"stored forecasts have unknown horizons {sorted(unknown)}")
    deadline = pd.concat(
        [
            HORIZONS[name].issue_deadline(rows["timestamp_utc"])
            for name, rows in model.groupby("horizon")
        ]
    ).reindex(model.index)
    return model.assign(deadline=deadline, on_time=model["issued_at"] <= deadline)


def latest_on_time_issues(issues: pd.DataFrame, model_version: str | None = None) -> pd.DataFrame:
    """Per hour, the row of the latest issue made by its deadline - the one that is scored.

    ``issues`` comes from ``model_issues``; the result keeps its columns (``issued_at``
    included), one row per ``timestamp_utc``. With ``model_version`` only that version's
    issues compete. A late issue never wins, whatever else exists.
    """
    scored = issues[issues["on_time"]]
    if model_version:
        scored = scored[scored["model_version"] == str(model_version)]
    return scored.sort_values(["timestamp_utc", "issued_at"]).drop_duplicates(
        "timestamp_utc", keep="last"
    )


def with_model_forecast(
    engine: Engine,
    base: pd.DataFrame,
    *,
    model_version: str | None = None,
    issues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Inner-join the model forecast onto ``actual_vs_official`` rows -> ``ALIGNED_COLUMNS``.

    Per hour, the latest issue made by its deadline: a re-issue before the deadline
    replaces an earlier one, a late issue never does. With ``model_version=None`` every
    version competes, so a new champion counts once. ``issues`` (``model_issues``) saves
    the read when the caller already has them.
    """
    if base.empty:
        return pd.DataFrame(columns=ALIGNED_COLUMNS)
    issues = model_issues(engine, base) if issues is None else issues
    model = latest_on_time_issues(issues, model_version)
    if model.empty:
        return pd.DataFrame(columns=ALIGNED_COLUMNS)
    model = model.rename(columns={"forecast_mw": "model_mw"})[
        ["timestamp_utc", "model_mw", "model_version"]
    ]
    out = base.merge(model, on="timestamp_utc")
    return out.sort_values("timestamp_utc").reset_index(drop=True)[ALIGNED_COLUMNS]


def forecast_status(
    engine: Engine, base: pd.DataFrame, issues: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per hour of ``base``: ``on_time``, ``late`` or ``missing`` -> ``STATUS_COLUMNS``.

    Every version counts: this is the forecasting service's punctuality, whichever version
    a report scores. Hours before the first forecast ever stored are left out - nothing was
    due then.
    """
    first = db.earliest_timestamp(engine, db.LoadForecastModel)
    if base.empty or first is None:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    issues = model_issues(engine, base) if issues is None else issues
    hours = base.loc[base["timestamp_utc"] >= first, "timestamp_utc"].reset_index(drop=True)
    covered = hours.isin(issues["timestamp_utc"])
    in_time = hours.isin(issues.loc[issues["on_time"], "timestamp_utc"])
    status = np.select([in_time, covered], [ON_TIME, LATE], default=MISSING)
    return pd.DataFrame({"timestamp_utc": hours, "status": status})


def on_time_counts(status: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """On-time, late and missing hours in ``(start, end]``; the share None below
    ``MIN_WINDOW_HOURS``. The keys are fields of ``WindowMetrics``."""
    ts = status["timestamp_utc"]
    s = status.loc[(ts > start) & (ts <= end), "status"]
    n_on_time = int((s == ON_TIME).sum())
    return {
        "n_on_time": n_on_time,
        "n_late": int((s == LATE).sum()),
        "n_missing": int((s == MISSING).sum()),
        "on_time_pct": 100.0 * n_on_time / len(s) if len(s) >= MIN_WINDOW_HOURS else None,
    }


def improvement_pct(reference_mae: float, candidate_mae: float) -> float:
    """How much lower the candidate's MAE is, in % of the reference (+x% = candidate better)."""
    return float((reference_mae - candidate_mae) / reference_mae * 100.0)


def score_pair(aligned: pd.DataFrame) -> dict[str, Any]:
    """Model and official forecast scored against the actuals of an aligned frame.

    The keys are the fields of ``HeadToHead``.
    """
    m = compute_metrics(aligned["actual_mw"], aligned["model_mw"])
    o = compute_metrics(aligned["actual_mw"], aligned["official_mw"])
    return {
        "model_mae": m["mae"],
        "model_mape": m["mape"],
        "official_mae": o["mae"],
        "official_mape": o["mape"],
        "model_beats_official": bool(m["mae"] < o["mae"]),
        "improvement_pct": improvement_pct(o["mae"], m["mae"]),
    }


@dataclass(kw_only=True)
class HeadToHead:
    """Model vs official forecast on the same hours (MAE in MW, MAPE in %)."""

    model_mae: float | None = None
    model_mape: float | None = None
    official_mae: float | None = None
    official_mape: float | None = None
    model_beats_official: bool | None = None
    improvement_pct: float | None = None  # +x% = model MAE is x% lower than official


@dataclass(kw_only=True)
class WindowMetrics(HeadToHead):
    window_days: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_hours: int
    # punctuality (``forecast_status``): hours with an on-time forecast, a late one, none
    n_on_time: int = 0
    n_late: int = 0
    n_missing: int = 0
    on_time_pct: float | None = None

    @property
    def judged(self) -> bool:
        return self.model_mae is not None


def window_metrics(
    aligned: pd.DataFrame,
    as_of: pd.Timestamp,
    days: int,
    status: pd.DataFrame | None = None,
) -> WindowMetrics:
    """Head-to-head over ``(as_of - days, as_of]``; unjudged below ``MIN_WINDOW_HOURS``.

    With ``status`` (``forecast_status``) the window's on-time counts are filled in too.
    """
    start = as_of - pd.Timedelta(days=days)
    win = aligned[(aligned["timestamp_utc"] > start) & (aligned["timestamp_utc"] <= as_of)]
    score = score_pair(win) if len(win) >= MIN_WINDOW_HOURS else {}
    punctuality = on_time_counts(status, start, as_of) if status is not None else {}
    return WindowMetrics(
        window_days=days, start=start, end=as_of, n_hours=len(win), **score, **punctuality
    )


def daily_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: MAE / MAPE of model and official forecast plus hours scored."""
    if aligned.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    rows = []
    for date, g in aligned.groupby(aligned["timestamp_utc"].dt.floor("D"), sort=True):
        s = score_pair(g)
        rows.append(
            {
                "date": date,
                "n_hours": len(g),
                "model_mae": s["model_mae"],
                "model_mape": s["model_mape"],
                "official_mae": s["official_mae"],
                "official_mape": s["official_mape"],
            }
        )
    return pd.DataFrame(rows, columns=DAILY_COLUMNS)


def official_accuracy(base: pd.DataFrame, as_of: pd.Timestamp, days: int) -> dict[str, Any]:
    """The official forecast scored alone over ``(as_of - days, as_of]`` - the bar to beat.

    ``base`` comes from ``actual_vs_official``. Available from day one, because the
    official forecast is ingested with the actuals long before any model forecast
    exists. Metrics are None below ``MIN_WINDOW_HOURS``.
    """
    start = as_of - pd.Timedelta(days=days)
    ts = base["timestamp_utc"]
    win = base[(ts > start) & (ts <= as_of)]
    if len(win) < MIN_WINDOW_HOURS:
        return {"n_hours": len(win), "mae": None, "mape": None}
    s = compute_metrics(win["actual_mw"], win["official_mw"])
    return {"n_hours": len(win), "mae": s["mae"], "mape": s["mape"]}
