"""Point-forecast metrics and the model-vs-official alignment.

Shared by training (``src.models.evaluate``), monitoring (``src.monitoring.performance``)
and the dashboard, so all three report the same numbers. Pure numpy/pandas on top of
the ``src.data.db`` readers - no LightGBM, MLflow or Evidently - because the dashboard
deploys with its lean requirements file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.data import db
from src.data.db import SOURCE_SMARD

MIN_WINDOW_HOURS = 24  # fewer aligned hours than this -> a window is not judged
ALIGNED_COLUMNS = ["timestamp_utc", "actual_mw", "model_mw", "official_mw", "model_version"]
DAILY_COLUMNS = ["date", "n_hours", "model_mae", "model_mape", "official_mae", "official_mape"]


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


def with_model_forecast(
    engine: Engine, base: pd.DataFrame, *, model_version: str | None = None
) -> pd.DataFrame:
    """Inner-join the model forecast onto ``actual_vs_official`` rows -> ``ALIGNED_COLUMNS``.

    With ``model_version=None`` the most recently *issued* model forecast per hour is
    used, so re-issued hours and a new champion count exactly once.
    """
    if base.empty:
        return pd.DataFrame(columns=ALIGNED_COLUMNS)
    filters = {"model_version": model_version} if model_version else {}
    model = db.read_table(
        engine,
        db.LoadForecastModel,
        start=base["timestamp_utc"].min(),
        end=base["timestamp_utc"].max(),
        **filters,
    )
    if model.empty:
        return pd.DataFrame(columns=ALIGNED_COLUMNS)
    model = (
        model.sort_values(["timestamp_utc", "issued_at"])
        .drop_duplicates("timestamp_utc", keep="last")
        .rename(columns={"forecast_mw": "model_mw"})[["timestamp_utc", "model_mw", "model_version"]]
    )
    out = base.merge(model, on="timestamp_utc")
    return out.sort_values("timestamp_utc").reset_index(drop=True)[ALIGNED_COLUMNS]


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

    @property
    def judged(self) -> bool:
        return self.model_mae is not None


def window_metrics(aligned: pd.DataFrame, as_of: pd.Timestamp, days: int) -> WindowMetrics:
    """Head-to-head over ``(as_of - days, as_of]``; unjudged below ``MIN_WINDOW_HOURS``."""
    start = as_of - pd.Timedelta(days=days)
    win = aligned[(aligned["timestamp_utc"] > start) & (aligned["timestamp_utc"] <= as_of)]
    score = score_pair(win) if len(win) >= MIN_WINDOW_HOURS else {}
    return WindowMetrics(window_days=days, start=start, end=as_of, n_hours=len(win), **score)


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
