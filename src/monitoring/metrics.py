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
from src.data.db import LOAD_SOURCE, OFFICIAL_SOURCE

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


def aligned_frame(
    engine: Engine,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    model_version: str | None = None,
    official_source: str = OFFICIAL_SOURCE,
    load_source: str = LOAD_SOURCE,
) -> pd.DataFrame:
    """Hours where actual, model forecast and official forecast all exist (inner join).

    With ``model_version=None`` the most recently *issued* model forecast per hour is
    used, so re-issued hours and a new champion count exactly once.
    """
    actual = db.read_load(engine, source=load_source, start=start, end=end).rename(
        columns={"load_mw": "actual_mw"}
    )
    filters = {"model_version": model_version} if model_version else {}
    model = db.read_table(engine, db.LoadForecastModel, start=start, end=end, **filters)
    official = db.read_table(
        engine, db.LoadForecastOfficial, start=start, end=end, source=official_source
    ).rename(columns={"forecast_mw": "official_mw"})[["timestamp_utc", "official_mw"]]

    if model.empty or actual.empty or official.empty:
        return pd.DataFrame(columns=ALIGNED_COLUMNS)
    model = (
        model.sort_values(["timestamp_utc", "issued_at"])
        .drop_duplicates("timestamp_utc", keep="last")
        .rename(columns={"forecast_mw": "model_mw"})[["timestamp_utc", "model_mw", "model_version"]]
    )
    out = actual.merge(model, on="timestamp_utc").merge(official, on="timestamp_utc")
    return out.sort_values("timestamp_utc").reset_index(drop=True)[ALIGNED_COLUMNS]


def score_pair(aligned: pd.DataFrame) -> dict[str, Any]:
    """Model and official forecast scored against the actuals of an aligned frame."""
    m = compute_metrics(aligned["actual_mw"], aligned["model_mw"])
    o = compute_metrics(aligned["actual_mw"], aligned["official_mw"])
    return {
        "model_mae": m["mae"],
        "model_mape": m["mape"],
        "official_mae": o["mae"],
        "official_mape": o["mape"],
        "model_beats_official": bool(m["mae"] < o["mae"]),
        "improvement_pct": float((o["mae"] - m["mae"]) / o["mae"] * 100.0),
    }


@dataclass
class WindowMetrics:
    window_days: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_hours: int
    model_mae: float | None = None
    model_mape: float | None = None
    official_mae: float | None = None
    official_mape: float | None = None
    model_beats_official: bool | None = None
    improvement_pct: float | None = None  # +x% = model MAE is x% lower than official

    @property
    def judged(self) -> bool:
        return self.model_mae is not None


def window_metrics(aligned: pd.DataFrame, as_of: pd.Timestamp, days: int) -> WindowMetrics:
    """Head-to-head over ``(as_of - days, as_of]``; unjudged below ``MIN_WINDOW_HOURS``."""
    start = as_of - pd.Timedelta(days=days)
    win = aligned[(aligned["timestamp_utc"] > start) & (aligned["timestamp_utc"] <= as_of)]
    wm = WindowMetrics(window_days=days, start=start, end=as_of, n_hours=len(win))
    if len(win) >= MIN_WINDOW_HOURS:
        for k, v in score_pair(win).items():
            setattr(wm, k, v)
    return wm


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


def official_accuracy(
    engine: Engine,
    as_of: pd.Timestamp,
    days: int,
    *,
    official_source: str = OFFICIAL_SOURCE,
    load_source: str = LOAD_SOURCE,
) -> dict[str, Any]:
    """The official forecast scored alone over ``(as_of - days, as_of]`` - the bar to beat.

    Available from day one, because the official forecast is ingested with the actuals
    long before any model forecast exists. Metrics are None below ``MIN_WINDOW_HOURS``.
    """
    start = as_of - pd.Timedelta(days=days)
    actual = db.read_load(engine, source=load_source, start=start, end=as_of)
    official = db.read_table(
        engine, db.LoadForecastOfficial, start=start, end=as_of, source=official_source
    )
    joined = actual.merge(official[["timestamp_utc", "forecast_mw"]], on="timestamp_utc")
    joined = joined[joined["timestamp_utc"] > start]
    if len(joined) < MIN_WINDOW_HOURS:
        return {"n_hours": len(joined), "mae": None, "mape": None}
    s = compute_metrics(joined["load_mw"], joined["forecast_mw"])
    return {"n_hours": len(joined), "mae": s["mae"], "mape": s["mape"]}
