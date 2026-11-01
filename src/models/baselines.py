"""Seasonal-naive baselines: the bar every real model must clear (README §8).

* ``seasonal_naive_24``  : load[t] = load[t-24]   (same hour yesterday)
* ``seasonal_naive_48``  : load[t] = load[t-48]   (same hour two days ago)
* ``seasonal_naive_168`` : load[t] = load[t-168]  (same hour last week)

Both read their prediction straight from the leakage-safe lag columns of the
feature frame. ``baseline_factories(horizon)`` picks the pair that is honest for the
horizon: the most recent same hour it allows (24 h day-ahead, 48 h for the calendar
day D+1) plus the same hour last week - a baseline must not see what the model cannot.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import pandas as pd

from src.features.horizons import DAY_AHEAD, Horizon


class SeasonalNaive:
    """Predict the value observed ``lag_hours`` earlier. Implements the fit/predict protocol."""

    def __init__(self, lag_hours: int) -> None:
        if lag_hours < 24:
            raise ValueError("a day-ahead baseline needs lag_hours >= 24")
        self.lag_hours = lag_hours
        self.column = f"load_lag_{lag_hours}"

    @property
    def name(self) -> str:
        return f"seasonal_naive_{self.lag_hours}"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SeasonalNaive:
        if self.column not in X.columns:
            raise KeyError(f"{self.name} needs column {self.column!r} in the feature frame")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype="float64")

    def get_params(self) -> dict[str, int]:
        return {"lag_hours": self.lag_hours}


def baseline_factories(horizon: Horizon = DAY_AHEAD) -> dict[str, partial[SeasonalNaive]]:
    """Same hour on the most recent day ``horizon`` allows, and same hour last week."""
    daily = 24 * math.ceil(max(horizon.min_lag_hours, 24) / 24)
    return {f"seasonal_naive_{k}": partial(SeasonalNaive, k) for k in (daily, 168)}


BASELINE_FACTORIES = baseline_factories(DAY_AHEAD)
