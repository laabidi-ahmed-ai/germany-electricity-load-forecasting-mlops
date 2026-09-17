"""Seasonal-naive baselines: the bar every real model must clear (README §8).

* ``seasonal_naive_24``  : load[t] = load[t-24]   (same hour yesterday)
* ``seasonal_naive_168`` : load[t] = load[t-168]  (same hour last week)

Both read their prediction straight from the leakage-safe lag columns of the
feature frame, so they are day-ahead valid by construction (most recent
observation >= 24h old, see ``src.features.horizons``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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

    def __repr__(self) -> str:
        return f"SeasonalNaive(lag_hours={self.lag_hours})"


def seasonal_naive_24() -> SeasonalNaive:
    return SeasonalNaive(24)


def seasonal_naive_168() -> SeasonalNaive:
    return SeasonalNaive(168)


BASELINE_FACTORIES = {
    "seasonal_naive_24": seasonal_naive_24,
    "seasonal_naive_168": seasonal_naive_168,
}
