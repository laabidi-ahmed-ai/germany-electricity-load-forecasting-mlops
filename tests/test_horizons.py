"""Horizon-aware feature selection: what a day-ahead model may and may not see."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    LAG_HOURS,
    ROLLING_STATS,
    ROLLING_WINDOWS,
    WEATHER_COLUMNS,
    build_feature_frame,
    feature_columns,
)
from src.features.horizons import (
    DAY_AHEAD,
    HORIZONS,
    NOWCAST,
    Horizon,
    excluded_features,
    most_recent_load_lag,
    select_features,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="UTC")
    load = pd.DataFrame({"timestamp_utc": ts, "load_mw": 50_000 + np.arange(len(ts))})
    return build_feature_frame(load)


def test_most_recent_load_lag_parses_the_naming_contract() -> None:
    assert most_recent_load_lag("load_lag_1") == 1
    assert most_recent_load_lag("load_lag_168") == 168
    assert most_recent_load_lag("load_roll_mean_24") == 1  # window ends at t-1
    assert most_recent_load_lag("load_roll_std_168") == 1
    assert most_recent_load_lag("hour") is None
    assert most_recent_load_lag("temperature_2m_de_avg") is None
    assert most_recent_load_lag("load_mw") is None


def test_day_ahead_keeps_only_lags_of_24h_or_more(frame) -> None:
    kept = select_features(frame.columns, DAY_AHEAD)
    dropped = excluded_features(frame.columns, DAY_AHEAD)

    assert "load_mw" not in kept and "load_mw" not in dropped
    assert {"load_lag_24", "load_lag_48", "load_lag_168"} <= set(kept)
    assert "load_lag_1" in dropped
    for w in ROLLING_WINDOWS:
        for s in ROLLING_STATS:
            assert f"load_roll_{s}_{w}" in dropped
    # Calendar + weather are always available.
    for col in ("hour", "day_of_week", "is_holiday", "hour_sin", "month_cos", *WEATHER_COLUMNS):
        assert col in kept
    assert set(kept) | set(dropped) == set(feature_columns(frame))
    assert not set(kept) & set(dropped)


def test_day_ahead_features_never_use_load_newer_than_24h(frame) -> None:
    for col in select_features(frame.columns, DAY_AHEAD):
        lag = most_recent_load_lag(col)
        assert lag is None or lag >= 24, col


def test_nowcast_uses_the_full_feature_set(frame) -> None:
    assert select_features(frame.columns, NOWCAST) == feature_columns(frame)
    assert excluded_features(frame.columns, NOWCAST) == []
    assert "load_lag_1" in select_features(frame.columns, NOWCAST)


def test_selection_preserves_order_and_is_generic() -> None:
    cols = ["load_mw", "hour", "load_lag_1", "load_lag_24", "x", "load_roll_min_24"]
    assert select_features(cols, DAY_AHEAD) == ["hour", "load_lag_24", "x"]
    two_day = Horizon("two_day", min_lag_hours=48)
    assert select_features(["load_lag_24", "load_lag_48", "load_lag_168"], two_day) == [
        "load_lag_48",
        "load_lag_168",
    ]


def test_registry_and_constants() -> None:
    assert HORIZONS == {"day_ahead": DAY_AHEAD, "nowcast": NOWCAST}
    assert DAY_AHEAD.min_lag_hours == 24 and NOWCAST.min_lag_hours == 1
    assert min(LAG_HOURS) == 1  # the frame really does contain the nowcast-only lag
