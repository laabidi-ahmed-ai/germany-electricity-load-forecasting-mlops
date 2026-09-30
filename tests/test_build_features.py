"""Feature-engineering tests: exact lag/rolling semantics, calendar/DST, weather join,
time-based splits and - most importantly - a dedicated leakage test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import db
from src.features import build_features as bf
from src.features.build_features import (
    LAG_HOURS,
    ROLLING_STATS,
    ROLLING_WINDOWS,
    TARGET,
    WEATHER_COLUMNS,
    build_feature_frame,
    build_features,
    expanding_window_splits,
    feature_columns,
    make_calendar_features,
)

N_HOURS = 24 * 30  # 30 days


def synthetic_load(start="2024-03-01", n=N_HOURS, *, seed=0) -> pd.DataFrame:
    """Load = row position + noise, so lag/rolling values are exactly predictable."""
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    values = 50_000 + np.arange(n, dtype=float) * 10 + rng.normal(0, 100, n)
    return pd.DataFrame({"timestamp_utc": ts, TARGET: values})


def synthetic_weather(load: pd.DataFrame, *, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = pd.DataFrame({"timestamp_utc": load["timestamp_utc"]})
    for col in WEATHER_COLUMNS:
        w[col] = rng.normal(10, 5, len(load))
    return w


@pytest.fixture
def load() -> pd.DataFrame:
    return synthetic_load()


@pytest.fixture
def weather(load) -> pd.DataFrame:
    return synthetic_weather(load)


# --- Lags & rolling ---
def test_lags_are_exact_time_offsets(load) -> None:
    df = build_feature_frame(load, dropna=False)
    s = load.set_index("timestamp_utc")[TARGET]
    for k in LAG_HOURS:
        expected = s.shift(k).to_numpy()
        got = df[f"load_lag_{k}"].to_numpy()
        assert np.array_equal(np.isnan(got), np.isnan(expected))
        assert np.allclose(got[~np.isnan(got)], expected[~np.isnan(expected)])
    # First row of the *kept* frame is exactly 168h in, and its lag_168 is row 0's load.
    kept = build_feature_frame(load)
    assert kept.index[0] == load["timestamp_utc"].iloc[168]
    assert kept["load_lag_168"].iloc[0] == pytest.approx(load[TARGET].iloc[0])


def test_lags_are_time_based_not_row_based(load) -> None:
    """Delete a day of data: lag_24 across the hole must be NaN, not '24 rows ago'."""
    hole = load["timestamp_utc"].between("2024-03-10T00:00Z", "2024-03-10T23:00Z")
    gappy = load[~hole].reset_index(drop=True)
    df = build_feature_frame(gappy, dropna=False)

    t = pd.Timestamp("2024-03-11T05:00Z")  # t-24 falls inside the hole
    assert np.isnan(df.loc[t, "load_lag_24"])
    assert df.loc[t, "load_lag_1"] == pytest.approx(
        gappy.set_index("timestamp_utc").loc[t - pd.Timedelta(hours=1), TARGET]
    )
    # The frame itself is on a complete hourly index (hole rows present, target NaN).
    assert len(df) == N_HOURS
    assert np.isnan(df.loc["2024-03-10T12:00Z", TARGET])


def test_rolling_windows_end_at_t_minus_1(load) -> None:
    df = build_feature_frame(load, dropna=False)
    s = load.set_index("timestamp_utc")[TARGET]
    t = s.index[200]
    for w in ROLLING_WINDOWS:
        window = s.iloc[200 - w : 200]  # t-w .. t-1, excludes t
        assert len(window) == w
        assert df.loc[t, f"load_roll_mean_{w}"] == pytest.approx(window.mean())
        assert df.loc[t, f"load_roll_std_{w}"] == pytest.approx(window.std())
        assert df.loc[t, f"load_roll_min_{w}"] == pytest.approx(window.min())
        assert df.loc[t, f"load_roll_max_{w}"] == pytest.approx(window.max())


def test_rolling_max_never_sees_current_hour() -> None:
    """With a strictly increasing series, roll_max at t must be < load at t."""
    ts = pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC")
    inc = pd.DataFrame({"timestamp_utc": ts, TARGET: np.arange(400, dtype=float)})
    df = build_feature_frame(inc)
    for w in ROLLING_WINDOWS:
        assert (df[f"load_roll_max_{w}"] < df[TARGET]).all()
        assert (df[f"load_roll_max_{w}"] == df["load_lag_1"]).all()


def test_dropna_removes_rows_without_full_history(load, weather) -> None:
    df = build_feature_frame(load, weather)
    assert len(df) == N_HOURS - max(LAG_HOURS)
    assert not df[[TARGET, *bf.lag_and_rolling_columns()]].isna().any().any()


# --- Calendar ---
def test_calendar_uses_berlin_local_time_across_dst() -> None:
    idx = pd.DatetimeIndex(
        [
            "2024-01-15T00:00Z",  # 01:00 CET, Monday
            "2024-03-31T00:30Z",  # 01:30 CET, just before spring-forward
            "2024-03-31T01:30Z",  # 03:30 CEST, right after (02:xx does not exist)
            "2024-07-01T22:00Z",  # 00:00 CEST on 2024-07-02 (Tuesday)
        ],
        tz="UTC",
    )
    cal = make_calendar_features(idx)
    assert list(cal["hour"]) == [1, 1, 3, 0]
    assert list(cal["day_of_week"]) == [0, 6, 6, 1]
    assert list(cal["is_weekend"]) == [0, 1, 1, 0]
    assert list(cal["month"]) == [1, 3, 3, 7]


def test_nationwide_german_holidays_only() -> None:
    idx = pd.DatetimeIndex(
        [
            "2024-10-03T10:00Z",  # Tag der Deutschen Einheit (nationwide)
            "2024-12-25T10:00Z",  # 1. Weihnachtstag (nationwide)
            "2024-05-30T10:00Z",  # Fronleichnam - regional only, must NOT count
            "2024-10-31T10:00Z",  # Reformationstag - regional only
            "2024-12-31T23:30Z",  # 00:30 local on Jan 1 -> Neujahr (local-date logic)
            "2024-06-12T10:00Z",  # ordinary Wednesday
        ],
        tz="UTC",
    )
    cal = make_calendar_features(idx)
    assert list(cal["is_holiday"]) == [1, 1, 0, 0, 1, 0]


def test_cyclical_encodings_are_unit_circle_and_wrap() -> None:
    idx = pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")
    cal = make_calendar_features(idx)
    for a, b in (("hour_sin", "hour_cos"), ("dow_sin", "dow_cos"), ("month_sin", "month_cos")):
        assert np.allclose(cal[a] ** 2 + cal[b] ** 2, 1.0)
    # hour 23 is a neighbour of hour 0 in (sin, cos) space, far from hour 12.
    h = cal.drop_duplicates("hour").set_index("hour")

    def d(i: int, j: int) -> float:
        return np.hypot(
            h.loc[i, "hour_sin"] - h.loc[j, "hour_sin"], h.loc[i, "hour_cos"] - h.loc[j, "hour_cos"]
        )

    assert d(23, 0) < d(23, 12)
    assert d(23, 0) == pytest.approx(d(0, 1))
    # December wraps to January.
    m = cal.drop_duplicates("month").set_index("month")
    assert np.hypot(
        m.loc[12, "month_sin"] - m.loc[1, "month_sin"],
        m.loc[12, "month_cos"] - m.loc[1, "month_cos"],
    ) == pytest.approx(
        np.hypot(
            m.loc[1, "month_sin"] - m.loc[2, "month_sin"],
            m.loc[1, "month_cos"] - m.loc[2, "month_cos"],
        )
    )


# --- Weather ---
def test_weather_is_joined_on_timestamp_and_missing_hours_are_nan(load, weather) -> None:
    partial = weather.iloc[:-48]  # last two days have no weather
    df = build_feature_frame(load, partial)
    assert list(WEATHER_COLUMNS) == [c for c in df.columns if c.endswith("_de_avg")]
    joined = df.join(partial.set_index("timestamp_utc"), rsuffix="_src", how="inner")
    for col in WEATHER_COLUMNS:
        assert np.allclose(joined[col], joined[f"{col}_src"])
    assert df[list(WEATHER_COLUMNS)].tail(48).isna().all().all()
    assert len(df) == N_HOURS - 168  # weather NaN does not drop rows


def test_no_weather_gives_nan_columns(load) -> None:
    df = build_feature_frame(load, None)
    assert df[list(WEATHER_COLUMNS)].isna().all().all()


# --- LEAKAGE: every feature at hour t depends only on data strictly before t ---
def _assert_rows_unchanged(a: pd.DataFrame, b: pd.DataFrame, upto: pd.Timestamp, cols) -> None:
    left, right = a.loc[:upto, cols], b.loc[:upto, cols]
    assert left.shape == right.shape
    both_nan = left.isna() & right.isna()
    assert (np.isclose(left, right) | both_nan).all().all(), (
        f"features up to {upto} changed after perturbing only data at/after {upto}"
    )


@pytest.mark.parametrize("t_pos", [170, 300, 500, 719])
def test_no_feature_uses_target_at_or_after_t(load, weather, t_pos: int) -> None:
    """Perturb the target at every hour >= t: no feature row <= t may change."""
    base = build_feature_frame(load, weather, dropna=False)
    t = load["timestamp_utc"].iloc[t_pos]

    poisoned = load.copy()
    poisoned.loc[poisoned["timestamp_utc"] >= t, TARGET] += 1e6
    after = build_feature_frame(poisoned, weather, dropna=False)

    _assert_rows_unchanged(base, after, t, feature_columns(base))
    # ...and the target itself did change at t (the test has teeth).
    assert after.loc[t, TARGET] == pytest.approx(base.loc[t, TARGET] + 1e6)


@pytest.mark.parametrize("t_pos", [170, 300, 500, 719])
def test_no_feature_uses_weather_after_t(load, weather, t_pos: int) -> None:
    """Perturb weather strictly after t: no feature row <= t may change.

    Weather *at* t is allowed (it will be a forecast for t at serving time).
    """
    base = build_feature_frame(load, weather, dropna=False)
    t = load["timestamp_utc"].iloc[t_pos]

    poisoned = weather.copy()
    mask = poisoned["timestamp_utc"] > t
    poisoned.loc[mask, list(WEATHER_COLUMNS)] += 1e6
    after = build_feature_frame(load, poisoned, dropna=False)

    _assert_rows_unchanged(base, after, t, feature_columns(base))


def test_leakage_test_has_teeth_past_values_do_propagate(load, weather) -> None:
    """Sanity: changing the target at t-1 *must* change lag_1 and every rolling stat at t."""
    base = build_feature_frame(load, weather, dropna=False)
    t = load["timestamp_utc"].iloc[400]
    bumped = load.copy()
    bumped.loc[bumped["timestamp_utc"] == t - pd.Timedelta(hours=1), TARGET] += 1e6
    after = build_feature_frame(bumped, weather, dropna=False)

    assert after.loc[t, "load_lag_1"] == pytest.approx(base.loc[t, "load_lag_1"] + 1e6)
    for w in ROLLING_WINDOWS:
        for stat in ("mean", "std", "max"):
            assert after.loc[t, f"load_roll_{stat}_{w}"] != base.loc[t, f"load_roll_{stat}_{w}"]
    # Rows before t are untouched.
    _assert_rows_unchanged(base, after, t - pd.Timedelta(hours=1), feature_columns(base))


def test_no_fitted_transform_in_feature_frame(load, weather) -> None:
    """Features are raw (no scaling): building on a subset gives identical values."""
    full = build_feature_frame(load, weather)
    half = build_feature_frame(load.iloc[: N_HOURS // 2], weather)
    common = half.index
    pd.testing.assert_frame_equal(full.loc[common], half, check_like=True)


# --- Time-based splits ---
def test_expanding_window_folds_grow_and_never_overlap(load, weather) -> None:
    df = build_feature_frame(load, weather)
    folds = list(
        expanding_window_splits(df, n_splits=3, val_hours=48, min_train_hours=100, gap_hours=24)
    )
    assert len(folds) == 3
    prev_train = 0
    for train, val in folds:
        assert len(val) == 48
        assert train.index.max() < val.index.min()
        assert val.index.min() - train.index.max() >= pd.Timedelta(hours=24 + 1)  # gap honoured
        assert len(train) > prev_train  # expanding
        prev_train = len(train)
    # Folds tile the end of the series, most recent last.
    assert folds[-1][1].index.max() == df.index.max()
    assert folds[0][1].index.min() == df.index.max() - pd.Timedelta(hours=3 * 48 - 1)


def test_expanding_window_refuses_too_little_training_data(load, weather) -> None:
    df = build_feature_frame(load, weather)
    with pytest.raises(ValueError, match="training hours"):
        list(expanding_window_splits(df, n_splits=2, val_hours=24, min_train_hours=10_000))


# --- End-to-end from the database + parquet ---
def test_build_features_from_db_and_persist(engine, tmp_path, load) -> None:
    db.upsert_dataframe(engine, db.LoadActual, load.assign(source="smard"))
    per_city = []
    for city, offset in (("A", 0.0), ("B", 2.0)):
        per_city.append(
            pd.DataFrame(
                {
                    "timestamp_utc": load["timestamp_utc"],
                    "city": city,
                    "source": "archive",
                    "temperature_2m": 10.0 + offset,
                    "wind_speed_10m": 5.0,
                    "shortwave_radiation": 100.0,
                    "cloud_cover": 50.0,
                    "relative_humidity_2m": 70.0,
                }
            )
        )
    db.upsert_dataframe(engine, db.WeatherHourly, pd.concat(per_city))

    out = tmp_path / "features.parquet"
    df = build_features(engine, start="2024-03-01", output=out)

    assert out.exists()
    assert len(df) == N_HOURS - 168
    assert df["temperature_2m_de_avg"].iloc[0] == pytest.approx(11.0)  # mean of the 2 cities
    assert str(df.index.tz) == "UTC"

    back = pd.read_parquet(out)
    assert str(back.index.tz) == "UTC"
    assert list(back.columns) == list(df.columns)
    assert len(back) == len(df)


def test_feature_columns_excludes_target(load) -> None:
    df = build_feature_frame(load)
    cols = feature_columns(df)
    assert TARGET not in cols
    assert set(cols) == set(df.columns) - {TARGET}
    assert len(cols) == len(LAG_HOURS) + len(ROLLING_WINDOWS) * len(ROLLING_STATS) + 12 + 5


def test_empty_load_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        build_feature_frame(pd.DataFrame(columns=["timestamp_utc", TARGET]))
