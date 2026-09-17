"""Leakage-safe feature engineering for day-ahead hourly load forecasting.

Input: hourly actual load (``db.read_load``) and national-average weather.
Output: one row per hour, indexed by ``timestamp_utc``, with the target ``load_mw``
plus the feature columns listed in ``feature_columns()``.

Leakage rules - every feature for hour t uses only information available strictly
before t:

* Lags ``load_lag_{k}`` = load at t-k hours (k in 1, 24, 48, 168). Computed on a
  complete hourly index, so a lag is always a true time offset, never "k rows ago"
  across a data gap.
* Rolling stats over the previous 24h / 168h are computed on ``load.shift(1)``,
  so the window ends at t-1 and never contains hour t itself.
* Calendar features come from the Europe/Berlin clock (demand follows local time).
  Public holidays are the nationwide German ones only; Bundesland-specific days
  (Fronleichnam, Reformationstag, ...) are deliberately left out.
* Weather at hour t is joined as-is (reanalysis in training, a forecast for t at
  serving time - see ``src.features.horizons`` for why that is not leakage).

Which columns a model may use for a given lead time is decided in
``src.features.horizons``, not here. No scaler or other fitted transform lives here
either; if one is ever needed it must be fit on the training split only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from config.log import configure_logging
from config.settings import PROJECT_ROOT, get_settings
from src.data import db
from src.data.weather_client import WEATHER_VARS, national_average

log = logging.getLogger(__name__)

TARGET = "load_mw"
LOCAL_TZ = "Europe/Berlin"

LAG_HOURS: tuple[int, ...] = (1, 24, 48, 168)
ROLLING_WINDOWS: tuple[int, ...] = (24, 168)
ROLLING_STATS: tuple[str, ...] = ("mean", "std", "min", "max")
# A rolling window still produces a value when a few hours are missing inside it.
ROLLING_MIN_PERIODS_FRAC = 0.75

WEATHER_COLUMNS: tuple[str, ...] = tuple(f"{v}_de_avg" for v in WEATHER_VARS)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features.parquet"


# --- Building blocks (each takes a complete hourly UTC index) ---
def complete_hourly_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Full hourly UTC range spanning ``index`` (so lags are true time offsets)."""
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("index must be tz-aware UTC")
    return pd.date_range(index.min(), index.max(), freq="h", tz="UTC", name="timestamp_utc")


def make_lag_features(load: pd.Series, lags: tuple[int, ...] = LAG_HOURS) -> pd.DataFrame:
    """``load_lag_k`` = value k hours before each row (requires a complete hourly index)."""
    return pd.DataFrame({f"load_lag_{k}": load.shift(k) for k in lags}, index=load.index)


def make_rolling_features(
    load: pd.Series,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    stats: tuple[str, ...] = ROLLING_STATS,
) -> pd.DataFrame:
    """Rolling stats of the *previous* ``w`` hours: window ends at t-1, never includes t."""
    past = load.shift(1)
    out: dict[str, pd.Series] = {}
    for w in windows:
        roll = past.rolling(window=w, min_periods=max(1, int(w * ROLLING_MIN_PERIODS_FRAC)))
        for stat in stats:
            out[f"load_roll_{stat}_{w}"] = getattr(roll, stat)()
    return pd.DataFrame(out, index=load.index)


def german_holidays(years: list[int] | range) -> holidays.HolidayBase:
    """Nationwide German public holidays (no Bundesland-specific ones, see module doc)."""
    return holidays.country_holidays("DE", years=list(years))


def make_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features from the *local* (Europe/Berlin) clock + cyclical encodings."""
    local = index.tz_convert(LOCAL_TZ)
    hol = german_holidays(range(local.year.min(), local.year.max() + 1))
    local_dates = local.date

    df = pd.DataFrame(index=index)
    df["hour"] = local.hour
    df["day_of_week"] = local.dayofweek  # 0 = Monday
    df["month"] = local.month
    df["day_of_year"] = local.dayofyear
    df["is_weekend"] = (local.dayofweek >= 5).astype("int8")
    df["is_holiday"] = np.fromiter((d in hol for d in local_dates), dtype="int8", count=len(index))

    # Cyclical encodings so 23:00 is next to 00:00, Sunday next to Monday, Dec next to Jan.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def make_weather_features(
    weather_avg: pd.DataFrame | None, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Align national-average weather to ``index`` (NaN where no weather is available).

    At serving time these columns are filled from *weather forecasts* for hour t.
    """
    cols = list(WEATHER_COLUMNS)
    if weather_avg is None or weather_avg.empty:
        return pd.DataFrame(np.nan, index=index, columns=cols)
    w = weather_avg.set_index(pd.to_datetime(weather_avg["timestamp_utc"], utc=True))
    return w.reindex(index)[cols]


# --- Assembly ---
def build_feature_frame(
    load: pd.DataFrame,
    weather_avg: pd.DataFrame | None = None,
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    """Assemble the modeling frame from ``[timestamp_utc, load_mw]`` + averaged weather.

    With ``dropna=True`` (training) rows lacking the target or any lag/rolling
    feature are removed - i.e. the first 168 hours and the hours right after a
    gap. Weather NaNs are kept (tree models handle them; the serving phase
    decides how to fill).
    """
    if load.empty:
        raise ValueError("load frame is empty")
    series = (
        load.assign(timestamp_utc=pd.to_datetime(load["timestamp_utc"], utc=True))
        .drop_duplicates("timestamp_utc")
        .set_index("timestamp_utc")[TARGET]
        .astype("float64")
        .sort_index()
    )
    index = complete_hourly_index(series.index)
    series = series.reindex(index)

    parts = [
        series.rename(TARGET).to_frame(),
        make_lag_features(series),
        make_rolling_features(series),
        make_calendar_features(index),
        make_weather_features(weather_avg, index),
    ]
    df = pd.concat(parts, axis=1)
    df.index.name = "timestamp_utc"

    if dropna:
        required = [TARGET, *lag_and_rolling_columns()]
        before = len(df)
        df = df.dropna(subset=required)
        log.info(
            "feature frame: %d rows (dropped %d without full history)", len(df), before - len(df)
        )
    return df


def lag_and_rolling_columns() -> list[str]:
    cols = [f"load_lag_{k}" for k in LAG_HOURS]
    cols += [f"load_roll_{s}_{w}" for w in ROLLING_WINDOWS for s in ROLLING_STATS]
    return cols


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column except the target."""
    return [c for c in df.columns if c != TARGET]


# --- Time-based splits (never random - a random split leaks future information into the past) ---
def expanding_window_splits(
    df: pd.DataFrame,
    *,
    n_splits: int,
    val_hours: int,
    min_train_hours: int = 24 * 365,
    gap_hours: int = 0,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window time-series CV: yields ``(train, val)`` pairs, oldest first.

    The last ``n_splits`` blocks of ``val_hours`` are validation folds; each fold
    trains on *everything* before it (minus an optional ``gap_hours`` buffer).
    """
    idx = df.index
    end = idx.max() + pd.Timedelta(hours=1)
    fold_starts = [end - pd.Timedelta(hours=val_hours * (n_splits - i)) for i in range(n_splits)]
    for start in fold_starts:
        stop = start + pd.Timedelta(hours=val_hours)
        train_stop = start - pd.Timedelta(hours=gap_hours)
        train = df[idx < train_stop]
        val = df[(idx >= start) & (idx < stop)]
        if len(train) < min_train_hours:
            raise ValueError(
                f"fold starting {start} has only {len(train)} training hours (< {min_train_hours})"
            )
        if not train.index.max() < val.index.min():
            raise ValueError(f"fold starting {start} overlaps its training data")
        yield train, val


# --- I/O ---
def load_inputs(
    engine: Engine,
    *,
    source: str = "smard",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read load + per-city weather from the DB; return (load, national-average weather)."""
    load = db.read_load(engine, source=source, start=start, end=end)
    weather = db.read_table(engine, db.WeatherHourly, start=start, end=end)
    weather_avg = national_average(weather)
    log.info(
        "inputs: %d load rows, %d weather rows -> %d averaged",
        len(load),
        len(weather),
        len(weather_avg),
    )
    return load, weather_avg


def build_features(
    engine: Engine | None = None,
    *,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
    output: Path | None = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Full pipeline: DB -> feature frame (-> parquet if ``output`` is given)."""
    engine = engine or db.get_engine()
    start_ts = db.to_utc(start) if start is not None else db.to_utc(get_settings().data_start_date)
    end_ts = db.to_utc(end) if end is not None else None
    load, weather_avg = load_inputs(engine, start=start_ts, end=end_ts)
    df = build_feature_frame(load, weather_avg)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output)
        log.info("saved -> %s (%d rows x %d columns)", output, len(df), df.shape[1])
    return df


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(prog="python -m src.features.build_features")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default DATA_START_DATE)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: everything)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="parquet path")
    p.add_argument("--no-persist", action="store_true", help="do not write parquet")
    args = p.parse_args(argv)

    df = build_features(
        start=args.start, end=args.end, output=None if args.no_persist else args.output
    )
    log.info("feature columns (%d): %s", len(feature_columns(df)), ", ".join(feature_columns(df)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
