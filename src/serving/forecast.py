"""Day-ahead forecast generation shared by the API and the batch job.

Serving-time features are produced by **the same code path as training**:
``build_feature_frame`` (lags, rolling stats, calendar, weather join) followed by
``select_features(DAY_AHEAD)``. The only difference is the *content* of the weather
columns - Open-Meteo *forecast* values for hours that have no reanalysis yet -
which is the intended, documented train/serve skew.

Forecast window
---------------
With a 24-hour information cutoff (``src.features.horizons``), every target hour
*t* needs load observed at *t-24h*. Given the latest stored actual load at *L*,
the honest day-ahead window is ``(L, L + 24h]``: all 24 rows have fully observed
``load_lag_24 / 48 / 168``. We deliberately do **not** feed predictions back in
as lags to reach further (that would be a second, unintended skew).

Pipeline per request / batch run:

1. read the last ``HISTORY_HOURS`` of actual load from the DB (enough for lag_168)
2. make sure per-city weather covers the target hours (fetch the Open-Meteo forecast
   and upsert it - archive rows always win - if not)
3. append the 24 target hours with NaN load, run ``build_feature_frame(dropna=False)``
4. keep the target rows, ``select_features(DAY_AHEAD)``, check they equal the
   model's training features **exactly**, predict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.data import db
from src.data.weather_client import WeatherClient, national_average
from src.features.build_features import TARGET, build_feature_frame
from src.features.horizons import DAY_AHEAD, most_recent_load_lag, select_features
from src.models.registry import LoadedModel

log = logging.getLogger(__name__)

DAY_AHEAD_HOURS = 24
HISTORY_HOURS = 24 * 10  # lag_168 for the last target hour needs L + 24 - 168 = L - 144h
LOAD_SOURCE = "smard"


class NoDataError(RuntimeError):
    """Raised when the database lacks the actual load needed to build features."""


@dataclass
class ForecastResult:
    issued_at: pd.Timestamp
    last_actual: pd.Timestamp
    target_start: pd.Timestamp
    target_end: pd.Timestamp
    model_name: str
    model_version: str
    horizon: str
    features: list[str]
    frame: pd.DataFrame  # index timestamp_utc; columns: forecast_mw + the served features

    @property
    def forecast(self) -> pd.DataFrame:
        return self.frame[["forecast_mw"]].reset_index()

    def to_records(self) -> list[dict]:
        return [
            {"timestamp_utc": ts.isoformat(), "forecast_mw": round(float(v), 1)}
            for ts, v in self.frame["forecast_mw"].items()
        ]


# --------------------------------------------------------------------------- #
# Feature reconstruction
# --------------------------------------------------------------------------- #
def forecast_window(last_actual: pd.Timestamp, hours: int = DAY_AHEAD_HOURS) -> pd.DatetimeIndex:
    """Target hours ``(last_actual, last_actual + hours]``; capped at the horizon cutoff."""
    if hours < 1 or hours > DAY_AHEAD.min_lag_hours:
        raise ValueError(
            f"hours must be in [1, {DAY_AHEAD.min_lag_hours}] for an honest day-ahead forecast"
        )
    start = last_actual + pd.Timedelta(hours=1)
    return pd.date_range(start, periods=hours, freq="h", tz="UTC", name="timestamp_utc")


def ensure_weather(
    engine: Engine,
    target: pd.DatetimeIndex,
    weather_client: WeatherClient | None,
    *,
    n_cities: int | None = None,
) -> None:
    """Fetch + upsert the Open-Meteo forecast if the DB lacks weather for any target hour."""
    have = db.read_table(engine, db.WeatherHourly, start=target.min(), end=target.max())
    expected_cities = n_cities or (len(weather_client.cities) if weather_client else 0)
    complete = (
        not have.empty
        and have.groupby("timestamp_utc").size().reindex(target).fillna(0).ge(expected_cities).all()
    )
    if complete:
        return
    if weather_client is None:
        log.warning(
            "weather incomplete for %s -> %s and no weather client; using NaN",
            target.min(),
            target.max(),
        )
        return
    days_ahead = (
        int(np.ceil((target.max() - pd.Timestamp.now(tz="UTC")) / pd.Timedelta(days=1))) + 1
    )
    fresh = weather_client.fetch_forecast(past_days=2, forecast_days=max(1, min(days_ahead, 7)))
    n = db.upsert_dataframe(engine, db.WeatherHourly, fresh, update_where=db.weather_update_where)
    log.info("weather forecast refreshed: %d rows upserted", n)


def build_serving_features(
    engine: Engine,
    *,
    weather_client: WeatherClient | None = None,
    hours: int = DAY_AHEAD_HOURS,
    as_of: pd.Timestamp | None = None,
    load_source: str = LOAD_SOURCE,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return ``(full feature frame for the target hours, last_actual)``.

    ``as_of`` pretends the latest known load is at/before that time (for backtests);
    default is the real latest actual in the database.
    """
    last_actual = db.latest_timestamp(engine, db.LoadActual, source=load_source)
    if last_actual is None:
        raise NoDataError("no actual load in the database - run the ingestion first")
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
        last_actual = min(last_actual, as_of.floor("h"))

    target = forecast_window(last_actual, hours)
    hist_start = last_actual - pd.Timedelta(hours=HISTORY_HOURS)
    history = db.read_load(engine, source=load_source, start=hist_start, end=last_actual)
    if history.empty or history["timestamp_utc"].max() != last_actual:
        raise NoDataError(f"could not read actual load up to {last_actual}")

    ensure_weather(engine, target, weather_client)
    weather = db.read_table(engine, db.WeatherHourly, start=hist_start, end=target.max())
    weather_avg = national_average(weather)

    # History + the target hours with an unknown target -> same code path as training.
    load = pd.concat(
        [history, pd.DataFrame({"timestamp_utc": target, TARGET: np.nan})], ignore_index=True
    )
    frame = build_feature_frame(load, weather_avg, dropna=False)
    served = frame.loc[target]

    # Every day-ahead feature must be fully observed (no NaN lags); weather may be NaN.
    lag_cols = [c for c in select_features(served.columns, DAY_AHEAD) if most_recent_load_lag(c)]
    if served[lag_cols].isna().any().any():
        bad = served[lag_cols].isna().sum()
        raise NoDataError(
            f"gaps in recent actual load break lag features: {bad[bad > 0].to_dict()}"
        )
    n_nan_weather = int(served.filter(like="_de_avg").isna().sum().sum())
    if n_nan_weather:
        log.warning("%d NaN weather values in served features (model handles NaN)", n_nan_weather)
    return served, last_actual


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
def make_day_ahead_forecast(
    engine: Engine,
    model: LoadedModel,
    *,
    weather_client: WeatherClient | None = None,
    hours: int = DAY_AHEAD_HOURS,
    as_of: pd.Timestamp | None = None,
) -> ForecastResult:
    served, last_actual = build_serving_features(
        engine, weather_client=weather_client, hours=hours, as_of=as_of
    )
    model.check_features(list(served.columns))  # exact train/serve feature parity
    X = served[model.features]
    pred = model.predict(X)
    if not np.isfinite(pred).all():
        raise RuntimeError("model produced non-finite predictions")

    frame = X.copy()
    frame.insert(0, "forecast_mw", pred)
    issued_at = pd.Timestamp(datetime.now(UTC))
    log.info(
        "day-ahead forecast %s -> %s by %s: mean %.0f MW",
        frame.index.min(),
        frame.index.max(),
        model.version_label,
        pred.mean(),
    )
    return ForecastResult(
        issued_at=issued_at,
        last_actual=last_actual,
        target_start=frame.index.min(),
        target_end=frame.index.max(),
        model_name=model.name,
        model_version=model.version,
        horizon=model.horizon.name,
        features=model.features,
        frame=frame,
    )


def store_forecast(engine: Engine, result: ForecastResult) -> int:
    """Idempotent upsert of the forecast rows into ``load_forecast_model``."""
    rows = pd.DataFrame(
        {
            "timestamp_utc": result.frame.index,
            "model_version": result.model_version,
            "model_name": result.model_name,
            "forecast_mw": result.frame["forecast_mw"].to_numpy(),
            "issued_at": result.issued_at,
        }
    )
    return db.upsert_dataframe(engine, db.LoadForecastModel, rows)
