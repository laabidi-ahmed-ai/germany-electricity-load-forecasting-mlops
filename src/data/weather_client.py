"""Open-Meteo client - hourly weather for representative German cities, no key needed.

Two endpoints are used:

* archive (``archive-api.open-meteo.com/v1/archive``) - reanalysis history,
  available from 1940 up to a few days ago. Used for the backfill.
* forecast (``api.open-meteo.com/v1/forecast``) - recent observations plus the
  next days' forecast (``past_days`` / ``forecast_days``). Used for incremental
  updates and for building the day-ahead feature rows.

Both are requested with ``timezone=UTC`` so timestamps are DST-unambiguous.
Rows are returned in long format, one row per ``(timestamp_utc, city)``, tagged
with ``source`` = ``"archive"`` or ``"forecast"`` so the storage layer can let
archive values supersede forecast values for the same hour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

from config.settings import get_settings
from src.data._http import get_json, make_session
from src.data.db import DateLike

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float


# Representative German cities; their average approximates the national driver of
# demand. The *target* is always the national load (README §7), weather is input.
GERMAN_CITIES: tuple[City, ...] = (
    City("Berlin", 52.5200, 13.4050),
    City("Hamburg", 53.5511, 10.0000),
    City("Munich", 48.1351, 11.5820),
    City("Cologne", 50.9375, 6.9603),
    City("Frankfurt", 50.1109, 8.6821),
)

WEATHER_VARS: tuple[str, ...] = (
    "temperature_2m",  # °C
    "wind_speed_10m",  # km/h
    "shortwave_radiation",  # W/m²
    "cloud_cover",  # %
    "relative_humidity_2m",  # %
)

WEATHER_COLUMNS = ["timestamp_utc", "city", "source", *WEATHER_VARS]

SOURCE_ARCHIVE = "archive"
SOURCE_FORECAST = "forecast"


def empty_weather_frame() -> pd.DataFrame:
    cols: dict[str, pd.Series] = {
        "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        "city": pd.Series(dtype="str"),
        "source": pd.Series(dtype="str"),
    }
    for var in WEATHER_VARS:
        cols[var] = pd.Series(dtype="float64")
    return pd.DataFrame(cols)


def _hourly_payload_to_frame(payload: dict, city: City, source: str) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({var: hourly.get(var) for var in WEATHER_VARS})
    df["timestamp_utc"] = pd.to_datetime(hourly["time"], utc=True)
    df["city"] = city.name
    df["source"] = source
    return df.loc[:, WEATHER_COLUMNS]


def _finalise(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return empty_weather_frame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=list(WEATHER_VARS), how="all")
    return (
        df.drop_duplicates(subset=["timestamp_utc", "city"], keep="last")
        .sort_values(["timestamp_utc", "city"])
        .reset_index(drop=True)
    )


class WeatherClient:
    """Fetch hourly weather for a set of cities from Open-Meteo."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        cities: tuple[City, ...] = GERMAN_CITIES,
        archive_url: str | None = None,
        forecast_base_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session or make_session()
        self.cities = cities
        self.archive_url = archive_url or settings.open_meteo_archive_url
        self.forecast_url = (forecast_base_url or settings.open_meteo_base_url).rstrip("/")
        self.forecast_url += "/forecast"

    # -------------------------------------------------------------- history
    def fetch_historical(self, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Hourly reanalysis weather for ``[start, end]`` (dates, inclusive) per city."""
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date() if end is not None else pd.Timestamp.now(tz="UTC").date()
        if end_d < start_d:
            raise ValueError(f"end ({end_d}) is before start ({start_d})")

        frames = []
        for city in self.cities:
            log.info("Open-Meteo archive: %s %s -> %s", city.name, start_d, end_d)
            params = {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "hourly": ",".join(WEATHER_VARS),
                "timezone": "UTC",
            }
            payload = get_json(self.session, self.archive_url, params)
            frames.append(_hourly_payload_to_frame(payload, city, SOURCE_ARCHIVE))

        df = _finalise(frames)
        log.info("Open-Meteo archive: %d rows for %d cities", len(df), len(self.cities))
        return df

    # ------------------------------------------------------------- forecast
    def fetch_forecast(self, *, past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
        """Recent observations (``past_days``) plus the next ``forecast_days`` per city."""
        frames = []
        for city in self.cities:
            log.info(
                "Open-Meteo forecast: %s (past_days=%d, forecast_days=%d)",
                city.name,
                past_days,
                forecast_days,
            )
            params = {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "hourly": ",".join(WEATHER_VARS),
                "past_days": past_days,
                "forecast_days": forecast_days,
                "timezone": "UTC",
            }
            payload = get_json(self.session, self.forecast_url, params)
            frames.append(_hourly_payload_to_frame(payload, city, SOURCE_FORECAST))

        df = _finalise(frames)
        log.info("Open-Meteo forecast: %d rows for %d cities", len(df), len(self.cities))
        return df


def national_average(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-city rows into one row per hour (mean across cities).

    Output columns: ``timestamp_utc`` plus ``<var>_de_avg`` for each weather variable.
    """
    if df.empty:
        out = pd.DataFrame({"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        for var in WEATHER_VARS:
            out[f"{var}_de_avg"] = pd.Series(dtype="float64")
        return out
    avg = df.groupby("timestamp_utc", sort=True)[list(WEATHER_VARS)].mean()
    avg = avg.rename(columns={var: f"{var}_de_avg" for var in WEATHER_VARS})
    return avg.reset_index()
