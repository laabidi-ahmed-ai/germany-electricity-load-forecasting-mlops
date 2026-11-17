"""Open-Meteo client - hourly weather for representative German cities, no key needed.

Two endpoints feed the system:

* archive (``archive-api.open-meteo.com/v1/archive``) - reanalysis history,
  available from 1940 up to a few days ago. Used for the backfill.
* forecast (``api.open-meteo.com/v1/forecast``) - recent observations plus the
  next days' forecast (``past_days`` / ``forecast_days``). Used for incremental
  updates and for building the day-ahead feature rows. Tried twice with short timeouts
  (``OPEN_METEO_PRIMARY_RETRY``); if that fails, the same query goes to the ``/forecast``
  endpoint of the previous-runs host (below), which answered with identical values when
  compared on 17 Nov 2026 (a comparison, not a documented guarantee). From 16 Nov 2026
  GitHub runners at times could not complete a handshake with api.open-meteo.com at all,
  while the archive host, the same server as previous-runs, answered.

Both are requested with ``timezone=UTC`` so timestamps are DST-unambiguous, and with
every city in one request: comma-separated ``latitude`` / ``longitude``, answered by a JSON
list in the same order (a single location is answered by one object). Each request uses
``OPEN_METEO_RETRY`` (the first try of the forecast, a shorter policy): Open-Meteo answers in about a second, but a few times a day a request
stalls without a byte, so the client stops waiting after 20 s and tries again.
Rows are returned in long format, one row per ``(timestamp_utc, city)``, tagged
with ``source`` = ``"archive"`` or ``"forecast"`` so the storage layer can let
archive values supersede forecast values for the same hour.

A third endpoint serves two purposes:

* previous runs (``previous-runs-api.open-meteo.com/v1/forecast``) - archived forecasts at
  fixed lead offsets: ``<var>_previous_day<N>`` at hour t is the value of the newest model
  run initialised at least N x 24 h before t. Requested without ``models=``, i.e. the same
  default model mix (best_match) as the live forecast. Read by the fair backtest
  (``src.monitoring.fair_backtest``), and nothing from that is stored.
* the same endpoint with the plain variable names and ``past_days`` / ``forecast_days`` is
  the fallback of ``fetch_forecast``; those rows are stored as ``source="forecast"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

import pandas as pd
import requests

from config.settings import get_settings
from src.data._http import RetryPolicy, get_json, make_session
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

CITY_COLUMNS = ["timestamp_utc", "city", "source", *WEATHER_VARS]
# National averages (``national_average``) - the weather features of the model.
NATIONAL_COLUMNS: tuple[str, ...] = tuple(f"{var}_de_avg" for var in WEATHER_VARS)

SOURCE_ARCHIVE = "archive"
SOURCE_FORECAST = "forecast"

# Short read timeout, more attempts, capped waits: waits 3, 9, 27, 30, 30 s, and a request
# that always stalls gives up after about 4.7 min (``worst_case_seconds``), well inside the
# 30 min of the ingest job and the 40 min of the forecast job. A healthy answer, even the full five-city
# backfill, starts in under a second; a 60 s read timeout only made a stall cost minutes.
OPEN_METEO_RETRY = RetryPolicy(retries=6, backoff=3.0, max_wait=30.0, timeout=(10.0, 20.0))

# api.open-meteo.com before the fallback: two quick tries, 1 s apart. The handshake takes
# well under a second when the host is reachable; when a runner cannot reach it, retrying in
# the same run does not help (ingest #1071, 17 Nov 2026), so the fallback should start soon.
# Worst case 2 x (5 + 10) + 1 = 31 s; with the fallback's 279 s, the forecast job's three
# Open-Meteo calls take at most 899 s, inside half its 40 minutes (1200 s; test_weather_client).
OPEN_METEO_PRIMARY_RETRY = RetryPolicy(retries=2, backoff=1.0, timeout=(5.0, 10.0))


def forecast_worst_case_seconds() -> float:
    """``fetch_forecast`` when both hosts stall on every attempt."""
    return OPEN_METEO_PRIMARY_RETRY.worst_case_seconds() + OPEN_METEO_RETRY.worst_case_seconds()


# Previous Runs: lead offsets in whole days, and the long-format columns of a fetch.
PREVIOUS_RUN_OFFSETS: tuple[int, ...] = (1, 2)
PREVIOUS_RUN_COLUMNS = ["timestamp_utc", "city", "offset_days", *WEATHER_VARS]


def previous_run_variable(var: str, offset_days: int) -> str:
    """Open-Meteo's name for ``var`` as forecast ``offset_days`` x 24 h before valid time."""
    return f"{var}_previous_day{offset_days}"


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
    return df.loc[:, CITY_COLUMNS]


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
        previous_runs_url: str | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session or make_session()
        self.cities = cities
        self.archive_url = archive_url or settings.open_meteo_archive_url
        self.forecast_url = (forecast_base_url or settings.open_meteo_base_url).rstrip("/")
        self.forecast_url += "/forecast"
        self.previous_runs_url = previous_runs_url or settings.open_meteo_previous_runs_url
        # The setting already names the previous-runs host's /forecast endpoint.
        self.forecast_fallback_url = self.previous_runs_url
        # Set by each fetch_forecast that returns data: True if the fallback host served it.
        self.forecast_fallback_used: bool | None = None

    def _get_all_cities(
        self, url: str, params: dict, *, policy: RetryPolicy = OPEN_METEO_RETRY
    ) -> list[dict]:
        """One request for every city; the answers in the order of ``self.cities``.

        Open-Meteo returns a list in request order for several locations, each item but the
        first carrying its 0-based ``location_id``, and a single object for one location.
        """
        params = {
            "latitude": ",".join(str(c.latitude) for c in self.cities),
            "longitude": ",".join(str(c.longitude) for c in self.cities),
            **params,
        }
        payload = get_json(self.session, url, params, policy=policy)
        items = payload if isinstance(payload, list) else [payload]
        if len(items) != len(self.cities):
            raise ValueError(
                f"Open-Meteo returned {len(items)} locations for {len(self.cities)} cities"
            )
        for position, item in enumerate(items):
            if item.get("location_id", 0) != position:
                raise ValueError(
                    f"Open-Meteo answer {position} is location_id {item.get('location_id')}"
                )
        return items

    # -------------------------------------------------------------- history
    def fetch_historical(self, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Hourly reanalysis weather for ``[start, end]`` (dates, inclusive) per city."""
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date() if end is not None else pd.Timestamp.now(tz="UTC").date()
        if end_d < start_d:
            raise ValueError(f"end ({end_d}) is before start ({start_d})")

        if not self.cities:
            return empty_weather_frame()
        log.info("Open-Meteo archive: %d cities %s -> %s", len(self.cities), start_d, end_d)
        params = {
            "start_date": start_d.isoformat(),
            "end_date": end_d.isoformat(),
            "hourly": ",".join(WEATHER_VARS),
            "timezone": "UTC",
        }
        payloads = self._get_all_cities(self.archive_url, params)
        df = _finalise(
            [
                _hourly_payload_to_frame(payload, city, SOURCE_ARCHIVE)
                for city, payload in zip(self.cities, payloads, strict=True)
            ]
        )
        log.info("Open-Meteo archive: %d rows for %d cities", len(df), len(self.cities))
        return df

    # ------------------------------------------------------------- forecast
    def fetch_forecast(self, *, past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
        """Recent observations (``past_days``) plus the next ``forecast_days`` per city.

        Asks api.open-meteo.com (``OPEN_METEO_PRIMARY_RETRY``), then, if that fails for any
        reason, the previous-runs host (``OPEN_METEO_RETRY``), whose answer is checked the
        same way. ``forecast_fallback_used`` records which one served; if both fail, the
        fallback's error is raised.
        """
        self.forecast_fallback_used = None
        if not self.cities:
            return empty_weather_frame()
        log.info(
            "Open-Meteo forecast: %d cities (past_days=%d, forecast_days=%d)",
            len(self.cities),
            past_days,
            forecast_days,
        )
        params = {
            "hourly": ",".join(WEATHER_VARS),
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "UTC",
        }
        try:
            url = self.forecast_url
            payloads = self._get_all_cities(url, params, policy=OPEN_METEO_PRIMARY_RETRY)
            fallback = False
        except Exception as err:
            url = self.forecast_fallback_url
            log.warning(
                "Open-Meteo forecast: %s failed (%s) - asking %s instead",
                _host(self.forecast_url),
                err,
                _host(url),
            )
            payloads = self._get_all_cities(url, params)
            fallback = True
        df = _finalise(
            [
                _hourly_payload_to_frame(payload, city, SOURCE_FORECAST)
                for city, payload in zip(self.cities, payloads, strict=True)
            ]
        )
        self.forecast_fallback_used = fallback
        log.info(
            "Open-Meteo forecast: %d rows for %d cities, served by %s%s",
            len(df),
            len(self.cities),
            _host(url),
            " (fallback)" if fallback else "",
        )
        return df

    # -------------------------------------------------------- previous runs
    def fetch_previous_runs(
        self,
        start: DateLike,
        end: DateLike,
        *,
        offsets: tuple[int, ...] = PREVIOUS_RUN_OFFSETS,
    ) -> pd.DataFrame:
        """Archived forecasts for ``[start, end]`` (UTC dates, inclusive) per city and offset.

        Long format (``PREVIOUS_RUN_COLUMNS``): one row per ``(timestamp_utc, city,
        offset_days)``, where the weather columns hold the forecast made ``offset_days``
        x 24 h (or more) before the hour. Hours an offset has no value for are left out.
        """
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date()
        if end_d < start_d:
            raise ValueError(f"end ({end_d}) is before start ({start_d})")
        if not offsets or min(offsets) < 1:
            raise ValueError(f"offsets must be whole days >= 1, got {offsets}")

        hourly = [previous_run_variable(v, n) for n in offsets for v in WEATHER_VARS]
        frames = []
        for city in self.cities:
            log.info("Open-Meteo previous runs: %s %s -> %s", city.name, start_d, end_d)
            params = {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "hourly": ",".join(hourly),
                "timezone": "UTC",
            }
            payload = get_json(self.session, self.previous_runs_url, params)
            data = payload["hourly"]
            times = pd.to_datetime(data["time"], utc=True)
            missing = [None] * len(times)  # a series the API left out entirely = no values
            for n in offsets:
                df = pd.DataFrame(
                    {v: data.get(previous_run_variable(v, n), missing) for v in WEATHER_VARS}
                ).astype("float64")
                df.insert(0, "timestamp_utc", times)
                df.insert(1, "city", city.name)
                df.insert(2, "offset_days", n)
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=PREVIOUS_RUN_COLUMNS)
        out = pd.concat(frames, ignore_index=True).dropna(subset=list(WEATHER_VARS), how="all")
        out = (
            out.drop_duplicates(subset=["timestamp_utc", "city", "offset_days"], keep="last")
            .sort_values(["offset_days", "timestamp_utc", "city"])
            .reset_index(drop=True)
        )
        log.info(
            "Open-Meteo previous runs: %d rows for %d cities x %d offsets",
            len(out),
            len(self.cities),
            len(offsets),
        )
        return out.loc[:, PREVIOUS_RUN_COLUMNS]


def _host(url: str) -> str:
    return urlsplit(url).netloc or url


def national_average(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-city rows into one row per hour (mean across cities).

    Output columns: ``timestamp_utc`` plus ``NATIONAL_COLUMNS`` (``<var>_de_avg``).
    """
    if df.empty:
        out = pd.DataFrame({"timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]")})
        for col in NATIONAL_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out
    avg = df.groupby("timestamp_utc", sort=True)[list(WEATHER_VARS)].mean()
    avg.columns = list(NATIONAL_COLUMNS)
    return avg.reset_index()
