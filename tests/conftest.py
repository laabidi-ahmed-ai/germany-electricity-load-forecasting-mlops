"""Shared fixtures: env isolation, a fake HTTP session, synthetic SMARD/Open-Meteo payloads,
and the synthetic load / weather / model stand-ins the modelling tests share.

No unit test here touches the network. Real-network tests live in
``tests/test_integration_network.py`` and carry the ``integration`` marker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pytest
import requests
from sqlalchemy.engine import Engine

from config.settings import Settings, get_settings
from src.data import db
from src.features.build_features import TARGET, WEATHER_COLUMNS, build_feature_frame
from src.features.horizons import DAY_AHEAD
from src.models.registry import LoadedModel

_ENV_KEYS = [
    "DATABASE_URL",
    "OPEN_METEO_BASE_URL",
    "MLFLOW_TRACKING_URI",
    "SMARD_REGION",
    "DATA_START_DATE",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a developer's real ``.env`` / env vars leak into tests."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Point pydantic-settings at a non-existent env file for the duration of the test.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- Fake HTTP ---
@dataclass
class FakeResponse:
    payload: Any
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> Any:
        return self.payload


@dataclass
class FakeSession:
    """Minimal stand-in for ``requests.Session`` with a URL -> payload routing table.

    Routes can be a payload, a callable ``(url, params) -> payload``, or a list of
    responses consumed in order (to simulate transient failures).
    """

    routes: dict[str, Any] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    @staticmethod
    def key(url: str, params: dict[str, Any] | None = None) -> str:
        return f"{url}?{urlencode(params)}" if params else url

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float = 0) -> Any:
        self.calls.append((url, params))
        full = self.key(url, params)
        route = self.routes.get(full, self.routes.get(url))
        if route is None:
            return FakeResponse({"error": f"no fake route for {full}"}, status_code=404)
        if isinstance(route, list):
            item = route.pop(0)
            return item if isinstance(item, FakeResponse) else FakeResponse(item)
        if callable(route):
            return FakeResponse(route(url, params))
        return FakeResponse(route)


# --- Synthetic SMARD data (modelled on the real DST week of 2021-03-28) ---
MS_PER_HOUR = 3_600_000

# Real week-file starts from SMARD around the spring DST switch:
#   Mon 2021-03-22 00:00 CET  = 2021-03-21T23:00Z  (167 hourly points: 23h Sunday)
#   Mon 2021-03-29 00:00 CEST = 2021-03-28T22:00Z  (168 hourly points)
DST_WEEK_STARTS_MS = [1_616_367_600_000, 1_616_968_800_000]
DST_WEEK_LENGTHS = [167, 168]


def smard_value(ts_ms: int) -> float:
    """A plausible German load curve (~55 GW mean, daily cycle)."""
    hour = (ts_ms // MS_PER_HOUR) % 24
    return 55_000 + 10_000 * math.sin((hour - 6) / 24 * 2 * math.pi)


FORECAST_FACTOR = 0.98  # fake official forecast = 98% of the actual load


def smard_week_payload(
    week_start_ms: int, n_points: int, *, null_from: int | None = None, factor: float = 1.0
) -> dict:
    series = []
    for i in range(n_points):
        ts = week_start_ms + i * MS_PER_HOUR
        val = None if (null_from is not None and i >= null_from) else smard_value(ts) * factor
        series.append([ts, val])
    return {"meta_data": None, "series": series}


def smard_routes(
    base_url: str, region: str, weeks: dict[int, int] | None = None, *, null_from: int | None = None
) -> dict[str, Any]:
    """Build fake routes for the SMARD index + weekly files.

    ``weeks`` maps ``week_start_ms -> number_of_points``.
    """
    if weeks is None:
        weeks = dict(zip(DST_WEEK_STARTS_MS, DST_WEEK_LENGTHS, strict=True))
    routes: dict[str, Any] = {}
    # 410 = actual load, 411 = official day-ahead forecast (same weeks, 98% of actual).
    for fid, factor in ((410, 1.0), (411, FORECAST_FACTOR)):
        routes[f"{base_url}/{fid}/{region}/index_hour.json"] = {"timestamps": sorted(weeks)}
        for start, n in weeks.items():
            routes[f"{base_url}/{fid}/{region}/{fid}_{region}_hour_{start}.json"] = (
                smard_week_payload(start, n, null_from=null_from, factor=factor)
            )
    return routes


# --- Synthetic Open-Meteo data ---
def open_meteo_payload(start: str, end: str, *, lat: float = 0.0) -> dict:
    times = pd.date_range(start, end, freq="h", tz="UTC")
    n = len(times)
    hours = np.arange(n)
    return {
        "latitude": lat,
        "longitude": 0.0,
        "timezone": "GMT",
        "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": (10 + lat / 10 + 5 * np.sin(hours / 24 * 2 * np.pi)).tolist(),
            "wind_speed_10m": (12 + np.cos(hours / 24 * 2 * np.pi)).tolist(),
            "shortwave_radiation": np.clip(300 * np.sin(hours / 24 * 2 * np.pi), 0, None).tolist(),
            "cloud_cover": (50 + 20 * np.sin(hours / 48 * 2 * np.pi)).tolist(),
            "relative_humidity_2m": (70 + 10 * np.cos(hours / 24 * 2 * np.pi)).tolist(),
        },
    }


def open_meteo_route(start: str, end: str) -> Callable[[str, dict[str, Any] | None], dict]:
    """Route that ignores the query string and returns a payload for [start, end]."""

    def _route(url: str, params: dict[str, Any] | None) -> dict:
        lat = float(params["latitude"]) if params else 0.0
        return open_meteo_payload(start, end, lat=lat)

    return _route


# --- Database ---
@pytest.fixture
def engine(tmp_path) -> Engine:
    eng = db.get_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    db.init_db(eng)
    return eng


@pytest.fixture
def load_frame() -> pd.DataFrame:
    """A clean 2-week hourly UTC load frame spanning the DST switch."""
    ts = pd.date_range("2021-03-21T23:00Z", periods=335, freq="h")
    values = [smard_value(int(t.timestamp() * 1000)) for t in ts]
    return pd.DataFrame({"timestamp_utc": ts, "load_mw": values})


# --- Synthetic modelling data shared by the model / serving / monitoring tests ---
FAST_LGBM = {"n_estimators": 60, "learning_rate": 0.2, "num_leaves": 15, "min_child_samples": 5}


def load_curve(n_hours: int, *, seed: int = 0, noise: float = 300.0) -> np.ndarray:
    """German-like hourly load (~55 GW): daily + weekly cycle plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    h = np.arange(n_hours)
    return (
        55_000
        + 10_000 * np.sin(2 * np.pi * (h - 6) / 24)
        + 3_000 * np.sin(2 * np.pi * h / 168)
        + rng.normal(0, noise, n_hours)
    )


def random_weather(timestamps: pd.Series, *, seed: int) -> pd.DataFrame:
    """National-average weather columns drawn from N(10, 5) - pure noise for the model."""
    rng = np.random.default_rng(seed)
    weather = pd.DataFrame({"timestamp_utc": timestamps})
    for col in WEATHER_COLUMNS:
        weather[col] = rng.normal(10, 5, len(timestamps))
    return weather


def synthetic_frame(days: int = 45, seed: int = 0) -> pd.DataFrame:
    """Load with daily + weekly cycles and a temperature effect, so a model can learn."""
    ts = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    h = np.arange(len(ts))
    temp = 5 + 10 * np.sin(2 * np.pi * h / (24 * 30)) + rng.normal(0, 1, len(ts))
    load = (
        55_000
        + 10_000 * np.sin(2 * np.pi * (h - 6) / 24)
        + 3_000 * np.sin(2 * np.pi * h / 168)
        - 300 * temp
        + rng.normal(0, 400, len(ts))
    )
    load_df = pd.DataFrame({"timestamp_utc": ts, TARGET: load})
    weather = pd.DataFrame({"timestamp_utc": ts})
    for col in WEATHER_COLUMNS:
        weather[col] = temp if col.startswith("temperature") else rng.normal(10, 5, len(ts))
    return build_feature_frame(load_df, weather)


class StubPyfunc:
    """Stands in for an MLflow pyfunc model: last week's load (+ ``bias``)."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["load_lag_168"].to_numpy() + self.bias


def fake_loaded_model(
    features: list[str],
    *,
    version: str = "1",
    bias: float = 0.0,
    pyfunc: Any = None,
    train_end: pd.Timestamp | None = None,
) -> LoadedModel:
    """A champion as ``registry.load_champion`` would return it, without MLflow."""
    return LoadedModel(
        name="germany-load-day-ahead",
        version=version,
        alias="champion",
        run_id=f"run-{version}",
        features=features,
        horizon=DAY_AHEAD,
        pyfunc=pyfunc if pyfunc is not None else StubPyfunc(bias),
        train_end=train_end,
    )
