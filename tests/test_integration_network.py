"""Real-network smoke tests against SMARD and Open-Meteo.

Marked ``integration``: excluded in CI (``pytest -m "not integration"``), run
locally with ``make test-integration`` when you want to confirm the live APIs
still answer in the shape the clients expect.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.smard_client import SmardClient
from src.data.validation import validate_load, validate_weather
from src.data.weather_client import GERMAN_CITIES, WeatherClient

pytestmark = pytest.mark.integration


def test_smard_one_week_of_real_german_load() -> None:
    df = SmardClient().fetch_load("2024-01-08", "2024-01-14T23:00")
    report = validate_load(df, min_rows=24 * 7)
    assert report.ok
    assert len(df) == 24 * 7
    assert 45_000 < df["load_mw"].mean() < 75_000  # a winter week in Germany


def test_smard_dst_week_is_contiguous_in_utc() -> None:
    df = SmardClient().fetch_load("2024-03-30T23:00", "2024-04-01T00:00")
    assert set(df["timestamp_utc"].diff().dropna()) == {pd.Timedelta(hours=1)}


def test_open_meteo_archive_two_days_one_city() -> None:
    client = WeatherClient(cities=GERMAN_CITIES[:1])
    df = client.fetch_historical("2024-01-08", "2024-01-09")
    validate_weather(df, min_rows=48)
    assert len(df) == 48
    assert df["temperature_2m"].between(-30, 40).all()


def test_open_meteo_forecast_one_city() -> None:
    client = WeatherClient(cities=GERMAN_CITIES[:1])
    df = client.fetch_forecast(past_days=1, forecast_days=2)
    validate_weather(df, min_rows=48)
    assert df["timestamp_utc"].max() > pd.Timestamp.now(tz="UTC")
