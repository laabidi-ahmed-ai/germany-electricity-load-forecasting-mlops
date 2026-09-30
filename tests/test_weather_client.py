"""Open-Meteo client tests - all offline via ``FakeSession``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.weather_client import (
    GERMAN_CITIES,
    WEATHER_COLUMNS,
    WEATHER_VARS,
    City,
    WeatherClient,
    national_average,
)
from tests.conftest import FakeSession, open_meteo_route

ARCHIVE = "https://archive.test/v1/archive"
FORECAST_BASE = "https://forecast.test/v1"

TWO_CITIES = (City("A", 50.0, 8.0), City("B", 52.0, 13.0))


def make_client(cities=TWO_CITIES, start="2021-03-27", end="2021-03-28T23:00"):
    session = FakeSession(
        routes={
            ARCHIVE: open_meteo_route(start, end),
            f"{FORECAST_BASE}/forecast": open_meteo_route(start, end),
        }
    )
    client = WeatherClient(
        session=session, cities=cities, archive_url=ARCHIVE, forecast_base_url=FORECAST_BASE
    )
    return client, session


def test_default_cities_are_the_five_representative_ones() -> None:
    assert [c.name for c in GERMAN_CITIES] == [
        "Berlin",
        "Hamburg",
        "Munich",
        "Cologne",
        "Frankfurt",
    ]


def test_fetch_historical_long_format_utc() -> None:
    client, session = make_client()
    df = client.fetch_historical("2021-03-27", "2021-03-28")

    assert list(df.columns) == WEATHER_COLUMNS
    assert len(df) == 48 * 2
    assert set(df["city"]) == {"A", "B"}
    assert (df["source"] == "archive").all()
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    assert not df.duplicated(subset=["timestamp_utc", "city"]).any()

    # One request per city, always asking for UTC and the full variable list.
    assert len(session.calls) == 2
    for _, params in session.calls:
        assert params["timezone"] == "UTC"
        assert params["hourly"] == ",".join(WEATHER_VARS)
        assert params["start_date"] == "2021-03-27"
        assert params["end_date"] == "2021-03-28"


def test_fetch_forecast_tags_source_and_passes_days() -> None:
    client, session = make_client()
    df = client.fetch_forecast(past_days=2, forecast_days=3)
    assert (df["source"] == "forecast").all()
    assert len(df) == 48 * 2
    for url, params in session.calls:
        assert url == f"{FORECAST_BASE}/forecast"
        assert params["past_days"] == 2
        assert params["forecast_days"] == 3


def test_national_average_collapses_cities() -> None:
    client, _ = make_client()
    df = client.fetch_historical("2021-03-27", "2021-03-28")
    avg = national_average(df)

    assert len(avg) == 48
    assert list(avg.columns) == ["timestamp_utc", *[f"{v}_de_avg" for v in WEATHER_VARS]]
    # Fake payload sets temperature = 10 + lat/10 + cycle; the cycle cancels in the
    # difference between cities, so the mean is exactly the mean of the offsets.
    first_hour = df[df["timestamp_utc"] == df["timestamp_utc"].min()]
    expected = first_hour["temperature_2m"].mean()
    assert avg["temperature_2m_de_avg"].iloc[0] == pytest.approx(expected)


def test_national_average_of_empty_frame_has_expected_columns() -> None:
    from src.data.weather_client import empty_weather_frame

    avg = national_average(empty_weather_frame())
    assert avg.empty
    assert "temperature_2m_de_avg" in avg.columns


def test_end_before_start_raises() -> None:
    client, _ = make_client()
    with pytest.raises(ValueError, match="before start"):
        client.fetch_historical("2021-04-01", "2021-03-01")


def test_rows_with_all_variables_missing_are_dropped() -> None:
    client, session = make_client()

    def route(url, params):
        payload = open_meteo_route("2021-03-27", "2021-03-27T05:00")(url, params)
        for var in WEATHER_VARS:  # blank out the last hour entirely
            payload["hourly"][var][-1] = None
        return payload

    session.routes[ARCHIVE] = route
    df = client.fetch_historical("2021-03-27", "2021-03-27")
    assert len(df) == 5 * 2
    assert df["timestamp_utc"].max() == pd.Timestamp("2021-03-27T04:00Z")
