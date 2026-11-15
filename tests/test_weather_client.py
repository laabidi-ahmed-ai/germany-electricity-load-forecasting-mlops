"""Open-Meteo client tests - all offline via ``FakeSession``."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
import requests
import yaml

import src.data.weather_client as wc
from config.settings import PROJECT_ROOT
from src.data import _http
from src.data.weather_client import (
    CITY_COLUMNS,
    GERMAN_CITIES,
    OPEN_METEO_RETRY,
    WEATHER_VARS,
    City,
    WeatherClient,
    national_average,
)
from tests.conftest import FakeSession, open_meteo_payload, open_meteo_route, previous_runs_route

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

    assert list(df.columns) == CITY_COLUMNS
    assert len(df) == 48 * 2
    assert set(df["city"]) == {"A", "B"}
    assert (df["source"] == "archive").all()
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    assert not df.duplicated(subset=["timestamp_utc", "city"]).any()

    # One request for both cities, asking for UTC and the full variable list.
    assert len(session.calls) == 1
    _, params = session.calls[0]
    assert params["latitude"] == "50.0,52.0" and params["longitude"] == "8.0,13.0"
    assert params["timezone"] == "UTC"
    assert params["hourly"] == ",".join(WEATHER_VARS)
    assert params["start_date"] == "2021-03-27"
    assert params["end_date"] == "2021-03-28"


def test_fetch_forecast_tags_source_and_passes_days() -> None:
    client, session = make_client()
    df = client.fetch_forecast(past_days=2, forecast_days=3)
    assert (df["source"] == "forecast").all()
    assert len(df) == 48 * 2
    assert set(df["city"]) == {"A", "B"}
    [(url, params)] = session.calls  # one request for both cities
    assert url == f"{FORECAST_BASE}/forecast"
    assert params["latitude"] == "50.0,52.0" and params["longitude"] == "8.0,13.0"
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
        payloads = open_meteo_route("2021-03-27", "2021-03-27T05:00")(url, params)
        for payload in payloads:
            for var in WEATHER_VARS:  # blank out the last hour entirely
                payload["hourly"][var][-1] = None
        return payloads

    session.routes[ARCHIVE] = route
    df = client.fetch_historical("2021-03-27", "2021-03-27")
    assert len(df) == 5 * 2
    assert df["timestamp_utc"].max() == pd.Timestamp("2021-03-27T04:00Z")


# --- One request for all cities ---
@pytest.mark.parametrize("method", ["fetch_historical", "fetch_forecast"])
def test_a_single_city_is_answered_by_one_object(method: str) -> None:
    client, session = make_client(cities=TWO_CITIES[:1])
    fetch = getattr(client, method)
    df = fetch("2021-03-27", "2021-03-28") if method == "fetch_historical" else fetch()

    [(_, params)] = session.calls
    assert params["latitude"] == "50.0" and params["longitude"] == "8.0"
    assert len(df) == 48 and set(df["city"]) == {"A"}


def test_answers_are_mapped_to_cities_by_position() -> None:
    """The cities are not in latitude order, so a mix-up would show in the values."""
    cities = (City("North", 54.0, 9.0), City("South", 47.5, 11.0), City("Mid", 51.0, 10.0))
    client, session = make_client(cities=cities)
    df = client.fetch_forecast()

    [(_, params)] = session.calls
    assert params["latitude"] == "54.0,47.5,51.0" and params["longitude"] == "9.0,11.0,10.0"
    first_hour = df[df["timestamp_utc"] == df["timestamp_utc"].min()].set_index("city")
    # The fake sets temperature = 10 + latitude / 10 + a daily cycle that is 0 at hour 0.
    for city in cities:
        assert first_hour.loc[city.name, "temperature_2m"] == pytest.approx(10 + city.latitude / 10)


def _answer(*latitudes: float, ids: list[int | None]) -> list[dict]:
    payloads = [open_meteo_payload("2021-03-27", "2021-03-27T05:00", lat=lat) for lat in latitudes]
    for payload, location_id in zip(payloads, ids, strict=True):
        if location_id is not None:
            payload["location_id"] = location_id
    return payloads


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (_answer(50.0, ids=[None]), "1 locations for 2 cities"),
        (_answer(50.0, 52.0, 51.0, ids=[None, 1, 2]), "3 locations for 2 cities"),
        (_answer(52.0, 50.0, ids=[1, None]), "answer 0 is location_id 1"),
    ],
)
def test_an_answer_that_does_not_match_the_cities_is_refused(answer, message) -> None:
    client, session = make_client()
    session.routes[f"{FORECAST_BASE}/forecast"] = lambda url, params: answer  # one response
    with pytest.raises(ValueError, match=message):
        client.fetch_forecast()


def test_no_cities_means_no_request() -> None:
    client, session = make_client(cities=())
    assert client.fetch_historical("2021-03-27", "2021-03-28").empty
    assert client.fetch_forecast().empty
    assert session.calls == []


# --- Stalls: short read timeout, capped retries (ingest #955, 13 Nov 2026) ---
@dataclass
class StallingSession(FakeSession):
    """Raises the first ``stalls`` exceptions, then serves the routes; records each timeout."""

    stalls: list[Exception] = field(default_factory=list)
    timeouts: list[Any] = field(default_factory=list)

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: Any = 0) -> Any:
        self.timeouts.append(timeout)
        if self.stalls:
            self.calls.append((url, params))
            raise self.stalls.pop(0)
        return super().get(url, params, timeout)


def stalled(n: int) -> list[Exception]:
    return [requests.ReadTimeout("Read timed out. (read timeout=20)")] * n


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record the waits between attempts instead of sleeping."""
    waits: list[float] = []
    monkeypatch.setattr(wc, "get_json", functools.partial(_http.get_json, sleep=waits.append))
    return waits


def stalling_client(n_stalls: int) -> tuple[WeatherClient, StallingSession]:
    session = StallingSession(
        routes={f"{FORECAST_BASE}/forecast": open_meteo_route("2021-03-27", "2021-03-28T23:00")},
        stalls=stalled(n_stalls),
    )
    client = WeatherClient(session=session, cities=TWO_CITIES, forecast_base_url=FORECAST_BASE)
    return client, session


def test_a_stalled_request_is_retried_and_the_success_is_logged(sleeps, caplog) -> None:
    client, session = stalling_client(n_stalls=1)
    with caplog.at_level(logging.INFO, logger="src.data._http"):
        df = client.fetch_forecast()

    assert len(df) == 48 * 2
    assert len(session.calls) == 2
    assert sleeps == [3.0]
    assert session.timeouts == [(10.0, 20.0)] * 2  # connect 10 s, read 20 s (was 60 s)
    assert "retrying in 3.0s" in caplog.text
    assert "request succeeded on attempt 2/6" in caplog.text


def test_a_request_that_keeps_stalling_gives_up_after_the_last_attempt(sleeps) -> None:
    client, session = stalling_client(n_stalls=6)
    with pytest.raises(RuntimeError, match="after 6 attempts: Read timed out"):
        client.fetch_forecast()

    assert len(session.calls) == 6
    assert sleeps == [3.0, 9.0, 27.0, 30.0, 30.0]  # capped at 30 s


def test_open_meteo_policy_worst_case() -> None:
    # 6 attempts x (10 s connect + 20 s read) + the waits above
    assert OPEN_METEO_RETRY.worst_case_seconds() == 6 * 30 + 99


@pytest.mark.parametrize(
    ("workflow", "open_meteo_calls"),
    [
        ("ingest.yml", 2),  # weather archive + weather forecast
        ("forecast.yml", 3),  # the same refresh, then the forecast's own weather refresh
        ("monitor_retrain.yml", 2),  # the same refresh
    ],
)
def test_open_meteo_stalls_fit_well_inside_the_job_timeout(
    workflow: str, open_meteo_calls: int
) -> None:
    """Every Open-Meteo call stalling until it gives up still leaves half the job's time."""
    text = (PROJECT_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    (job,) = yaml.safe_load(text)["jobs"].values()
    worst = open_meteo_calls * OPEN_METEO_RETRY.worst_case_seconds()
    assert worst <= job["timeout-minutes"] * 60 / 2


# --- Previous Runs (archived forecasts; fair backtest only) ---
PREVIOUS_RUNS = "https://previous-runs.test/v1/forecast"


def test_fetch_previous_runs_requests_lagged_variables_and_tags_offsets() -> None:
    from src.data.weather_client import PREVIOUS_RUN_COLUMNS

    session = FakeSession(routes={PREVIOUS_RUNS: previous_runs_route})
    client = WeatherClient(session=session, cities=TWO_CITIES, previous_runs_url=PREVIOUS_RUNS)
    df = client.fetch_previous_runs(
        pd.Timestamp("2025-11-01T07:00Z"), pd.Timestamp("2025-11-02T22:00Z"), offsets=(1, 2)
    )

    assert list(df.columns) == PREVIOUS_RUN_COLUMNS
    assert len(df) == 48 * 2 * 2  # hours x cities x offsets
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    for (offset, city), rows in df.groupby(["offset_days", "city"]):
        lat = next(c.latitude for c in TWO_CITIES if c.name == city)
        assert (rows[list(WEATHER_VARS)] == 100.0 * offset + lat / 100).all().all()

    # One request per city: whole UTC dates, UTC clock, every variable at each offset, and
    # no model / relative-window parameters (same default model mix as the live forecast).
    assert len(session.calls) == 2
    for url, params in session.calls:
        assert url == PREVIOUS_RUNS
        assert params["start_date"] == "2025-11-01" and params["end_date"] == "2025-11-02"
        assert params["timezone"] == "UTC"
        assert params["hourly"].split(",") == [
            f"{v}_previous_day{n}" for n in (1, 2) for v in WEATHER_VARS
        ]
        assert not {"models", "past_days", "forecast_days"} & set(params)


def test_fetch_previous_runs_leaves_out_hours_without_values() -> None:
    def route(url, params):
        payload = previous_runs_route(url, params)
        for name in payload["hourly"]:
            if name.endswith("_previous_day2"):
                payload["hourly"][name][:3] = [None, None, None]
        del payload["hourly"]["cloud_cover_previous_day1"]  # a series left out entirely
        return payload

    session = FakeSession(routes={PREVIOUS_RUNS: route})
    client = WeatherClient(session=session, cities=TWO_CITIES[:1], previous_runs_url=PREVIOUS_RUNS)
    df = client.fetch_previous_runs("2025-11-01", "2025-11-01")
    assert (df["offset_days"] == 2).sum() == 21
    day1 = df[df["offset_days"] == 1]
    assert len(day1) == 24 and day1["cloud_cover"].isna().all()


def test_fetch_previous_runs_rejects_bad_ranges() -> None:
    client = WeatherClient(
        session=FakeSession(), cities=TWO_CITIES, previous_runs_url=PREVIOUS_RUNS
    )
    with pytest.raises(ValueError, match="before start"):
        client.fetch_previous_runs("2025-11-02", "2025-11-01")
    with pytest.raises(ValueError, match="offsets"):
        client.fetch_previous_runs("2025-11-01", "2025-11-01", offsets=(0, 1))
