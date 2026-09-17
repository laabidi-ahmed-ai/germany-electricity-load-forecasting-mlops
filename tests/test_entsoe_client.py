"""ENTSO-E client tests - the entsoe-py client is replaced by a fake, no network."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data.entsoe_client import EntsoeClient


class FakePandasClient:
    """Mimics ``EntsoePandasClient`` for DE_LU: 15-min data in Europe/Berlin local time."""

    def __init__(self, *, raise_no_data: bool = False) -> None:
        self.calls: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
        self.raise_no_data = raise_no_data

    def _series(self, start: pd.Timestamp, end: pd.Timestamp, base: float) -> pd.Series:
        if self.raise_no_data:
            from entsoe.exceptions import NoMatchingDataError

            raise NoMatchingDataError
        idx = pd.date_range(start, end, freq="15min", inclusive="left").tz_convert("Europe/Berlin")
        values = base + np.arange(len(idx), dtype=float)  # strictly increasing quarter-hours
        return pd.Series(values, index=idx)

    def query_load(self, country_code, start, end):
        self.calls.append(("query_load", country_code, start, end))
        return self._series(start, end, 50_000).to_frame("Actual Load")

    def query_load_forecast(self, country_code, start, end):
        self.calls.append(("query_load_forecast", country_code, start, end))
        return self._series(start, end, 51_000).to_frame("Forecasted Load")


def test_without_token_is_unavailable_and_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    factory_calls: list[str] = []

    def factory(token: str):
        factory_calls.append(token)
        raise AssertionError("must not be called without a token")

    client = EntsoeClient(token=None, client_factory=factory)
    assert client.available is False

    with caplog.at_level(logging.INFO):
        actual = client.fetch_actual_load("2024-01-01", "2024-01-02")
        forecast = client.fetch_dayahead_forecast("2024-01-01", "2024-01-02")

    assert actual.empty and list(actual.columns) == ["timestamp_utc", "load_mw"]
    assert forecast.empty and list(forecast.columns) == ["timestamp_utc", "forecast_mw"]
    assert str(actual["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert factory_calls == []
    assert "no ENTSOE_API_TOKEN" in caplog.text


def test_token_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "tok-123")
    seen: list[str] = []
    client = EntsoeClient(client_factory=lambda t: seen.append(t) or FakePandasClient())
    assert client.available is True
    client.fetch_actual_load("2024-01-01", "2024-01-02")
    assert seen == ["tok-123"]


def test_placeholder_token_means_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "your_entsoe_security_token_here")
    assert EntsoeClient().available is False


def test_actual_load_is_resampled_to_hourly_utc() -> None:
    fake = FakePandasClient()
    client = EntsoeClient(token="t", bidding_zone="DE_LU", client_factory=lambda _: fake)
    df = client.fetch_actual_load("2024-01-01", "2024-01-02")

    assert list(df.columns) == ["timestamp_utc", "load_mw"]
    assert len(df) == 24
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2024-01-01T00:00Z")
    # Hourly mean of four increasing quarter-hours: 50000 + (0+1+2+3)/4 = 50001.5
    assert df["load_mw"].iloc[0] == pytest.approx(50_001.5)
    assert df["load_mw"].iloc[1] == pytest.approx(50_005.5)

    method, zone, start, end = fake.calls[0]
    assert (method, zone) == ("query_load", "DE_LU")
    assert start.tzinfo is not None and end.tzinfo is not None


def test_dayahead_forecast_column() -> None:
    client = EntsoeClient(token="t", client_factory=lambda _: FakePandasClient())
    df = client.fetch_dayahead_forecast("2024-01-01", "2024-01-01T06:00")
    assert list(df.columns) == ["timestamp_utc", "forecast_mw"]
    assert len(df) == 6


def test_dst_transition_yields_uniform_utc_hours() -> None:
    """Spring-forward day (2024-03-31): 23 local hours but 24 UTC hours, no gap."""
    client = EntsoeClient(token="t", client_factory=lambda _: FakePandasClient())
    df = client.fetch_actual_load("2024-03-30T23:00Z", "2024-03-31T23:00Z")
    assert len(df) == 24
    assert set(df["timestamp_utc"].diff().dropna()) == {pd.Timedelta(hours=1)}
    local = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    assert local.dt.date.value_counts()[pd.Timestamp("2024-03-31").date()] == 23


def test_no_matching_data_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    client = EntsoeClient(token="t", client_factory=lambda _: FakePandasClient(raise_no_data=True))
    with caplog.at_level(logging.WARNING):
        df = client.fetch_actual_load("2024-01-01", "2024-01-02")
    assert df.empty
    assert "no data" in caplog.text


def test_end_not_after_start_raises() -> None:
    client = EntsoeClient(token="t", client_factory=lambda _: FakePandasClient())
    with pytest.raises(ValueError, match="must be after"):
        client.fetch_actual_load("2024-01-02", "2024-01-01")
