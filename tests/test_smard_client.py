"""SMARD client tests - all offline via ``FakeSession``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data._http import get_json
from src.data.db import to_utc
from src.data.smard_client import SmardClient
from tests.conftest import (
    DST_WEEK_STARTS_MS,
    MS_PER_HOUR,
    FakeResponse,
    FakeSession,
    smard_routes,
)

BASE = "https://smard.test/app/chart_data"


def make_client(**kwargs) -> tuple[SmardClient, FakeSession]:
    session = FakeSession(routes=smard_routes(BASE, "DE", **kwargs))
    return SmardClient(session=session, base_url=BASE, region="DE"), session


def test_urls_follow_smard_scheme() -> None:
    client, _ = make_client()
    assert client.index_url() == f"{BASE}/410/DE/index_hour.json"
    assert client.week_url(1_616_367_600_000) == f"{BASE}/410/DE/410_DE_hour_1616367600000.json"


def test_fetch_load_returns_sorted_unique_utc_hours() -> None:
    client, _ = make_client()
    df = client.fetch_load("2021-03-21", "2021-04-05")

    assert list(df.columns) == ["timestamp_utc", "load_mw"]
    assert str(df["timestamp_utc"].dt.tz) == "UTC"
    assert df["timestamp_utc"].is_monotonic_increasing
    assert not df["timestamp_utc"].duplicated().any()
    assert len(df) == 167 + 168
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2021-03-21T23:00Z")
    assert df["timestamp_utc"].iloc[-1] == pd.Timestamp("2021-04-04T21:00Z")


def test_dst_week_has_no_gap_in_utc() -> None:
    """The 23-hour local Sunday is a plain 1h-step sequence in UTC."""
    client, _ = make_client()
    df = client.fetch_load("2021-03-21", "2021-04-05")
    steps = df["timestamp_utc"].diff().dropna().unique()
    assert list(steps) == [pd.Timedelta(hours=1)]
    # And re-expressed in Berlin local time, the DST Sunday really has 23 hours.
    local = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    per_day = local.dt.date.value_counts()
    assert per_day[pd.Timestamp("2021-03-28").date()] == 23


def test_window_filtering_only_downloads_overlapping_weeks() -> None:
    client, session = make_client()
    df = client.fetch_load("2021-03-30", "2021-03-31T12:00")

    week_calls = [u for u, _ in session.calls if "410_DE_hour_" in u]
    assert week_calls == [f"{BASE}/410/DE/410_DE_hour_{DST_WEEK_STARTS_MS[1]}.json"]
    assert df["timestamp_utc"].min() == pd.Timestamp("2021-03-30T00:00Z")
    assert df["timestamp_utc"].max() == pd.Timestamp("2021-03-31T12:00Z")


def test_null_values_are_dropped() -> None:
    """SMARD fills unpublished future hours with null; they must not be stored."""
    client, _ = make_client(null_from=100)
    df = client.fetch_load("2021-03-21", "2021-04-05")
    assert len(df) == 100 + 100
    assert (df["load_mw"] > 0).all()


def test_empty_when_window_has_no_data() -> None:
    client, _ = make_client()
    df = client.fetch_load("2025-01-01", "2025-01-02")
    assert df.empty
    assert list(df.columns) == ["timestamp_utc", "load_mw"]
    assert str(df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"


def test_end_before_start_raises() -> None:
    client, _ = make_client()
    with pytest.raises(ValueError, match="before start"):
        client.fetch_load("2021-04-01", "2021-03-01")


def test_to_utc_treats_naive_as_utc_and_converts_aware() -> None:
    assert to_utc("2021-03-01") == pd.Timestamp("2021-03-01T00:00Z")
    berlin = pd.Timestamp("2021-03-01T00:00", tz="Europe/Berlin")
    assert to_utc(berlin) == pd.Timestamp("2021-02-28T23:00Z")


def test_get_json_retries_then_succeeds() -> None:
    url = f"{BASE}/x.json"
    session = FakeSession(routes={url: [FakeResponse({}, status_code=503), {"ok": True}]})
    sleeps: list[float] = []
    assert get_json(session, url, retries=3, sleep=sleeps.append) == {"ok": True}
    assert len(session.calls) == 2
    assert len(sleeps) == 1


def test_get_json_gives_up_after_retries() -> None:
    url = f"{BASE}/x.json"
    session = FakeSession(routes={url: [FakeResponse({}, status_code=500)] * 3})
    with pytest.raises(RuntimeError, match="giving up"):
        get_json(session, url, retries=3, sleep=lambda _: None)
    assert len(session.calls) == 3


def test_get_json_does_not_retry_client_errors() -> None:
    url = f"{BASE}/missing.json"
    session = FakeSession(routes={url: [FakeResponse({}, status_code=404)] * 3})
    with pytest.raises(RuntimeError, match="HTTP 404"):
        get_json(session, url, retries=3, sleep=lambda _: None)
    assert len(session.calls) == 1


def test_week_overlap_math() -> None:
    """A window that starts mid-week must still pull that week's file."""
    week = DST_WEEK_STARTS_MS[0]
    start = pd.Timestamp(week + 100 * MS_PER_HOUR, unit="ms", tz="UTC")
    client, session = make_client()
    client.fetch_load(start, start + pd.Timedelta(hours=1))
    assert any(str(week) in u for u, _ in session.calls)
