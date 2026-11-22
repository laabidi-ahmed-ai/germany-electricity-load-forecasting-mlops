"""SMARD client tests - all offline via ``FakeSession``."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pytest

from src.data._http import FetchError, RetryPolicy, get_json
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
    assert get_json(session, url, policy=RetryPolicy(retries=3), sleep=sleeps.append) == {
        "ok": True
    }
    assert len(session.calls) == 2
    assert len(sleeps) == 1


def test_get_json_gives_up_after_retries() -> None:
    url = f"{BASE}/x.json"
    session = FakeSession(routes={url: [FakeResponse({}, status_code=500)] * 3})
    with pytest.raises(FetchError, match="giving up"):  # a RuntimeError, as before
        get_json(session, url, policy=RetryPolicy(retries=3), sleep=lambda _: None)
    assert len(session.calls) == 3


def test_get_json_does_not_retry_client_errors() -> None:
    url = f"{BASE}/missing.json"
    session = FakeSession(routes={url: [FakeResponse({}, status_code=404)] * 3})
    with pytest.raises(FetchError, match="HTTP 404"):
        get_json(session, url, policy=RetryPolicy(retries=3), sleep=lambda _: None)
    assert len(session.calls) == 1


def test_week_overlap_math() -> None:
    """A window that starts mid-week must still pull that week's file."""
    week = DST_WEEK_STARTS_MS[0]
    start = pd.Timestamp(week + 100 * MS_PER_HOUR, unit="ms", tz="UTC")
    client, session = make_client()
    client.fetch_load(start, start + pd.Timedelta(hours=1))
    assert any(str(week) in u for u, _ in session.calls)


def test_get_json_retries_rate_limits_with_growing_waits() -> None:
    url = f"{BASE}/x.json"
    session = FakeSession(
        routes={url: [FakeResponse({}, status_code=429), FakeResponse({}, status_code=503), {}]}
    )
    sleeps: list[float] = []
    assert get_json(session, url, policy=RetryPolicy(retries=3), sleep=sleeps.append) == {}
    assert sleeps == [3.0, 9.0]  # default backoff rides out a short provider hiccup


class HtmlResponse(FakeResponse):
    """A body that is not JSON (ingest run #95 only logged "Expecting value" for one)."""

    headers: ClassVar[dict[str, str]] = {"Content-Type": "text/html"}
    text = "<html>502 Bad Gateway</html>"

    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_get_json_names_what_came_back_when_the_body_is_not_json() -> None:
    url = f"{BASE}/x.json"
    session = FakeSession(routes={url: [HtmlResponse(None)] * 2})
    with pytest.raises(RuntimeError, match=r"non-JSON response \(HTTP 200, text/html, body") as e:
        get_json(session, url, policy=RetryPolicy(retries=2), sleep=lambda _: None)
    assert "502 Bad Gateway" in str(e.value)


# --- Week lengths across DST (fix plan Step 12) ---
# Mon 2024-10-21 00:00 CEST = 2024-10-20T22:00Z; the next Monday 00:00 CET = 2024-10-27T23:00Z.
AUTUMN_WEEK_MS = 1_729_461_600_000
AUTUMN_NEXT_WEEK_MS = AUTUMN_WEEK_MS + 169 * MS_PER_HOUR


def test_week_end_follows_the_local_calendar() -> None:
    from src.data.smard_client import week_end_ms

    assert pd.Timestamp(AUTUMN_WEEK_MS, unit="ms", tz="UTC") == pd.Timestamp("2024-10-20T22:00Z")
    spring, normal = DST_WEEK_STARTS_MS  # 167 h, then 168 h (tests/conftest.py)
    assert (week_end_ms(spring) - spring) // MS_PER_HOUR == 167
    assert (week_end_ms(normal) - normal) // MS_PER_HOUR == 168
    assert (week_end_ms(AUTUMN_WEEK_MS) - AUTUMN_WEEK_MS) // MS_PER_HOUR == 169


def test_autumn_week_keeps_its_169th_hour() -> None:
    """A fetch starting in the last hour of the 169-hour week used to skip that week file
    (a fixed 168 h said it ended an hour earlier) and lose the hour."""
    weeks = {AUTUMN_WEEK_MS: 169, AUTUMN_NEXT_WEEK_MS: 168}
    client, session = make_client(weeks=weeks)
    df = client.fetch_load("2024-10-27T22:00Z", "2024-10-28T01:00Z")

    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2024-10-27T22:00Z")
    assert len(df) == 4  # 22:00, 23:00, 00:00, 01:00 - no hour lost
    week_urls = [u for u, _ in session.calls if "410_DE_hour_" in u]
    assert len(week_urls) == 2  # both files: the hour lives in the first


def test_spring_week_file_is_not_downloaded_for_the_next_weeks_hours() -> None:
    """The 167-hour week ends an hour earlier than 168 h: a fetch starting at the next week's
    first hour needs only that week's file."""
    client, session = make_client()
    df = client.fetch_load("2021-03-28T22:00Z", "2021-03-29T05:00Z")
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2021-03-28T22:00Z")
    week_urls = [u for u, _ in session.calls if "410_DE_hour_" in u]
    assert week_urls == [f"{BASE}/410/DE/410_DE_hour_{DST_WEEK_STARTS_MS[1]}.json"]


def test_week_end_never_raises_on_nonexistent_or_ambiguous_local_times() -> None:
    """Not a real file start, but must not crash a fetch: +7 days lands on a DST switch."""
    from src.data.smard_client import week_end_ms

    def ms(ts: str) -> int:
        return int(pd.Timestamp(ts).timestamp() * 1000)

    # 2021-03-21 02:30 CET + 7 days = 2021-03-28 02:30, which does not exist -> 03:00 CEST
    start = ms("2021-03-21T01:30Z")
    assert week_end_ms(start) == ms("2021-03-28T01:00Z")
    # 2024-10-20 02:30 CEST + 7 days = 2024-10-27 02:30, which exists twice -> the later (CET)
    start = ms("2024-10-20T00:30Z")
    assert week_end_ms(start) == ms("2024-10-27T01:30Z")
    assert (week_end_ms(start) - start) // MS_PER_HOUR == 169  # never too early


def test_fetch_error_is_a_runtime_error() -> None:
    """Callers that catch RuntimeError keep working; the ingest can tell fetch failures apart."""
    assert issubclass(FetchError, RuntimeError)
