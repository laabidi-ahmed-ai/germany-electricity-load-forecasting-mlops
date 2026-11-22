"""End-to-end ingestion tests (backfill + incremental) with fake clients and SQLite, plus the
archive-failure rule and the gap scan."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, update
from sqlalchemy.exc import OperationalError

from src.data import db, ingest
from src.data.smard_client import SmardClient, week_end_ms
from src.data.validation import DataValidationError, find_gaps
from src.data.weather_client import WEATHER_VARS, City, WeatherClient
from src.observability import JobMetrics
from tests.conftest import (
    DST_WEEK_STARTS_MS,
    FakeResponse,
    FakeSession,
    open_meteo_route,
    smard_routes,
)

SMARD_BASE = "https://smard.test/app/chart_data"
ARCHIVE = "https://archive.test/v1/archive"
FORECAST_BASE = "https://forecast.test/v1"
FALLBACK = "https://previous-runs.test/v1/forecast"
CITIES = (City("A", 50.0, 8.0), City("B", 52.0, 13.0))


def make_clients(
    *,
    weeks: dict[int, int] | None = None,
    weather_end: str = "2021-04-04T21:00",
    smard_routes_override: dict | None = None,
) -> tuple[ingest.Clients, FakeSession]:
    routes = smard_routes_override or smard_routes(SMARD_BASE, "DE", weeks)
    routes[ARCHIVE] = open_meteo_route("2021-03-21T23:00", weather_end)
    routes[f"{FORECAST_BASE}/forecast"] = open_meteo_route("2021-04-03T00:00", "2021-04-07T23:00")
    session = FakeSession(routes=routes)
    clients = ingest.Clients(
        smard=SmardClient(session=session, base_url=SMARD_BASE, region="DE"),
        weather=WeatherClient(
            session=session,
            cities=CITIES,
            archive_url=ARCHIVE,
            forecast_base_url=FORECAST_BASE,
            previous_runs_url=FALLBACK,
        ),
    )
    return clients, session


def by_source(results: list[ingest.IngestResult]) -> dict[str, ingest.IngestResult]:
    return {r.source: r for r in results}


# --- backfill ---
def test_backfill_loads_smard_and_weather(engine) -> None:
    clients, _ = make_clients()
    results = by_source(ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05"))

    assert results["smard"].status == "ok" and results["smard"].rows == 335
    assert results["smard_forecast"].status == "ok" and results["smard_forecast"].rows == 335
    assert results["weather_archive"].status == "ok"
    assert not any(r.failed for r in results.values())

    load = db.read_load(engine)
    assert len(load) == 335
    assert 50_000 < load["load_mw"].mean() < 60_000
    assert load["timestamp_utc"].min() == pd.Timestamp("2021-03-21T23:00Z")
    # The official day-ahead forecast (SMARD 411) lands in load_forecast_official.
    official = db.read_table(engine, db.LoadForecastOfficial, source="smard")
    assert len(official) == 335
    merged = load.merge(official, on="timestamp_utc")
    np.testing.assert_allclose(merged["forecast_mw"], merged["load_mw"] * 0.98)
    assert db.count_rows(engine, db.WeatherHourly) == 335 * len(CITIES)


def test_backfill_default_start_is_data_start_date(engine, monkeypatch) -> None:
    """With no --start the backfill begins at DATA_START_DATE (2021-03-01)."""
    clients, session = make_clients()
    ingest.run_backfill(engine, clients, sources=("weather",))
    _, params = next(c for c in session.calls if c[0] == ARCHIVE)
    assert params["start_date"] == "2021-03-01"


def test_backfill_is_idempotent(engine) -> None:
    clients, _ = make_clients()
    ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05")
    ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05")
    assert db.count_rows(engine, db.LoadActual) == 335
    assert db.count_rows(engine, db.WeatherHourly) == 335 * len(CITIES)


def test_validation_failure_writes_nothing_and_does_not_block_other_sources(engine) -> None:
    routes = smard_routes(SMARD_BASE, "DE")
    # Corrupt the SMARD data: values 100x too small -> mean load ~550 MW.
    for url, payload in routes.items():
        if "410_DE_hour_" in url:
            payload["series"] = [[t, v / 100 if v else v] for t, v in payload["series"]]
    clients, _ = make_clients(smard_routes_override=routes)

    results = by_source(ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05"))
    assert results["smard"].failed
    assert "validation" in results["smard"].message
    assert results["weather_archive"].status == "ok"
    assert db.count_rows(engine, db.LoadActual) == 0
    assert db.count_rows(engine, db.WeatherHourly) > 0


def test_network_failure_is_contained(engine, monkeypatch) -> None:
    clients, session = make_clients()
    session.routes.pop(f"{SMARD_BASE}/410/DE/index_hour.json")  # -> 404 on every try

    import src.data.smard_client as smard_mod

    original = smard_mod.get_json
    monkeypatch.setattr(
        smard_mod,
        "get_json",
        lambda s, url, params=None, **kw: original(s, url, params, sleep=lambda _: None, **kw),
    )
    results = by_source(ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05"))

    assert results["smard"].failed
    assert results["weather_archive"].status == "ok"


# --- incremental ---
def test_incremental_fetches_only_new_hours_without_duplicates(engine) -> None:
    # Backfill first week only.
    first_week = {DST_WEEK_STARTS_MS[0]: 167}
    clients, _ = make_clients(weeks=first_week, weather_end="2021-03-28T21:00")
    ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05")
    assert db.count_rows(engine, db.LoadActual) == 167
    latest_before = db.latest_timestamp(engine, db.LoadActual, source="smard")

    # Now SMARD has the second week too.
    clients2, session2 = make_clients()
    now = pd.Timestamp("2021-04-05T00:00Z")
    results = by_source(ingest.run_incremental(engine, clients2, now=now))

    assert results["smard"].status == "ok"
    assert results["smard_forecast"].status == "ok"
    assert results["weather_archive"].status == "ok"
    assert results["weather_forecast"].status == "ok"
    assert db.count_rows(engine, db.LoadForecastOfficial, source="smard") == 335
    # the 48 h overlap re-read unchanged values: one revision per hour, no duplicates
    assert db.count_rows(engine, db.LoadForecastOfficialRevision, source="smard") == 335

    # No duplicates, everything contiguous in UTC.
    load = db.read_load(engine)
    assert len(load) == 335
    assert not load["timestamp_utc"].duplicated().any()
    assert set(load["timestamp_utc"].diff().dropna()) == {pd.Timedelta(hours=1)}

    # The fetch started at latest - 48h (overlap), which reaches back into week 1,
    # so both week files are downloaded - and the overlap produced no duplicates.
    assert results["smard"].start == latest_before - pd.Timedelta(hours=48)
    week_urls = [u for u, _ in session2.calls if "410_DE_hour_" in u]
    assert week_urls == [f"{SMARD_BASE}/410/DE/410_DE_hour_{w}.json" for w in DST_WEEK_STARTS_MS]


def test_incremental_on_empty_db_falls_back_to_full_backfill(engine) -> None:
    clients, session = make_clients()
    results = by_source(
        ingest.run_incremental(engine, clients, now=pd.Timestamp("2021-04-05T00:00Z"))
    )
    assert results["smard"].rows == 335
    _, params = next(c for c in session.calls if c[0] == ARCHIVE)
    assert params["start_date"] == "2021-03-01"


def test_incremental_weather_forecast_does_not_overwrite_archive(engine) -> None:
    clients, _ = make_clients()
    now = pd.Timestamp("2021-04-05T00:00Z")
    ingest.run_incremental(engine, clients, sources=("weather",), now=now)

    w = db.read_table(engine, db.WeatherHourly)
    # Archive covers up to 2021-04-04T21:00; forecast covers 04-03 .. 04-07.
    overlap = w[w["timestamp_utc"] <= pd.Timestamp("2021-04-04T21:00Z")]
    assert (overlap["source"] == "archive").all()
    future = w[w["timestamp_utc"] > pd.Timestamp("2021-04-04T21:00Z")]
    assert (future["source"] == "forecast").all() and len(future) > 0


def test_weather_forecast_reports_whether_the_fallback_host_served(engine) -> None:
    clients, session = make_clients()
    now = pd.Timestamp("2021-04-05T00:00Z")
    first = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=now))
    assert first["weather_forecast"].gauges == {"weather.forecast_fallback": 0}

    # api.open-meteo.com refuses (HTTP 404 is final), the previous-runs host answers.
    session.routes[FALLBACK] = session.routes.pop(f"{FORECAST_BASE}/forecast")
    second = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=now))
    assert second["weather_forecast"].status == "ok"
    assert second["weather_forecast"].rows == first["weather_forecast"].rows
    assert second["weather_forecast"].gauges == {"weather.forecast_fallback": 1}


# --- archive failures: a warning while the stored archive is fresh (Step 12) ---
NOW = pd.Timestamp("2021-04-05T00:00Z")  # the fixture archive ends 2021-04-04T21:00: 3 h old


def seeded_weather(engine) -> tuple[ingest.Clients, FakeSession]:
    clients, session = make_clients()
    ingest.run_incremental(engine, clients, sources=("weather",), now=NOW)
    return clients, session


def test_failed_archive_fetch_is_a_warning_while_the_archive_is_fresh(engine, caplog) -> None:
    clients, session = seeded_weather(engine)
    session.routes.pop(ARCHIVE)  # HTTP 404 from archive-api (final, so no retry waits)
    with caplog.at_level(logging.WARNING, logger="src.data.ingest"):
        results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=NOW))

    archive = results["weather_archive"]
    assert archive.status == "warning" and not archive.failed
    assert archive.message.startswith("fetch failed, archive 3.0 h old:")
    assert "warning only" in caplog.text
    assert results["weather_forecast"].status == "ok"
    assert not any(r.failed for r in results.values())


def test_failed_archive_fetch_fails_once_the_archive_is_behind(engine) -> None:
    clients, session = seeded_weather(engine)
    session.routes.pop(ARCHIVE)
    later = NOW + pd.Timedelta(hours=22)  # 25 h after the newest archive hour
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=later))
    assert results["weather_archive"].failed
    assert "archive 25.0 h old, above the 24 h limit" in results["weather_archive"].message

    at_limit = NOW + pd.Timedelta(hours=21)  # exactly 24 h: still a warning
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=at_limit))
    assert results["weather_archive"].status == "warning"


def test_failed_archive_fetch_with_no_archive_stored_fails(engine) -> None:
    clients, session = make_clients()
    session.routes.pop(ARCHIVE)
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=NOW))
    assert results["weather_archive"].failed
    assert "no archive stored" in results["weather_archive"].message


def test_archive_http_500_after_retries_is_a_warning(engine, monkeypatch) -> None:
    import src.data.weather_client as weather_mod

    clients, session = seeded_weather(engine)
    original = weather_mod.get_json
    monkeypatch.setattr(
        weather_mod,
        "get_json",
        lambda s, u, p=None, **kw: original(s, u, p, sleep=lambda _: None, **kw),
    )
    session.routes[ARCHIVE] = [FakeResponse({"error": True}, status_code=500)] * 6
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=NOW))
    archive = results["weather_archive"]
    assert archive.status == "warning" and archive.error_kind == ingest.ERROR_FETCH
    assert "giving up" in archive.message


def unexpected_answers():
    def no_hourly(url, params):
        return [{"latitude": 50.0}, {"latitude": 52.0, "location_id": 1}]

    def wrong_order(url, params):
        return [{"hourly": {}}, {"hourly": {}, "location_id": 7}]

    return [no_hourly, wrong_order]


@pytest.mark.parametrize("answer", unexpected_answers(), ids=["no-hourly", "location-id"])
def test_an_unexpected_archive_answer_stays_a_failure(engine, answer) -> None:
    """A changed payload is not provider noise: it must not hide as a warning."""
    clients, session = seeded_weather(engine)
    session.routes[ARCHIVE] = answer
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=NOW))
    assert results["weather_archive"].failed
    assert results["weather_archive"].error_kind == ingest.ERROR_OTHER


def test_an_archive_write_error_stays_a_failure(engine, monkeypatch) -> None:
    clients, _ = seeded_weather(engine)
    original = ingest.db.upsert_dataframe

    def failing_upsert(eng, model, df, **kwargs):
        if model is db.WeatherHourly and (df["source"] == "archive").all():
            raise OperationalError("INSERT ...", {}, Exception("disk full"))
        return original(eng, model, df, **kwargs)

    monkeypatch.setattr(ingest.db, "upsert_dataframe", failing_upsert)
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=NOW))
    assert results["weather_archive"].failed
    assert results["weather_archive"].error_kind == ingest.ERROR_OTHER


def test_invalid_archive_data_stays_a_failure_even_when_fresh(engine) -> None:
    clients, session = seeded_weather(engine)
    good = session.routes[ARCHIVE]

    def hot(url, params):
        payloads = good(url, params)
        for payload in payloads:
            payload["hourly"]["temperature_2m"] = [99.0] * len(payload["hourly"]["time"])
        return payloads

    session.routes[ARCHIVE] = hot
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=NOW))
    assert results["weather_archive"].failed
    assert results["weather_archive"].message.startswith("validation: ")


def test_a_forecast_that_does_not_cover_tomorrow_is_refused(engine) -> None:
    clients, _ = make_clients()
    # the fake forecast ends 2021-04-07T23:00; on 04-07 at noon "tomorrow" ends 04-08T21:00Z
    late = pd.Timestamp("2021-04-07T12:00Z")
    results = by_source(ingest.run_incremental(engine, clients, sources=("weather",), now=late))
    assert results["weather_forecast"].failed
    assert "stale or cut off" in results["weather_forecast"].message


def test_end_of_next_local_day_is_dst_aware() -> None:
    summer = ingest.end_of_next_local_day(pd.Timestamp("2021-04-05T00:00Z"))
    assert summer == pd.Timestamp("2021-04-06T21:00Z")  # CEST
    winter = ingest.end_of_next_local_day(pd.Timestamp("2021-01-05T23:30Z"))
    assert winter == pd.Timestamp("2021-01-07T22:00Z")  # already 6 Jan in Berlin (CET)


def test_metrics_report_archive_age_and_no_failure_for_a_warning(engine) -> None:
    seeded_weather(engine)
    results = [
        ingest.IngestResult("weather_archive", status="warning", message="fetch failed"),
        ingest.IngestResult("smard_gaps", gauges={"ingest.gap_hours": 3}),
    ]
    metrics = JobMetrics("ingest")
    ingest.report_metrics(metrics, engine, results)
    series = {(s["metric"], tuple(s["tags"])): s["points"][0]["value"] for s in metrics._series}

    def value(name: str, *tags: str) -> float:
        return series[(f"germany_load.{name}", ("job:ingest", "runner:local", *tags))]

    assert value("ingest.step_failed", "step:weather_archive") == 0
    assert value("ingest.gap_hours", "step:smard_gaps") == 3
    assert value("weather.archive_age_hours") > 24 * 365  # the 2021 archive is old by now
    assert metrics.status == "ok"


# --- gap scan (Step 12) ---
# After the backfill, the latest hour is 2021-04-04T21:00 for load and archive alike: the
# scan looks before 04-02T21:00 (48 h) for load and before 03-28T21:00 (7 d) for weather.
OLD_HOLE = pd.date_range("2021-03-24T05:00Z", periods=3, freq="h")
RECENT_HOLE = pd.date_range("2021-04-04T10:00Z", periods=2, freq="h")  # inside the overlap


def backfilled(engine, **kwargs) -> tuple[ingest.Clients, FakeSession]:
    clients, session = make_clients(**kwargs)
    ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05")
    return clients, session


def drop_hours(engine, model, hours, **filters) -> None:
    with engine.begin() as conn:
        stmt = delete(model).where(model.timestamp_utc.in_([t.to_pydatetime() for t in hours]))
        for col, val in filters.items():
            stmt = stmt.where(getattr(model, col) == val)
        conn.execute(stmt)


def test_gap_scan_refills_holes_older_than_the_overlap(engine) -> None:
    clients, _ = backfilled(engine)
    for model in (db.LoadActual, db.LoadForecastOfficial):
        drop_hours(engine, model, OLD_HOLE)
        drop_hours(engine, model, RECENT_HOLE)
    drop_hours(engine, db.WeatherHourly, OLD_HOLE, city="A")
    # an hour that only has a forecast row is a hole in the archive
    with engine.begin() as conn:
        conn.execute(
            update(db.WeatherHourly)
            .where(db.WeatherHourly.timestamp_utc == pd.Timestamp("2021-03-26T12:00Z"))
            .values(source="forecast")
        )

    results = by_source(ingest.run_gap_scan(engine, clients))

    for step in ("smard_gaps", "smard_forecast_gaps"):
        assert results[step].status == "ok"
        assert results[step].gauges == {"ingest.gap_hours": 3}  # the recent hole is not counted
        assert results[step].message.startswith("1 holes, 3 h older than the overlap")
        assert "re-fetched 1 of 1 window(s)" in results[step].message
    assert results["weather_archive_gaps"].gauges == {"ingest.gap_hours": 4}  # 3 + 1
    assert "re-fetched 1 of 1 window(s)" in results["weather_archive_gaps"].message  # merged

    load = db.read_load(engine)
    assert set(OLD_HOLE) <= set(load["timestamp_utc"])
    assert not set(RECENT_HOLE) & set(load["timestamp_utc"])  # left to the normal run
    official = db.read_table(engine, db.LoadForecastOfficial)
    assert set(OLD_HOLE) <= set(official["timestamp_utc"])
    archive = db.covered_hours(engine, db.WeatherHourly, cities=["A", "B"], source="archive")
    assert len(archive) == 335


def test_gap_scan_on_a_complete_history_fetches_nothing(engine) -> None:
    clients, session = backfilled(engine)
    calls = len(session.calls)
    results = ingest.run_gap_scan(engine, clients)
    assert [r.gauges for r in results] == [{"ingest.gap_hours": 0}] * 3
    assert len(session.calls) == calls


def test_gap_scan_on_an_empty_database_is_skipped(engine) -> None:
    clients, _ = make_clients()
    assert {r.status for r in ingest.run_gap_scan(engine, clients)} == {"skipped"}


def test_holes_the_provider_cannot_fill_are_logged_every_scan(engine, caplog) -> None:
    routes = smard_routes(SMARD_BASE, "DE")
    week = routes[f"{SMARD_BASE}/410/DE/410_DE_hour_{DST_WEEK_STARTS_MS[0]}.json"]["series"]
    hole_ms = {int(t.timestamp() * 1000) for t in OLD_HOLE}
    for point in week:
        if point[0] in hole_ms:
            point[1] = None  # SMARD has no value for these hours
    clients, _ = backfilled(engine, smard_routes_override=routes)

    for _ in range(2):  # the next day's scan finds and logs them again
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="src.data.ingest"):
            results = by_source(ingest.run_gap_scan(engine, clients, sources=("smard",)))
        assert results["smard_gaps"].gauges == {"ingest.gap_hours": 3}
        assert results["smard_gaps"].status == "ok"
        assert "smard_gaps: 3 hour(s) between" in caplog.text
        assert "still missing after re-fetch" in caplog.text


def test_gap_scan_stops_starting_refetches_after_its_budget(engine, caplog) -> None:
    clients, _ = backfilled(engine)
    for model in (db.LoadActual, db.LoadForecastOfficial):
        drop_hours(engine, model, OLD_HOLE)
    ticks = iter([0.0, 0.0, 180.0])  # start, before the 1st re-fetch, before the 2nd
    with caplog.at_level(logging.WARNING, logger="src.data.ingest"):
        results = by_source(
            ingest.run_gap_scan(engine, clients, sources=("smard",), clock=lambda: next(ticks))
        )

    assert "re-fetched 1 of 1 window(s)" in results["smard_gaps"].message
    assert "re-fetched 0 of 1 window(s)" in results["smard_forecast_gaps"].message
    assert "budget is used - 1 re-fetch window(s) left for the next scan" in caplog.text
    official = db.read_table(engine, db.LoadForecastOfficial)
    assert not set(OLD_HOLE) & set(official["timestamp_utc"])  # waits for the next scan


def test_gap_scan_caps_windows_per_table_newest_first(engine, caplog) -> None:
    """Newest first: permanent old holes under the cap cannot starve a newer, fillable one."""
    clients, _ = backfilled(engine)
    later_hole = pd.date_range("2021-03-31T05:00Z", periods=2, freq="h")  # 7 days later
    drop_hours(engine, db.LoadActual, OLD_HOLE.append(later_hole))
    with caplog.at_level(logging.WARNING, logger="src.data.ingest"):
        results = by_source(ingest.run_gap_scan(engine, clients, sources=("smard",), max_windows=1))
    assert results["smard_gaps"].gauges == {"ingest.gap_hours": 5}
    assert "re-fetched 1 of 1 window(s)" in results["smard_gaps"].message
    assert "1 window(s) beyond the cap of 1 left for the next scan" in caplog.text
    load = set(db.read_load(engine)["timestamp_utc"])
    assert set(later_hole) <= load and not set(OLD_HOLE) & load


def test_a_failed_refetch_is_a_warning_and_never_fails_the_job(engine) -> None:
    clients, session = backfilled(engine)
    drop_hours(engine, db.LoadActual, OLD_HOLE)
    session.routes.pop(f"{SMARD_BASE}/410/DE/index_hour.json")  # HTTP 404 (final)
    results = by_source(ingest.run_gap_scan(engine, clients, sources=("smard",)))
    assert results["smard_gaps"].status == "warning" and not results["smard_gaps"].failed


def test_gap_scan_requests_use_the_short_retry_policy(engine, monkeypatch) -> None:
    import src.data.smard_client as smard_mod
    import src.data.weather_client as weather_mod

    clients, _ = backfilled(engine)
    drop_hours(engine, db.LoadActual, OLD_HOLE)
    drop_hours(engine, db.WeatherHourly, OLD_HOLE)
    seen = []
    for mod in (smard_mod, weather_mod):

        def recording(session, url, params=None, *, policy, _original=mod.get_json):
            seen.append(policy)
            return _original(session, url, params, policy=policy)

        monkeypatch.setattr(mod, "get_json", recording)
    ingest.run_gap_scan(engine, clients)
    assert seen and set(seen) == {ingest.GAP_SCAN_RETRY}


def test_gap_scan_weather_refetch_only_warns_on_missing_values(engine, caplog) -> None:
    """An unfillable stretch in the same answer must not reject the fillable hours."""
    clients, session = backfilled(engine)
    drop_hours(engine, db.WeatherHourly, OLD_HOLE)
    good = session.routes[ARCHIVE]
    blank = pd.date_range("2021-03-25T00:00Z", periods=40, freq="h")  # 12 % of the answer
    blank_text = {t.strftime("%Y-%m-%dT%H:%M") for t in blank}

    def patchy(url, params):
        payloads = good(url, params)
        for payload in payloads:
            hourly = payload["hourly"]
            for i, t in enumerate(hourly["time"]):
                if t in blank_text:
                    for var in WEATHER_VARS:
                        hourly[var][i] = None
        return payloads

    session.routes[ARCHIVE] = patchy
    with pytest.raises(DataValidationError, match="missing \\(limit 10%\\)"):
        ingest.ingest_weather_history(engine, clients.weather, OLD_HOLE[0], OLD_HOLE[-1])

    with caplog.at_level(logging.WARNING):
        results = by_source(ingest.run_gap_scan(engine, clients, sources=("weather",)))
    assert results["weather_archive_gaps"].status == "ok"
    archive = db.covered_hours(engine, db.WeatherHourly, cities=["A", "B"], source="archive")
    assert set(OLD_HOLE) <= set(archive)
    assert "11.9% missing" in caplog.text  # still reported, as a warning (40 of 335 h)


def test_split_hole_hours_are_each_counted_once(engine) -> None:
    """A hole split into windows: the split hour is counted by the window it ends."""
    clients, _ = backfilled(engine)
    hole = pd.date_range("2021-03-22T00:00Z", "2021-04-01T23:00Z", freq="h")  # 264 h
    drop_hours(engine, db.LoadActual, hole)
    (source, *_) = ingest.gap_sources(engine, clients, ("smard",))
    (gap,) = find_gaps(pd.Series(db.covered_hours(engine, db.LoadActual, source="smard")))
    windows = ingest.refetch_windows([gap])
    assert len(windows) == 2
    counts = [ingest._missing_hours(engine, source, start, end) for start, end in windows]
    assert sum(counts) == gap.missing_hours == 264


def test_gap_scan_query_errors_are_warnings(engine, monkeypatch) -> None:
    clients, _ = backfilled(engine)

    def broken(*args, **kwargs):
        raise OperationalError("SELECT ...", {}, Exception("statement timeout"))

    monkeypatch.setattr(ingest.db, "covered_hours", broken)
    results = ingest.run_gap_scan(engine, clients)
    assert [r.status for r in results] == ["warning"] * 3
    assert all("statement timeout" in r.message for r in results)


def test_post_refetch_recheck_errors_are_warnings(engine, monkeypatch) -> None:
    clients, _ = backfilled(engine)
    drop_hours(engine, db.LoadActual, OLD_HOLE)

    def broken(*args, **kwargs):
        raise OperationalError("SELECT ...", {}, Exception("connection dropped"))

    monkeypatch.setattr(ingest, "_missing_hours", broken)
    results = by_source(ingest.run_gap_scan(engine, clients, sources=("smard",)))
    assert results["smard_gaps"].status == "warning"
    assert set(OLD_HOLE) <= set(db.read_load(engine)["timestamp_utc"])  # still written


def test_a_crashing_gap_scan_is_a_warning_and_the_run_reports(tmp_path, monkeypatch) -> None:
    """The summary and report_metrics run whatever the scan does; the exit code is 0."""
    clients, _ = make_clients()
    url = f"sqlite:///{(tmp_path / 'scan.db').as_posix()}"
    ingest.run_backfill(db.get_engine(url), clients, start="2021-03-21", end="2021-04-05")

    def crash(*args, **kwargs):
        raise RuntimeError("scan exploded")

    reported = []
    original_report = ingest.report_metrics
    monkeypatch.setattr(ingest, "run_gap_scan", crash)
    monkeypatch.setattr(ingest, "Clients", lambda: clients)
    monkeypatch.setattr(
        ingest,
        "report_metrics",
        lambda m, e, results: (reported.append(results), original_report(m, e, results)),
    )
    rc = ingest.main(["--sources", "smard", "--gap-scan", "now", "--database-url", url])
    assert rc == 0
    (results,) = reported
    scan = by_source(results)["gap_scan"]
    assert scan.status == "warning" and scan.message == "scan exploded"


@pytest.mark.parametrize(
    ("mode", "hour", "scans"),
    [("off", 2, False), ("daily", 2, True), ("daily", 3, False), ("now", 13, True)],
)
def test_gap_scan_modes(engine, mode: str, hour: int, scans: bool) -> None:
    clients, _ = backfilled(engine)
    now = pd.Timestamp(f"2021-04-05T{hour:02d}:30Z")
    steps = {r.source for r in ingest.run_incremental(engine, clients, now=now, gap_scan=mode)}
    assert ("smard_gaps" in steps) is scans
    assert ("weather_archive_gaps" in steps) is scans


def test_unknown_gap_scan_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="gap scan mode"):
        ingest.should_scan("hourly", NOW)


def test_refetch_windows_merge_close_holes_and_split_long_ones() -> None:
    t = pd.Timestamp("2021-03-01T00:00Z")
    h = pd.Timedelta(hours=1)
    gap = ingest.Gap
    close = [gap(t, t + 3 * h, 2), gap(t + 50 * h, t + 60 * h, 9)]
    assert ingest.refetch_windows(close) == [(t, t + 60 * h)]
    apart = [gap(t, t + 3 * h, 2), gap(t + 200 * h, t + 202 * h, 1)]
    assert ingest.refetch_windows(apart) == [(t, t + 3 * h), (t + 200 * h, t + 202 * h)]
    long = [gap(t, t + 15 * 24 * h, 359)]  # 15 days -> 6 + 6 + 3
    windows = ingest.refetch_windows(long)
    assert windows == [(t, t + 144 * h), (t + 144 * h, t + 288 * h), (t + 288 * h, t + 360 * h)]
    assert all(end - start <= ingest.GAP_SCAN_WINDOW for start, end in windows)


def test_a_six_day_window_needs_at_most_two_smard_week_files() -> None:
    """Weeks are >= 167 h, so a 144 h window crosses at most one file boundary: with the
    index, ``SMARD_REQUESTS_PER_WINDOW`` requests - what the timing test counts."""
    starts = [DST_WEEK_STARTS_MS[0]]
    for _ in range(60):  # a year of real week starts, both DST switches included
        starts.append(week_end_ms(starts[-1]))
    span = int(ingest.GAP_SCAN_WINDOW.total_seconds() * 1000)
    for lo in range(starts[0], starts[-3], 3_600_000):
        files = [w for w in starts if week_end_ms(w) > lo and w <= lo + span]
        assert 1 + len(files) <= ingest.SMARD_REQUESTS_PER_WINDOW


# --- official forecast revisions ---
def test_official_revisions_are_recorded_only_for_new_or_changed_values(engine) -> None:
    span = {"start": "2021-03-21", "end": "2021-04-05", "sources": ("smard",)}
    first_week = {DST_WEEK_STARTS_MS[0]: 167}
    clients, _ = make_clients(weeks=first_week)
    results = by_source(ingest.run_backfill(engine, clients, **span))
    assert results["smard_forecast"].message == "167 new/changed hours"
    assert db.count_rows(engine, db.LoadForecastOfficialRevision) == 167

    results = by_source(ingest.run_backfill(engine, clients, **span))  # the same values again
    assert results["smard_forecast"].message == "0 new/changed hours"
    assert db.count_rows(engine, db.LoadForecastOfficialRevision) == 167

    # SMARD revises one hour of week 1 and publishes week 2.
    routes = smard_routes(SMARD_BASE, "DE")
    series = routes[f"{SMARD_BASE}/411/DE/411_DE_hour_{DST_WEEK_STARTS_MS[0]}.json"]["series"]
    ts_ms, old = series[10]
    series[10] = [ts_ms, old + 500.0]
    clients2, _ = make_clients(smard_routes_override=routes)
    results = by_source(ingest.run_backfill(engine, clients2, **span))
    assert results["smard_forecast"].message == "169 new/changed hours"  # 168 new + 1 changed

    revisions = db.read_table(engine, db.LoadForecastOfficialRevision)
    assert len(revisions) == 167 + 168 + 1
    hour = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
    history = revisions[revisions["timestamp_utc"] == hour].sort_values("ingested_at")
    assert list(history["forecast_mw"]) == [old, old + 500.0]  # the earlier value is kept
    latest = db.read_table(engine, db.LoadForecastOfficial, start=hour, end=hour)
    assert latest["forecast_mw"].tolist() == [old + 500.0]  # the scores use the latest value


# --- CLI ---
def test_cli_backfill_smard_only(tmp_path, monkeypatch) -> None:
    clients, _ = make_clients()
    monkeypatch.setattr(ingest, "Clients", lambda: clients)
    url = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    rc = ingest.main(
        [
            "--backfill",
            "--start",
            "2021-03-21",
            "--end",
            "2021-04-05",
            "--sources",
            "smard",
            "--database-url",
            url,
        ]
    )
    assert rc == 0
    eng = db.get_engine(url)
    assert db.count_rows(eng, db.LoadActual, source="smard") == 335
    assert db.count_rows(eng, db.LoadForecastOfficial, source="smard") == 335
    assert db.count_rows(eng, db.WeatherHourly) == 0


def test_cli_rejects_unknown_source(tmp_path) -> None:
    url = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    assert ingest.main(["--sources", "nope", "--database-url", url]) == 2


def test_cli_returns_nonzero_when_a_source_fails(tmp_path, monkeypatch) -> None:
    routes = smard_routes(SMARD_BASE, "DE")
    for url, payload in routes.items():
        if "410_DE_hour_" in url:
            payload["series"] = [[t, v / 100 if v else v] for t, v in payload["series"]]
    clients, _ = make_clients(smard_routes_override=routes)
    monkeypatch.setattr(ingest, "Clients", lambda: clients)
    url = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    rc = ingest.main(
        ["--backfill", "--start", "2021-03-21", "--end", "2021-04-05", "--database-url", url]
    )
    assert rc == 1


def test_summarize_lists_every_result() -> None:
    text = ingest.summarize(
        [
            ingest.IngestResult("smard", rows=5, start=pd.Timestamp("2021-01-01T00:00Z")),
            ingest.IngestResult("weather_forecast", status="skipped", message="no rows"),
        ]
    )
    assert "smard" in text and "weather_forecast" in text and "no rows" in text
