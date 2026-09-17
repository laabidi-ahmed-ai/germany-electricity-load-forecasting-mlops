"""End-to-end ingestion tests (backfill + incremental) with fake clients and SQLite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import db, ingest
from src.data.entsoe_client import EntsoeClient
from src.data.smard_client import SmardClient
from src.data.weather_client import City, WeatherClient
from tests.conftest import (
    DST_WEEK_STARTS_MS,
    FakeSession,
    open_meteo_route,
    smard_routes,
)
from tests.test_entsoe_client import FakePandasClient

SMARD_BASE = "https://smard.test/app/chart_data"
ARCHIVE = "https://archive.test/v1/archive"
FORECAST_BASE = "https://forecast.test/v1"
CITIES = (City("A", 50.0, 8.0), City("B", 52.0, 13.0))


def make_clients(
    *,
    weeks: dict[int, int] | None = None,
    weather_end: str = "2021-04-04T21:00",
    entsoe_token: str | None = None,
    smard_routes_override: dict | None = None,
) -> tuple[ingest.Clients, FakeSession]:
    routes = smard_routes_override or smard_routes(SMARD_BASE, "DE", weeks)
    routes[ARCHIVE] = open_meteo_route("2021-03-21T23:00", weather_end)
    routes[f"{FORECAST_BASE}/forecast"] = open_meteo_route("2021-04-03T00:00", "2021-04-07T23:00")
    session = FakeSession(routes=routes)
    clients = ingest.Clients(
        smard=SmardClient(session=session, base_url=SMARD_BASE, region="DE"),
        weather=WeatherClient(
            session=session, cities=CITIES, archive_url=ARCHIVE, forecast_base_url=FORECAST_BASE
        ),
        entsoe=EntsoeClient(token=entsoe_token, client_factory=lambda _: FakePandasClient()),
    )
    return clients, session


def by_source(results: list[ingest.IngestResult]) -> dict[str, ingest.IngestResult]:
    return {r.source: r for r in results}


# --------------------------------------------------------------------------- #
# backfill
# --------------------------------------------------------------------------- #
def test_backfill_without_token_loads_smard_and_weather_and_skips_entsoe(engine) -> None:
    clients, _ = make_clients()
    results = by_source(ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-04-05"))

    assert results["smard"].status == "ok" and results["smard"].rows == 335
    assert results["smard_forecast"].status == "ok" and results["smard_forecast"].rows == 335
    assert results["weather_archive"].status == "ok"
    assert results["entsoe_actual"].status == "skipped"
    assert results["entsoe_forecast"].status == "skipped"
    assert not any(r.failed for r in results.values())

    load = db.read_load(engine, source="smard")
    assert len(load) == 335
    assert 50_000 < load["load_mw"].mean() < 60_000
    assert load["timestamp_utc"].min() == pd.Timestamp("2021-03-21T23:00Z")
    assert db.count_rows(engine, db.LoadActual, source="entsoe") == 0
    # The official day-ahead forecast (SMARD 411) lands in load_forecast_official.
    official = db.read_table(engine, db.LoadForecastOfficial, source="smard")
    assert len(official) == 335
    merged = load.merge(official, on="timestamp_utc")
    np.testing.assert_allclose(merged["forecast_mw"], merged["load_mw"] * 0.98)
    assert db.count_rows(engine, db.LoadForecastOfficial, source="entsoe") == 0
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


def test_backfill_with_token_stores_entsoe_actual_and_official_forecast(engine) -> None:
    clients, _ = make_clients(entsoe_token="tok")
    results = by_source(
        ingest.run_backfill(
            engine, clients, start="2021-03-21", end="2021-03-23", sources=("entsoe",)
        )
    )
    assert results["entsoe_actual"].status == "ok" and results["entsoe_actual"].rows == 48
    assert results["entsoe_forecast"].status == "ok" and results["entsoe_forecast"].rows == 48
    assert db.count_rows(engine, db.LoadActual, source="entsoe") == 48
    assert db.count_rows(engine, db.LoadForecastOfficial, source="entsoe") == 48
    assert db.count_rows(engine, db.LoadActual, source="smard") == 0


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


# --------------------------------------------------------------------------- #
# incremental
# --------------------------------------------------------------------------- #
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
    assert results["entsoe_actual"].status == "skipped"
    assert db.count_rows(engine, db.LoadForecastOfficial, source="smard") == 335

    # No duplicates, everything contiguous in UTC.
    load = db.read_load(engine, source="smard")
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


def test_incremental_with_token_queries_entsoe_from_latest(engine) -> None:
    clients, _ = make_clients(entsoe_token="tok")
    ingest.run_backfill(engine, clients, start="2021-03-21", end="2021-03-23", sources=("entsoe",))
    now = pd.Timestamp("2021-03-24T00:00Z")
    results = by_source(ingest.run_incremental(engine, clients, sources=("entsoe",), now=now))
    assert results["entsoe_actual"].status == "ok"
    assert results["entsoe_forecast"].status == "ok"
    # actual: from latest(03-22T23) - 48h = 03-21T23 -> 03-24T00 => 49 rows, no duplicates
    assert db.count_rows(engine, db.LoadActual, source="entsoe") == 73  # 03-21T00 .. 03-24T00
    # official forecast is fetched two days into the future
    latest_fc = db.latest_timestamp(engine, db.LoadForecastOfficial)
    assert latest_fc == now + pd.Timedelta(days=2) - pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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
            ingest.IngestResult("entsoe_actual", status="skipped", message="no token"),
        ]
    )
    assert "smard" in text and "entsoe_actual" in text and "no token" in text
