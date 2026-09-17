"""Ingestion pipeline + CLI.

``--backfill`` pulls the full history from ``DATA_START_DATE`` (or ``--start``) for
every requested source; the default, incremental mode pulls only what is new since the
latest stored hour, re-fetching a small overlap window (upserts make that safe).

Sources: ``smard`` (actual load and the official day-ahead load forecast, both
keyless), ``weather`` (Open-Meteo, keyless) and ``entsoe`` (actual load + official
forecast as a second source; silently skipped without a token). Each source is
validated before it is written and runs independently, so one failing source never
blocks the others.

Examples::

    python -m src.data.ingest --backfill
    python -m src.data.ingest --backfill --sources smard --end 2021-03-31
    python -m src.data.ingest                       # incremental
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from config.log import configure_logging
from config.settings import get_settings
from src.data import db
from src.data.entsoe_client import EntsoeClient
from src.data.smard_client import SmardClient
from src.data.validation import DataValidationError, validate_load, validate_weather
from src.data.weather_client import SOURCE_ARCHIVE, WeatherClient

log = logging.getLogger(__name__)

SOURCE_SMARD = db.SOURCE_SMARD
SOURCE_ENTSOE = db.SOURCE_ENTSOE
SOURCE_WEATHER = "weather"
ALL_SOURCES: tuple[str, ...] = (SOURCE_SMARD, SOURCE_WEATHER, SOURCE_ENTSOE)

# How far back an incremental run re-fetches, to pick up late corrections.
LOAD_OVERLAP = pd.Timedelta(hours=48)
WEATHER_ARCHIVE_OVERLAP = pd.Timedelta(days=7)  # the archive lags a few days


@dataclass
class IngestResult:
    source: str
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    status: str = "ok"  # ok | skipped | failed
    message: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "failed"


@dataclass
class Clients:
    """Bundle of data clients; tests inject fakes here."""

    smard: SmardClient = field(default_factory=SmardClient)
    weather: WeatherClient = field(default_factory=WeatherClient)
    entsoe: EntsoeClient = field(default_factory=EntsoeClient)


def _span(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty:
        return None, None
    return df["timestamp_utc"].min(), df["timestamp_utc"].max()


# --- Per-source ingestion steps ---
def ingest_smard(
    engine: Engine, client: SmardClient, start: pd.Timestamp, end: pd.Timestamp | None = None
) -> IngestResult:
    df = client.fetch_load(start, end)
    if df.empty:
        return IngestResult(SOURCE_SMARD, status="skipped", message="no new rows")
    validate_load(df)
    n = db.upsert_dataframe(engine, db.LoadActual, df.assign(source=SOURCE_SMARD))
    lo, hi = _span(df)
    return IngestResult(SOURCE_SMARD, rows=n, start=lo, end=hi)


def ingest_smard_forecast(
    engine: Engine, client: SmardClient, start: pd.Timestamp, end: pd.Timestamp | None = None
) -> IngestResult:
    """Official day-ahead load forecast (SMARD filter 411) -> ``load_forecast_official``."""
    df = client.fetch_forecast(start, end)
    if df.empty:
        return IngestResult("smard_forecast", status="skipped", message="no new rows")
    validate_load(df.rename(columns={"forecast_mw": "load_mw"}))  # same sanity range
    n = db.upsert_dataframe(engine, db.LoadForecastOfficial, df.assign(source=SOURCE_SMARD))
    lo, hi = _span(df)
    return IngestResult("smard_forecast", rows=n, start=lo, end=hi)


def ingest_weather_history(
    engine: Engine, client: WeatherClient, start: pd.Timestamp, end: pd.Timestamp | None = None
) -> IngestResult:
    df = client.fetch_historical(start, end)
    if df.empty:
        return IngestResult("weather_archive", status="skipped", message="no new rows")
    validate_weather(df)
    n = db.upsert_dataframe(engine, db.WeatherHourly, df, update_where=db.weather_update_where)
    lo, hi = _span(df)
    return IngestResult("weather_archive", rows=n, start=lo, end=hi)


def ingest_weather_forecast(
    engine: Engine, client: WeatherClient, *, past_days: int = 2, forecast_days: int = 3
) -> IngestResult:
    df = client.fetch_forecast(past_days=past_days, forecast_days=forecast_days)
    if df.empty:
        return IngestResult("weather_forecast", status="skipped", message="no rows")
    validate_weather(df)
    n = db.upsert_dataframe(engine, db.WeatherHourly, df, update_where=db.weather_update_where)
    lo, hi = _span(df)
    return IngestResult("weather_forecast", rows=n, start=lo, end=hi)


def ingest_entsoe(
    engine: Engine,
    client: EntsoeClient,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    forecast_start: pd.Timestamp | None = None,
    forecast_end: pd.Timestamp | None = None,
) -> list[IngestResult]:
    if not client.available:
        msg = "no ENTSOE_API_TOKEN configured - skipped (SMARD covers actual load)"
        log.info("ENTSO-E %s", msg)
        return [
            IngestResult("entsoe_actual", status="skipped", message=msg),
            IngestResult("entsoe_forecast", status="skipped", message=msg),
        ]

    results: list[IngestResult] = []

    actual = client.fetch_actual_load(start, end)
    if actual.empty:
        results.append(IngestResult("entsoe_actual", status="skipped", message="no rows"))
    else:
        validate_load(actual)
        n = db.upsert_dataframe(engine, db.LoadActual, actual.assign(source=SOURCE_ENTSOE))
        lo, hi = _span(actual)
        results.append(IngestResult("entsoe_actual", rows=n, start=lo, end=hi))

    f_start = forecast_start if forecast_start is not None else start
    f_end = forecast_end if forecast_end is not None else end
    forecast = client.fetch_dayahead_forecast(f_start, f_end)
    if forecast.empty:
        results.append(IngestResult("entsoe_forecast", status="skipped", message="no rows"))
    else:
        # Same sanity checks as actual load (it forecasts the same quantity).
        validate_load(forecast.rename(columns={"forecast_mw": "load_mw"}))
        n = db.upsert_dataframe(
            engine, db.LoadForecastOfficial, forecast.assign(source=SOURCE_ENTSOE)
        )
        lo, hi = _span(forecast)
        results.append(IngestResult("entsoe_forecast", rows=n, start=lo, end=hi))
    return results


# --- Modes ---
def _guard(
    source: str, fn: Callable[..., IngestResult | list[IngestResult]], *args: Any, **kwargs: Any
) -> list[IngestResult]:
    """Run one ingestion step; convert any failure into a ``failed`` result."""
    try:
        out = fn(*args, **kwargs)
        return out if isinstance(out, list) else [out]
    except DataValidationError as err:
        log.error("%s: validation failed - nothing written: %s", source, err)
        return [IngestResult(source, status="failed", message=f"validation: {err}")]
    except Exception as err:  # keep other sources running
        log.exception("%s: ingestion failed", source)
        return [IngestResult(source, status="failed", message=str(err))]


def run_backfill(
    engine: Engine,
    clients: Clients,
    *,
    start: date | str | pd.Timestamp | None = None,
    end: date | str | pd.Timestamp | None = None,
    sources: Sequence[str] = ALL_SOURCES,
) -> list[IngestResult]:
    """Pull full history from ``start`` (default ``DATA_START_DATE``) for ``sources``."""
    settings = get_settings()
    start_ts = db.to_utc(start or settings.data_start_date)
    end_ts = db.to_utc(end) if end is not None else None
    log.info("=== backfill %s -> %s | sources=%s", start_ts, end_ts or "now", ",".join(sources))

    db.init_db(engine)
    results: list[IngestResult] = []
    if SOURCE_SMARD in sources:
        results += _guard("smard", ingest_smard, engine, clients.smard, start_ts, end_ts)
        fc_end = end_ts if end_ts is not None else pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
        results += _guard(
            "smard_forecast", ingest_smard_forecast, engine, clients.smard, start_ts, fc_end
        )
    if SOURCE_WEATHER in sources:
        results += _guard(
            "weather_archive", ingest_weather_history, engine, clients.weather, start_ts, end_ts
        )
    if SOURCE_ENTSOE in sources:
        results += _guard("entsoe", ingest_entsoe, engine, clients.entsoe, start_ts, end_ts)
    return results


def run_incremental(
    engine: Engine,
    clients: Clients,
    *,
    sources: Sequence[str] = ALL_SOURCES,
    now: pd.Timestamp | None = None,
) -> list[IngestResult]:
    """Pull only new hours since the latest stored timestamp per source."""
    settings = get_settings()
    now_ts = now or pd.Timestamp.now(tz="UTC")
    default_start = db.to_utc(settings.data_start_date)
    log.info("=== incremental ingestion at %s | sources=%s", now_ts, ",".join(sources))

    db.init_db(engine)
    results: list[IngestResult] = []

    if SOURCE_SMARD in sources:
        latest = db.latest_timestamp(engine, db.LoadActual, source=SOURCE_SMARD)
        start = (latest - LOAD_OVERLAP) if latest is not None else default_start
        results += _guard("smard", ingest_smard, engine, clients.smard, start, now_ts)
        latest_fc = db.latest_timestamp(engine, db.LoadForecastOfficial, source=SOURCE_SMARD)
        fc_start = (latest_fc - LOAD_OVERLAP) if latest_fc is not None else default_start
        results += _guard(
            "smard_forecast",
            ingest_smard_forecast,
            engine,
            clients.smard,
            fc_start,
            now_ts + pd.Timedelta(days=2),  # tomorrow's forecast is already published
        )

    if SOURCE_WEATHER in sources:
        latest = db.latest_timestamp(engine, db.WeatherHourly, source=SOURCE_ARCHIVE)
        start = (latest - WEATHER_ARCHIVE_OVERLAP) if latest is not None else default_start
        results += _guard(
            "weather_archive", ingest_weather_history, engine, clients.weather, start, now_ts
        )
        results += _guard("weather_forecast", ingest_weather_forecast, engine, clients.weather)

    if SOURCE_ENTSOE in sources:
        latest_actual = db.latest_timestamp(engine, db.LoadActual, source=SOURCE_ENTSOE)
        latest_fc = db.latest_timestamp(engine, db.LoadForecastOfficial, source=SOURCE_ENTSOE)
        a_start = (latest_actual - LOAD_OVERLAP) if latest_actual is not None else default_start
        f_start = (latest_fc - LOAD_OVERLAP) if latest_fc is not None else default_start
        results += _guard(
            "entsoe",
            ingest_entsoe,
            engine,
            clients.entsoe,
            a_start,
            now_ts,
            forecast_start=f_start,
            forecast_end=now_ts + pd.Timedelta(days=2),  # day-ahead is published for tomorrow
        )
    return results


def summarize(results: Sequence[IngestResult]) -> str:
    lines = []
    for r in results:
        span = f"{r.start} -> {r.end}" if r.start is not None else "-"
        lines.append(f"  {r.source:<17} {r.status:<8} rows={r.rows:<7} {span} {r.message}")
    return "\n".join(lines)


# --- CLI ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.ingest",
        description="Ingest German load (SMARD / ENTSO-E) and weather (Open-Meteo).",
    )
    p.add_argument("--backfill", action="store_true", help="pull full history from --start")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default DATA_START_DATE)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: now)")
    p.add_argument(
        "--sources",
        type=str,
        default=",".join(ALL_SOURCES),
        help=f"comma-separated subset of {','.join(ALL_SOURCES)}",
    )
    p.add_argument("--database-url", type=str, default=None, help="override DATABASE_URL")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    sources = tuple(s.strip().lower() for s in args.sources.split(",") if s.strip())
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        log.error("unknown sources: %s (choose from %s)", sorted(unknown), ALL_SOURCES)
        return 2

    engine = db.get_engine(args.database_url)
    clients = Clients()
    if args.backfill:
        results = run_backfill(engine, clients, start=args.start, end=args.end, sources=sources)
    else:
        results = run_incremental(engine, clients, sources=sources)

    log.info("ingestion summary:\n%s", summarize(results))
    return 1 if any(r.failed for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
