"""Ingestion pipeline + CLI.

``--backfill`` pulls the full history from ``DATA_START_DATE`` (or ``--start``) for
every requested source; the default, incremental mode pulls only what is new since the
latest stored hour, re-fetching a small overlap window (upserts make that safe).

Sources: ``smard`` (actual load and the official day-ahead load forecast) and
``weather`` (Open-Meteo), both keyless. Each source is validated before it is written
and runs independently, so one failing source never blocks the others. Every value the
official forecast has had is kept (``db.store_official_forecast``). With
``DD_API_KEY`` set, each run also reports rows per step, failed steps and the age of
the newest actual load, weather archive hour and model forecast to Datadog
(``src.observability``), and whether the weather forecast came from Open-Meteo's
fallback host.

A failed weather-archive *fetch* (``FetchError``: an HTTP error, a timeout, a connection
error, the retries giving up) in an incremental run is a warning, not a failure, while the
newest stored archive hour is at most ``WEATHER_ARCHIVE_MAX_AGE`` old: the next run re-reads
the last 7 days anyway. Anything else - a failed validation, an answer of an unexpected
shape, a database error - stays a failure.

``--gap-scan daily`` (the hourly ingest workflow) runs the gap scan in the run that starts in
the ``GAP_SCAN_HOUR_UTC`` hour; ``--gap-scan now`` runs it at once. The scan looks at the
whole stored history of actual load, official forecast and weather archive for holes older
than the overlap window - the part a normal run never re-reads - and re-fetches them,
within a fixed time budget (``run_gap_scan``). It never fails the job: any error in it
becomes a ``warning`` result. Daily is best effort: a day whose 02:00 UTC hour has no run
start is skipped, and two runs starting in that hour both scan - harmless, because the scan
is idempotent and bounded.

Examples::

    python -m src.data.ingest --backfill
    python -m src.data.ingest --backfill --sources smard --end 2021-03-31
    python -m src.data.ingest                       # incremental
    python -m src.data.ingest --gap-scan now        # incremental + gap scan
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from config.log import configure_logging
from config.settings import get_settings
from src.data import db
from src.data._http import DEFAULT_RETRY, FetchError, RetryPolicy
from src.data.smard_client import FORECAST_LOOKAHEAD, SmardClient
from src.data.validation import (
    WEATHER_MAX_NAN_SHARE,
    DataValidationError,
    Gap,
    find_gaps,
    validate_load,
    validate_weather,
)
from src.data.weather_client import OPEN_METEO_RETRY, SOURCE_ARCHIVE, WeatherClient
from src.observability import JobMetrics, job_metrics

log = logging.getLogger(__name__)

SOURCE_SMARD = db.SOURCE_SMARD
SOURCE_WEATHER = "weather"
ALL_SOURCES: tuple[str, ...] = (SOURCE_SMARD, SOURCE_WEATHER)
LOCAL_TZ = "Europe/Berlin"

# How far back an incremental run re-fetches, to pick up late corrections.
LOAD_OVERLAP = pd.Timedelta(hours=48)
# The archive's rows arrive the same day (measured 21 Nov 2026: values up to 23:00 UTC of the
# current day; src.data.weather_client). Recent values may still change - Open-Meteo's docs
# list its reanalysis as "daily with 5 days delay" behind a no-delay IFS - so every run
# re-reads 7 days to pick up revisions.
WEATHER_ARCHIVE_OVERLAP = pd.Timedelta(days=7)
# A failed archive fetch is only a warning while the newest stored archive hour is at most
# this old. Normal age is -23 h to +1 h (rows reach 23:00 UTC of the current day), so this
# is about one to two days of failed hourly fetches.
WEATHER_ARCHIVE_MAX_AGE = pd.Timedelta(hours=24)

# --- Gap scan ---
GAP_SCAN_OFF, GAP_SCAN_DAILY, GAP_SCAN_NOW = "off", "daily", "now"
GAP_SCAN_MODES: tuple[str, ...] = (GAP_SCAN_OFF, GAP_SCAN_DAILY, GAP_SCAN_NOW)
GAP_SCAN_HOUR_UTC = 2  # "daily" scans in the run that starts in this hour
# Old data is not urgent: two short tries per request, then leave it for the next day.
GAP_SCAN_RETRY = RetryPolicy(retries=2, backoff=3.0, timeout=(5.0, 10.0))
GAP_SCAN_BUDGET_SECONDS = 120.0  # no re-fetch starts after this
GAP_SCAN_MAX_WINDOWS = 10  # per source and scan; the rest waits for the next scan
# A re-fetch window spans at most 6 days, so it crosses at most one SMARD week boundary
# (weeks are >= 167 h): an index request + at most 2 week files; one Open-Meteo request.
GAP_SCAN_WINDOW = pd.Timedelta(days=6)
SMARD_REQUESTS_PER_WINDOW = 3


def gap_scan_worst_case_seconds() -> float:
    """The scan when every request stalls: re-fetches start until the budget is used, and the
    last one started may then take its own worst case (a SMARD window, the longest)."""
    return GAP_SCAN_BUDGET_SECONDS + SMARD_REQUESTS_PER_WINDOW * GAP_SCAN_RETRY.worst_case_seconds()


@dataclass
class IngestResult:
    source: str
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    status: str = "ok"  # ok | skipped | warning | failed
    message: str = ""
    error_kind: str = ""  # of a failed step: validation | fetch | other
    gauges: dict[str, float] = field(default_factory=dict)  # extra metrics of this step

    @property
    def failed(self) -> bool:
        return self.status == "failed"


@dataclass
class Clients:
    """Bundle of data clients; tests inject fakes here."""

    smard: SmardClient = field(default_factory=SmardClient)
    weather: WeatherClient = field(default_factory=WeatherClient)


def _span(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty:
        return None, None
    return df["timestamp_utc"].min(), df["timestamp_utc"].max()


def end_of_next_local_day(now: pd.Timestamp) -> pd.Timestamp:
    """The last UTC hour of tomorrow in Berlin - what the D+1 forecast needs weather for."""
    today = db.to_utc(now).tz_convert(LOCAL_TZ).normalize().tz_localize(None)
    end = (today + pd.Timedelta(days=2)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    return end - pd.Timedelta(hours=1)


# --- Per-source ingestion steps ---
def _validate_forecast(df: pd.DataFrame) -> None:
    """An official forecast is checked like actual load: same quantity, same sane range."""
    validate_load(df.rename(columns={"forecast_mw": "load_mw"}))


def _store(
    engine: Engine,
    name: str,
    df: pd.DataFrame,
    model: type[db.Base],
    *,
    validate: Callable[[pd.DataFrame], Any],
    source: str | None = None,
    update_where: db.UpdateWhere | None = None,
) -> IngestResult:
    """Validate one fetched batch and upsert it; an empty batch is reported as skipped."""
    if df.empty:
        return IngestResult(name, status="skipped", message="no rows")
    validate(df)
    if source is not None:
        df = df.assign(source=source)
    n = db.upsert_dataframe(engine, model, df, update_where=update_where)
    lo, hi = _span(df)
    return IngestResult(name, rows=n, start=lo, end=hi)


def _city_names(client: WeatherClient) -> list[str]:
    return [c.name for c in client.cities]


def ingest_smard(
    engine: Engine,
    client: SmardClient,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    policy: RetryPolicy = DEFAULT_RETRY,
) -> IngestResult:
    df = client.fetch_load(start, end, policy=policy)
    return _store(engine, "smard", df, db.LoadActual, validate=validate_load, source=SOURCE_SMARD)


def ingest_smard_forecast(
    engine: Engine,
    client: SmardClient,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    policy: RetryPolicy = DEFAULT_RETRY,
) -> IngestResult:
    """Official day-ahead load forecast (SMARD filter 411) -> ``load_forecast_official``,
    plus a row in ``load_forecast_official_revision`` for every hour new or changed."""
    df = client.fetch_forecast(start, end, policy=policy)
    if df.empty:
        return IngestResult("smard_forecast", status="skipped", message="no rows")
    _validate_forecast(df)
    n, revisions = db.store_official_forecast(engine, df.assign(source=SOURCE_SMARD))
    lo, hi = _span(df)
    return IngestResult(
        "smard_forecast", rows=n, start=lo, end=hi, message=f"{revisions} new/changed hours"
    )


def ingest_weather_history(
    engine: Engine,
    client: WeatherClient,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
    *,
    policy: RetryPolicy = OPEN_METEO_RETRY,
    max_nan_share: float = WEATHER_MAX_NAN_SHARE,
) -> IngestResult:
    """Archive weather for ``[start, end]``. The gap scan passes ``max_nan_share=1.0``: its
    windows may hold hours the provider never fills, which must not reject the rest."""
    df = client.fetch_historical(start, end, policy=policy)
    return _store(
        engine,
        "weather_archive",
        df,
        db.WeatherHourly,
        validate=partial(
            validate_weather, expected_cities=_city_names(client), max_nan_share=max_nan_share
        ),
        update_where=db.weather_update_where,
    )


def ingest_weather_forecast(
    engine: Engine,
    client: WeatherClient,
    *,
    past_days: int = 2,
    forecast_days: int = 3,
    now: pd.Timestamp | None = None,
) -> IngestResult:
    """The Open-Meteo forecast; it must reach the end of tomorrow (Berlin) to be stored."""
    df = client.fetch_forecast(past_days=past_days, forecast_days=forecast_days)
    until = end_of_next_local_day(now if now is not None else pd.Timestamp.now(tz="UTC"))
    result = _store(
        engine,
        "weather_forecast",
        df,
        db.WeatherHourly,
        validate=partial(
            validate_weather, expected_cities=_city_names(client), forecast_until=until
        ),
        update_where=db.weather_update_where,
    )
    if client.forecast_fallback_used is not None:
        result.gauges["weather.forecast_fallback"] = int(client.forecast_fallback_used)
    return result


# --- Modes ---
ERROR_VALIDATION, ERROR_FETCH, ERROR_OTHER = "validation", "fetch", "other"


def _guard(source: str, fn: Callable[..., IngestResult], *args: Any, **kwargs: Any) -> IngestResult:
    """Run one ingestion step; convert any failure into a ``failed`` result that records
    its ``error_kind``."""
    try:
        return fn(*args, **kwargs)
    except DataValidationError as err:
        log.error("%s: validation failed - nothing written: %s", source, err)
        return IngestResult(
            source, status="failed", message=f"validation: {err}", error_kind=ERROR_VALIDATION
        )
    except FetchError as err:
        log.error("%s: fetch failed: %s", source, err)
        return IngestResult(source, status="failed", message=str(err), error_kind=ERROR_FETCH)
    except Exception as err:  # keep other sources running
        log.exception("%s: ingestion failed", source)
        return IngestResult(source, status="failed", message=str(err), error_kind=ERROR_OTHER)


def archive_age(engine: Engine, now: pd.Timestamp) -> pd.Timedelta | None:
    """How far the newest stored archive hour lies before ``now`` (negative: after), or None."""
    latest = db.latest_timestamp(engine, db.WeatherHourly, source=SOURCE_ARCHIVE)
    return None if latest is None else db.to_utc(now) - latest


def tolerate_archive_failure(
    engine: Engine, result: IngestResult, now: pd.Timestamp
) -> IngestResult:
    """Turn a failed archive *fetch* into a warning while the stored archive is fresh enough.

    Only ``FetchError`` (HTTP errors, timeouts, connection errors, the retries giving up)
    counts. A failed validation, an answer of an unexpected shape, a database error, and an
    archive older than ``WEATHER_ARCHIVE_MAX_AGE`` (or none at all) stay failures: those are
    not noise.
    """
    if not result.failed or result.error_kind != ERROR_FETCH:
        return result
    age = archive_age(engine, now)
    limit_h = WEATHER_ARCHIVE_MAX_AGE / pd.Timedelta(hours=1)
    if age is None or age > WEATHER_ARCHIVE_MAX_AGE:
        shown = (
            "no archive stored"
            if age is None
            else f"archive {age / pd.Timedelta(hours=1):.1f} h old"
        )
        log.error("weather_archive: fetch failed and %s (limit %.0f h)", shown, limit_h)
        result.message = f"{shown}, above the {limit_h:.0f} h limit: {result.message}"
        return result
    age_h = age / pd.Timedelta(hours=1)
    log.warning(
        "weather_archive: fetch failed, but the stored archive is %.1f h old (limit %.0f h) - "
        "warning only; the next run re-reads the last %d days",
        age_h,
        limit_h,
        WEATHER_ARCHIVE_OVERLAP.days,
    )
    result.status = "warning"
    result.message = f"fetch failed, archive {age_h:.1f} h old: {result.message}"
    return result


def _resume_from(
    engine: Engine,
    model: type[db.Base],
    overlap: pd.Timedelta,
    default: pd.Timestamp,
    **filters: Any,
) -> pd.Timestamp:
    """Latest stored hour minus ``overlap`` (to pick up late corrections), else ``default``."""
    latest = db.latest_timestamp(engine, model, **filters)
    return default if latest is None else latest - overlap


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
        results.append(_guard("smard", ingest_smard, engine, clients.smard, start_ts, end_ts))
        results.append(
            _guard("smard_forecast", ingest_smard_forecast, engine, clients.smard, start_ts, end_ts)
        )
    if SOURCE_WEATHER in sources:
        results.append(
            _guard(
                "weather_archive", ingest_weather_history, engine, clients.weather, start_ts, end_ts
            )
        )
    return results


def should_scan(mode: str, now: pd.Timestamp) -> bool:
    if mode not in GAP_SCAN_MODES:
        raise ValueError(f"gap scan mode must be one of {GAP_SCAN_MODES}, got {mode!r}")
    return mode == GAP_SCAN_NOW or (mode == GAP_SCAN_DAILY and now.hour == GAP_SCAN_HOUR_UTC)


def run_incremental(
    engine: Engine,
    clients: Clients,
    *,
    sources: Sequence[str] = ALL_SOURCES,
    now: pd.Timestamp | None = None,
    gap_scan: str = GAP_SCAN_OFF,
) -> list[IngestResult]:
    """Pull only new hours since the latest stored timestamp per source (+ the gap scan)."""
    settings = get_settings()
    now_ts = now or pd.Timestamp.now(tz="UTC")
    scan = should_scan(gap_scan, now_ts)
    forecast_end = now_ts + FORECAST_LOOKAHEAD
    default_start = db.to_utc(settings.data_start_date)
    log.info("=== incremental ingestion at %s | sources=%s", now_ts, ",".join(sources))

    db.init_db(engine)
    results: list[IngestResult] = []

    if SOURCE_SMARD in sources:
        start = _resume_from(
            engine, db.LoadActual, LOAD_OVERLAP, default_start, source=SOURCE_SMARD
        )
        fc_start = _resume_from(
            engine, db.LoadForecastOfficial, LOAD_OVERLAP, default_start, source=SOURCE_SMARD
        )
        results.append(_guard("smard", ingest_smard, engine, clients.smard, start, now_ts))
        results.append(
            _guard(
                "smard_forecast",
                ingest_smard_forecast,
                engine,
                clients.smard,
                fc_start,
                forecast_end,
            )
        )

    if SOURCE_WEATHER in sources:
        start = _resume_from(
            engine,
            db.WeatherHourly,
            WEATHER_ARCHIVE_OVERLAP,
            default_start,
            source=SOURCE_ARCHIVE,
        )
        archive = _guard(
            "weather_archive", ingest_weather_history, engine, clients.weather, start, now_ts
        )
        results.append(tolerate_archive_failure(engine, archive, now_ts))
        results.append(
            _guard("weather_forecast", ingest_weather_forecast, engine, clients.weather, now=now_ts)
        )

    if scan:
        try:
            results += run_gap_scan(engine, clients, sources=sources)
        except Exception as err:  # the scan never fails the run
            log.exception("gap scan failed")
            results.append(IngestResult("gap_scan", status="warning", message=str(err)))
    return results


# --- Gap scan ---
@dataclass
class GapSource:
    """One table the gap scan checks, and how to re-fetch a window of it."""

    step: str
    model: type[db.Base]
    overlap: pd.Timedelta
    filters: dict[str, Any]
    refetch: Callable[[pd.Timestamp, pd.Timestamp], IngestResult]
    cities: list[str] | None = None


def gap_sources(engine: Engine, clients: Clients, sources: Sequence[str]) -> list[GapSource]:
    out: list[GapSource] = []
    smard = {"source": SOURCE_SMARD}
    if SOURCE_SMARD in sources:
        out.append(
            GapSource(
                "smard_gaps",
                db.LoadActual,
                LOAD_OVERLAP,
                smard,
                partial(ingest_smard, engine, clients.smard, policy=GAP_SCAN_RETRY),
            )
        )
        out.append(
            GapSource(
                "smard_forecast_gaps",
                db.LoadForecastOfficial,
                LOAD_OVERLAP,
                smard,
                partial(ingest_smard_forecast, engine, clients.smard, policy=GAP_SCAN_RETRY),
            )
        )
    if SOURCE_WEATHER in sources:
        out.append(
            GapSource(
                "weather_archive_gaps",
                db.WeatherHourly,
                WEATHER_ARCHIVE_OVERLAP,
                {"source": SOURCE_ARCHIVE},
                partial(
                    ingest_weather_history,
                    engine,
                    clients.weather,
                    policy=GAP_SCAN_RETRY,
                    max_nan_share=1.0,  # missing values only warn (city set, ranges still fail)
                ),
                cities=_city_names(clients.weather),
            )
        )
    return out


def refetch_windows(
    gaps: Sequence[Gap], max_span: pd.Timedelta = GAP_SCAN_WINDOW
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Group holes into re-fetch windows ``[start, end]`` of at most ``max_span``.

    A window runs from the present hour before a hole to the present hour after it; holes
    close together share a window and a long hole is split. Chronological order.
    """
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for gap in gaps:
        start, end = gap.start, gap.end
        if windows and end - windows[-1][0] <= max_span:
            windows[-1] = (windows[-1][0], end)
            continue
        while end - start > max_span:
            windows.append((start, start + max_span))
            start += max_span
        windows.append((start, end))
    return windows


def _missing_hours(
    engine: Engine, source: GapSource, start: pd.Timestamp, end: pd.Timestamp
) -> int:
    """Hours in ``(start, end]`` still without a (complete) row.

    ``start`` is a stored hour or the end of the previous window of a split hole, which that
    window counts; ``end`` is a stored hour unless the hole was split there - so every hour
    is counted by exactly one window.
    """
    have = db.covered_hours(
        engine, source.model, start=start, end=end, cities=source.cities, **source.filters
    )
    inside = pd.date_range(start, end, freq="h", inclusive="right")
    return int((~inside.isin(have)).sum())


def _plan(
    engine: Engine, source: GapSource
) -> tuple[IngestResult, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """The scan result of one table and its re-fetch windows, newest first."""
    latest = db.latest_timestamp(engine, source.model, **source.filters)
    if latest is None:
        return IngestResult(source.step, status="skipped", message="no rows"), []
    hours = db.covered_hours(engine, source.model, cities=source.cities, **source.filters)
    boundary = latest - source.overlap
    gaps = [g for g in find_gaps(pd.Series(hours)) if g.start < boundary]
    missing = sum(g.missing_hours for g in gaps)
    windows = refetch_windows(gaps)[::-1]  # newest first: old permanent holes cannot starve
    result = IngestResult(source.step, gauges={"ingest.gap_hours": missing})
    result.message = f"{len(gaps)} holes, {missing} h older than the overlap"
    if gaps:
        result.start, result.end = gaps[0].start, gaps[-1].end
        log.warning(
            "%s: %d hole(s), %d missing hour(s) older than the %s overlap; %d window(s)",
            source.step,
            len(gaps),
            missing,
            source.overlap,
            len(windows),
        )
    return result, windows


def run_gap_scan(
    engine: Engine,
    clients: Clients,
    *,
    sources: Sequence[str] = ALL_SOURCES,
    budget_seconds: float = GAP_SCAN_BUDGET_SECONDS,
    max_windows: int = GAP_SCAN_MAX_WINDOWS,
    clock: Callable[[], float] = time.monotonic,
) -> list[IngestResult]:
    """Find holes older than the overlap window in the whole stored history and re-fetch them.

    Per table: one grouped query for the hours present (for weather: an archive row for
    every city; forecast rows do not count), ``find_gaps``, keep the holes that start before
    ``latest - overlap`` (a normal run re-reads the rest), group them into windows
    (``refetch_windows``), at most ``max_windows`` per table, newest first (permanent old
    holes cannot starve newer ones). The windows of all tables take turns; no new re-fetch
    starts once ``budget_seconds`` have passed, and the rest waits for the next scan.
    Requests use ``GAP_SCAN_RETRY``. Each re-fetch goes through the normal, validated step
    (for weather, missing values only warn); holes the provider cannot fill are logged as
    "still missing after re-fetch". ``ingest.gap_hours`` reports the missing hours the scan
    found, before re-fetching. Any error - in a table's queries or a re-fetch - is a
    ``warning`` result, never an exception.
    """
    started = clock()
    results: list[IngestResult] = []
    queues: list[list[tuple[GapSource, IngestResult, tuple[pd.Timestamp, pd.Timestamp]]]] = []
    for source in gap_sources(engine, clients, sources):
        try:
            result, windows = _plan(engine, source)
        except Exception as err:  # e.g. a statement timeout on the full-history query
            log.warning("%s: gap scan of this table failed: %s", source.step, err)
            results.append(IngestResult(source.step, status="warning", message=str(err)))
            continue
        results.append(result)
        if not windows:
            continue
        queues.append([(source, result, w) for w in windows[:max_windows]])
        if len(windows) > max_windows:
            log.warning(
                "%s: %d window(s) beyond the cap of %d left for the next scan",
                source.step,
                len(windows) - max_windows,
                max_windows,
            )

    done: dict[str, int] = {}
    turns = [t for row in itertools.zip_longest(*queues) for t in row if t is not None]
    for i, (source, result, (start, end)) in enumerate(turns):
        if clock() - started >= budget_seconds:
            log.warning(
                "gap scan: the %.0f s budget is used - %d re-fetch window(s) left for the next "
                "scan",
                budget_seconds,
                len(turns) - i,
            )
            break
        done[source.step] = done.get(source.step, 0) + 1
        try:
            fetched = source.refetch(start, end)
            result.rows += fetched.rows
        except Exception as err:  # old data: never fail the job, try again next scan
            log.warning("%s: re-fetch %s -> %s failed: %s", source.step, start, end, err)
            result.status = "warning"
            continue
        try:
            still = _missing_hours(engine, source, start, end)
        except Exception as err:
            log.warning("%s: could not re-check %s -> %s: %s", source.step, start, end, err)
            result.status = "warning"
            continue
        if still:
            log.warning(
                "%s: %d hour(s) between %s and %s still missing after re-fetch - the provider "
                "has no data for them",
                source.step,
                still,
                start,
                end,
            )

    for row in queues:
        if row:
            source, result, _ = row[0]
            result.message += (
                f"; re-fetched {done.get(source.step, 0)} of {len(row)} window(s), "
                f"{result.rows} rows"
            )
    return results


def summarize(results: Sequence[IngestResult]) -> str:
    lines = []
    for r in results:
        span = f"{r.start} -> {r.end}" if r.start is not None else "-"
        lines.append(f"  {r.source:<20} {r.status:<8} rows={r.rows:<7} {span} {r.message}")
    return "\n".join(lines)


# --- CLI ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.ingest",
        description="Ingest German load (SMARD) and weather (Open-Meteo).",
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
    p.add_argument(
        "--gap-scan",
        choices=GAP_SCAN_MODES,
        default=GAP_SCAN_OFF,
        help=(
            f"incremental runs only: re-fetch holes older than the overlap - '{GAP_SCAN_DAILY}' "
            f"in the run that starts in the {GAP_SCAN_HOUR_UTC:02d}:00 UTC hour, "
            f"'{GAP_SCAN_NOW}' at once (default: {GAP_SCAN_OFF})"
        ),
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

    with job_metrics("ingest") as metrics:
        engine = db.get_engine(args.database_url)
        clients = Clients()
        if args.backfill:
            results = run_backfill(engine, clients, start=args.start, end=args.end, sources=sources)
        else:
            results = run_incremental(engine, clients, sources=sources, gap_scan=args.gap_scan)

        log.info("ingestion summary:\n%s", summarize(results))
        report_metrics(metrics, engine, results)
    return 1 if any(r.failed for r in results) else 0


def report_metrics(metrics: JobMetrics, engine: Engine, results: Sequence[IngestResult]) -> None:
    """Rows per step, failed steps, each step's extra gauges (tagged with the step), and how
    old the newest actual load, weather archive hour and forecast are.

    The hourly ingest doubles as a watchdog for the daily forecast: alerting on the
    forecast's age catches a forecast job that never started, which the job itself
    cannot report. The archive's age is what still shows a real archive outage now that
    a single failed fetch is only a warning.
    """
    for r in results:
        metrics.gauge("ingest.rows", r.rows, step=r.source)
        metrics.gauge("ingest.step_failed", int(r.failed), step=r.source)
        for name, value in r.gauges.items():
            metrics.gauge(name, value, step=r.source)
    if any(r.failed for r in results):
        metrics.fail()
    now = pd.Timestamp.now(tz="UTC")
    latest = db.latest_timestamp(engine, db.LoadActual, source=SOURCE_SMARD)
    if latest is not None:
        age = (now - latest) / pd.Timedelta(hours=1)
        metrics.gauge("data.freshness_hours", age)
    archive = archive_age(engine, now)
    if archive is not None:
        metrics.gauge("weather.archive_age_hours", archive / pd.Timedelta(hours=1))
    issued = db.latest_issued_at(engine)
    if issued is not None:
        age = (now - issued) / pd.Timedelta(hours=1)
        metrics.gauge("forecast.age_hours", age)


if __name__ == "__main__":
    sys.exit(main())
