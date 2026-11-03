"""Batch day-ahead forecast job: champion model -> ``load_forecast_model`` table.

Meant to run every morning after the incremental ingestion (``forecast.yml``, started
by an external scheduler with GitHub's cron as a backup). Every run is a new issue: its
rows are added, never overwrite an earlier issue, and are scored only if issued by the
horizon's deadline (for D+1, 12:00 Berlin the day before; a later run is logged as late).
``--once-per-day`` makes a late backup run a no-op when today's forecast was already
issued. The monitoring jobs compare this table against the actuals and the official
day-ahead forecast.

CLI (``make forecast``)::

    python -m src.serving.batch_forecast [--hours 24] [--as-of 2026-09-10T22:00Z] [--dry-run]
                                         [--once-per-day]

``--as-of`` replays a past issue and needs ``--dry-run``: stored, it would be a new issue
long after its deadline - never scored, yet the "latest forecast" the dashboard and API show.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config.log import configure_logging
from src.data import db
from src.data.weather_client import WeatherClient
from src.models.registry import load_champion
from src.observability import JobMetrics, job_metrics
from src.serving.forecast import (
    DAY_AHEAD_HOURS,
    ForecastResult,
    make_day_ahead_forecast,
    store_forecast,
)

log = logging.getLogger(__name__)

AS_OF_NEEDS_DRY_RUN = (
    "--as-of replays a past issue and is only allowed with --dry-run: stored, it would be a "
    "late issue that is never scored but shows as the latest forecast"
)


def run(
    *,
    hours: int = DAY_AHEAD_HOURS,
    as_of: pd.Timestamp | None = None,
    dry_run: bool = False,
    once_per_day: bool = False,
    database_url: str | None = None,
) -> ForecastResult | None:
    """Forecast and store; ``None`` when ``once_per_day`` finds today's forecast already issued."""
    if as_of is not None and not dry_run:
        raise ValueError(AS_OF_NEEDS_DRY_RUN)
    engine = db.get_engine(database_url)
    db.init_db(engine)
    if once_per_day:
        issued = db.latest_issued_at(engine)
        if db.happened_today(issued):
            log.info("today's forecast was already issued at %s - skipping", issued)
            return None
    model = load_champion(engine)
    result = make_day_ahead_forecast(
        engine, model, weather_client=WeatherClient(), hours=hours, as_of=as_of
    )
    if result.deadline_margin_hours < 0:
        log.warning(
            "issued %.1f h after its deadline: the forecast is kept but its late hours are "
            "not scored",
            -result.deadline_margin_hours,
        )
    if dry_run:
        log.info("dry run - %d rows not written", len(result.frame))
    else:
        n = store_forecast(engine, result)
        log.info("stored %d forecast rows (%s)", n, model.version_label)
    return result


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(prog="python -m src.serving.batch_forecast")
    p.add_argument("--hours", type=int, default=DAY_AHEAD_HOURS)
    p.add_argument(
        "--as-of",
        default=None,
        help="pretend the latest actual is at this UTC time (with --dry-run only)",
    )
    p.add_argument("--dry-run", action="store_true", help="compute but do not write")
    p.add_argument(
        "--once-per-day", action="store_true", help="skip if a forecast was already issued today"
    )
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)
    if args.as_of and not args.dry_run:
        p.error(AS_OF_NEEDS_DRY_RUN)  # before the job starts: nothing runs, nothing is reported

    with job_metrics("forecast") as metrics:
        result = run(
            hours=args.hours,
            as_of=pd.Timestamp(args.as_of) if args.as_of else None,
            dry_run=args.dry_run,
            once_per_day=args.once_per_day,
            database_url=args.database_url,
        )
        report_metrics(metrics, result, dry_run=args.dry_run)
    if result is None:
        print("Today's forecast was already issued - nothing to do.")
        return 0
    print(
        f"{result.model_name} v{result.model_version} | issued {result.issued_at:%Y-%m-%d %H:%M}Z | "
        f"last actual {result.last_actual} | target {result.target_start} -> {result.target_end} | "
        f"deadline margin {result.deadline_margin_hours:+.1f} h"
    )
    print(result.forecast.to_string(index=False))
    return 0


def report_metrics(metrics: JobMetrics, result: ForecastResult | None, *, dry_run: bool) -> None:
    """``forecast.issued`` only for a stored forecast: its absence is what alerting watches."""
    if result is None:
        metrics.gauge("forecast.skipped", 1)
        return
    if dry_run:
        return
    version = f"v{result.model_version}"
    metrics.gauge("forecast.issued", 1, model_version=version)
    metrics.gauge("forecast.hours", len(result.frame), model_version=version)
    data_age = (result.issued_at - result.last_actual) / pd.Timedelta(hours=1)
    metrics.gauge("forecast.data_age_hours", data_age, model_version=version)
    metrics.gauge(
        "forecast.deadline_margin_hours", result.deadline_margin_hours, model_version=version
    )


if __name__ == "__main__":
    sys.exit(main())
