"""Batch day-ahead forecast job: champion model -> ``load_forecast_model`` table.

Meant to run every morning after the incremental ingestion (the GitHub Actions
cron in ``forecast.yml``). Idempotent: re-running for the same hours and model
version updates the rows in place. The monitoring jobs compare this table against
the actuals and the official ENTSO-E forecast.

CLI (``make forecast``)::

    python -m src.serving.batch_forecast [--hours 24] [--as-of 2026-09-10T22:00Z] [--dry-run]
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
from src.serving.forecast import (
    DAY_AHEAD_HOURS,
    ForecastResult,
    make_day_ahead_forecast,
    store_forecast,
)

log = logging.getLogger(__name__)


def run(
    *,
    hours: int = DAY_AHEAD_HOURS,
    as_of: pd.Timestamp | None = None,
    dry_run: bool = False,
    database_url: str | None = None,
) -> ForecastResult:
    engine = db.get_engine(database_url)
    db.init_db(engine)
    model = load_champion(engine)
    result = make_day_ahead_forecast(
        engine, model, weather_client=WeatherClient(), hours=hours, as_of=as_of
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
    p.add_argument("--as-of", default=None, help="pretend the latest actual is at this UTC time")
    p.add_argument("--dry-run", action="store_true", help="compute but do not write")
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    result = run(
        hours=args.hours,
        as_of=pd.Timestamp(args.as_of) if args.as_of else None,
        dry_run=args.dry_run,
        database_url=args.database_url,
    )
    print(
        f"{result.model_name} v{result.model_version} | issued {result.issued_at:%Y-%m-%d %H:%M}Z | "
        f"last actual {result.last_actual} | target {result.target_start} -> {result.target_end}"
    )
    print(result.forecast.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
