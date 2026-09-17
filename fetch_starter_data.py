#!/usr/bin/env python3
"""Build a flat starter CSV (German load + national-average weather), no API token needed.

This is a thin convenience wrapper: since Phase 1 the real logic lives in
``src/data/`` (``smard_client``, ``weather_client``, ``validation``). The
canonical ingestion path is the database pipeline::

    python -m src.data.ingest --backfill        # or: make data

Use this script only when you want a single CSV for quick exploration
(notebooks). It fetches straight from SMARD and Open-Meteo and never touches the
database.

Output: ``data/raw/germany_load_weather_2021-03_onward.csv``
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from config.settings import get_settings
from src.data.smard_client import SmardClient
from src.data.validation import validate_load
from src.data.weather_client import WeatherClient, national_average

OUTPUT_PATH = Path("data/raw/germany_load_weather_2021-03_onward.csv")

log = logging.getLogger("starter-data")


def build_starter_dataset() -> pd.DataFrame:
    settings = get_settings()
    start = settings.data_start_date

    load_df = SmardClient().fetch_load(start)
    validate_load(load_df, min_rows=24 * 30)

    try:
        weather = national_average(WeatherClient().fetch_historical(start))
        df = load_df.merge(weather, on="timestamp_utc", how="left")
    except Exception as err:  # keep the load-only CSV usable if weather fails
        log.warning("weather fetch failed (%s) - saving load-only dataset", err)
        df = load_df

    # Simple calendar columns for a first look (local time, as demand follows it).
    local = df["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    df["hour"] = local.dt.hour
    df["dayofweek"] = local.dt.dayofweek  # 0 = Monday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = local.dt.month
    return df


def main() -> int:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    df = build_starter_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("saved -> %s  (%d rows, %d columns)", OUTPUT_PATH, len(df), df.shape[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
