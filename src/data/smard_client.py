"""SMARD (Bundesnetzagentur) client - actual German electricity load, no token needed.

SMARD publishes hourly data as one JSON file per week::

    {base}/{filter}/{region}/index_{resolution}.json
        -> {"timestamps": [<epoch ms of each available week start>, ...]}
    {base}/{filter}/{region}/{filter}_{region}_{resolution}_{week_start_ms}.json
        -> {"meta_data": ..., "series": [[<epoch ms>, <value|null>], ...]}

Timestamps are epoch milliseconds, i.e. **UTC and DST-unambiguous**. Week files
start at Monday 00:00 *local* time, so their UTC start drifts between 22:00 and
23:00 across DST; we only ever filter by epoch and never reason in local time.

Values for filter 410 ("Realisierter Stromverbrauch > Gesamt (Netzlast)") are
hourly energy in MWh, numerically equal to the average power in MW for that hour.
Filter 411 ("Prognostizierter Stromverbrauch > Gesamt (Netzlast)") is the TSOs'
day-ahead load forecast for the same quantity - the official benchmark, published
tokenless by SMARD (it is the same series ENTSO-E exposes as "Forecasted Load").
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
import requests

from config.settings import get_settings
from src.data._http import get_json, make_session

log = logging.getLogger(__name__)

# SMARD "filter" ids (chart series).
FILTER_ACTUAL_LOAD = 410  # Realisierter Stromverbrauch: Gesamt (Netzlast)
FILTER_FORECAST_LOAD = 411  # Prognostizierter Stromverbrauch: Gesamt (Netzlast) - day-ahead
RESOLUTION_HOUR = "hour"

LOAD_COLUMNS = ["timestamp_utc", "load_mw"]
FORECAST_COLUMNS = ["timestamp_utc", "forecast_mw"]

_MS_PER_HOUR = 3_600_000
_WEEK_MS = 7 * 24 * _MS_PER_HOUR

DateLike = date | datetime | str | pd.Timestamp


def to_utc_timestamp(value: DateLike) -> pd.Timestamp:
    """Coerce a date-like value to a tz-aware UTC ``pd.Timestamp``.

    Naive inputs are *assumed* to be UTC (never local time).
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def empty_load_frame(value_col: str = "load_mw") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            value_col: pd.Series(dtype="float64"),
        }
    )


class SmardClient:
    """Fetch hourly actual load for a SMARD region (default: ``DE``)."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        base_url: str | None = None,
        region: str | None = None,
    ) -> None:
        settings = get_settings()
        self.session = session or make_session()
        self.base_url = (base_url or settings.smard_base_url).rstrip("/")
        self.region = region or settings.smard_region

    # ------------------------------------------------------------------ URLs
    def index_url(self, filter_id: int = FILTER_ACTUAL_LOAD) -> str:
        return f"{self.base_url}/{filter_id}/{self.region}/index_{RESOLUTION_HOUR}.json"

    def week_url(self, week_start_ms: int, filter_id: int = FILTER_ACTUAL_LOAD) -> str:
        return (
            f"{self.base_url}/{filter_id}/{self.region}/"
            f"{filter_id}_{self.region}_{RESOLUTION_HOUR}_{week_start_ms}.json"
        )

    # --------------------------------------------------------------- fetching
    def available_weeks(self, filter_id: int = FILTER_ACTUAL_LOAD) -> list[int]:
        """Return the sorted epoch-ms start of every weekly file SMARD offers."""
        payload = get_json(self.session, self.index_url(filter_id))
        return sorted(int(t) for t in payload["timestamps"])

    def fetch_load(self, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Return hourly actual load in ``[start, end]`` as ``[timestamp_utc, load_mw]``.

        ``end`` defaults to "now". Null / non-positive values (hours SMARD has not
        published yet) are dropped. The result is sorted, de-duplicated and UTC.
        """
        return self._fetch_series(FILTER_ACTUAL_LOAD, "load_mw", start, end)

    def fetch_forecast(self, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Return the official day-ahead load forecast as ``[timestamp_utc, forecast_mw]``.

        ``end`` defaults to "now + 2 days": the forecast for tomorrow is already
        published today, and that is what monitoring compares the model against.
        """
        if end is None:
            end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
        return self._fetch_series(FILTER_FORECAST_LOAD, "forecast_mw", start, end)

    def _fetch_series(
        self, filter_id: int, value_col: str, start: DateLike, end: DateLike | None
    ) -> pd.DataFrame:
        start_ts = to_utc_timestamp(start)
        end_ts = to_utc_timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
        if end_ts < start_ts:
            raise ValueError(f"end ({end_ts}) is before start ({start_ts})")

        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        weeks = self.available_weeks(filter_id)
        # A week file covers [week_start, week_start + 7d); keep every file that
        # overlaps the requested window.
        wanted = [w for w in weeks if w + _WEEK_MS > start_ms and w <= end_ms]
        log.info(
            "SMARD %s filter %d: %d weekly files cover %s -> %s (of %d available)",
            self.region,
            filter_id,
            len(wanted),
            start_ts,
            end_ts,
            len(weeks),
        )

        rows: list[tuple[int, float]] = []
        for i, week_start in enumerate(wanted, 1):
            payload = get_json(self.session, self.week_url(week_start, filter_id))
            for point_ts, value in payload.get("series") or []:
                if value is not None:
                    rows.append((int(point_ts), float(value)))
            if i % 25 == 0 or i == len(wanted):
                log.info("  downloaded %d/%d weekly files", i, len(wanted))

        if not rows:
            return empty_load_frame(value_col)

        df = pd.DataFrame(rows, columns=["timestamp_ms", value_col])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = (
            df.loc[:, ["timestamp_utc", value_col]]
            .drop_duplicates(subset="timestamp_utc", keep="last")
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )
        df = df[(df["timestamp_utc"] >= start_ts) & (df["timestamp_utc"] <= end_ts)]
        df = df[df[value_col] > 0].reset_index(drop=True)

        if not df.empty:
            log.info(
                "SMARD filter %d: %d hourly rows, %s -> %s",
                filter_id,
                len(df),
                df["timestamp_utc"].iloc[0],
                df["timestamp_utc"].iloc[-1],
            )
        return df
