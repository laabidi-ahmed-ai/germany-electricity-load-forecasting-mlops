"""ENTSO-E Transparency Platform client - actual load + the official day-ahead forecast.

Uses ``entsoe-py`` and requires ``ENTSOE_API_TOKEN``. **Without a token the client
never raises**: every fetch logs a notice and returns an empty, correctly-typed
frame, so the whole pipeline keeps running on SMARD until the token arrives.

Time handling
-------------
``entsoe-py`` returns a tz-aware index in the bidding zone's local time at 15-minute
resolution for ``DE_LU``. We convert to **UTC first** and only then resample to
hourly means, so the DST transitions (23h / 25h local days) are handled by the
tz-aware arithmetic and never by hand.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, Protocol

import pandas as pd

from config.settings import get_settings

log = logging.getLogger(__name__)

DateLike = date | datetime | str | pd.Timestamp

# Column names used by entsoe-py for the two load series.
_ACTUAL_COL = "Actual Load"
_FORECAST_COL = "Forecasted Load"

LOAD_COLUMNS = ["timestamp_utc", "load_mw"]
FORECAST_COLUMNS = ["timestamp_utc", "forecast_mw"]


class _PandasClient(Protocol):
    """The slice of ``entsoe.EntsoePandasClient`` we rely on (for typing + fakes)."""

    def query_load(self, country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> Any: ...

    def query_load_forecast(
        self, country_code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> Any: ...


def _default_client_factory(token: str) -> _PandasClient:
    # Imported lazily so the (fairly heavy) dependency is only loaded when used.
    from entsoe import EntsoePandasClient

    return EntsoePandasClient(api_key=token)


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            columns[0]: pd.Series(dtype="datetime64[ns, UTC]"),
            columns[1]: pd.Series(dtype="float64"),
        }
    )


def _to_utc(value: DateLike) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_hourly_utc(raw: Any, column: str, value_name: str) -> pd.DataFrame:
    """Normalise an entsoe-py result (Series or DataFrame) to hourly UTC rows."""
    if raw is None or len(raw) == 0:
        return _empty(["timestamp_utc", value_name])

    series = raw[column] if isinstance(raw, pd.DataFrame) else raw
    series = pd.Series(series, copy=True).astype("float64")
    series.index = pd.DatetimeIndex(series.index)
    if series.index.tz is None:  # pragma: no cover - entsoe-py always returns tz-aware
        raise ValueError("ENTSO-E result index must be tz-aware")
    series = series.tz_convert("UTC").sort_index()

    hourly = series.resample("1h", label="left", closed="left").mean().dropna()
    df = hourly.rename(value_name).rename_axis("timestamp_utc").reset_index()
    return df.loc[:, ["timestamp_utc", value_name]]


class EntsoeClient:
    """Fetch DE_LU actual load and official day-ahead load forecast (hourly, UTC)."""

    def __init__(
        self,
        token: str | None = None,
        *,
        bidding_zone: str | None = None,
        client_factory: Callable[[str], _PandasClient] = _default_client_factory,
    ) -> None:
        settings = get_settings()
        if token is None and settings.entsoe_api_token is not None:
            token = settings.entsoe_api_token.get_secret_value()
        self._token = token or None
        self.bidding_zone = bidding_zone or settings.bidding_zone
        self._client_factory = client_factory
        self._client: _PandasClient | None = None

    @property
    def available(self) -> bool:
        """True when a token is configured and live ENTSO-E queries can be made."""
        return self._token is not None

    def _get_client(self) -> _PandasClient:
        if self._client is None:
            assert self._token is not None
            self._client = self._client_factory(self._token)
        return self._client

    def _query(
        self,
        method: str,
        column: str,
        value_name: str,
        start: DateLike,
        end: DateLike | None,
    ) -> pd.DataFrame:
        if not self.available:
            log.info(
                "ENTSO-E %s skipped: no ENTSOE_API_TOKEN configured (running on SMARD only)",
                method,
            )
            return _empty(["timestamp_utc", value_name])

        start_ts = _to_utc(start)
        end_ts = _to_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
        if end_ts <= start_ts:
            raise ValueError(f"end ({end_ts}) must be after start ({start_ts})")

        from entsoe.exceptions import NoMatchingDataError

        log.info("ENTSO-E %s %s: %s -> %s", method, self.bidding_zone, start_ts, end_ts)
        try:
            raw = getattr(self._get_client(), method)(self.bidding_zone, start=start_ts, end=end_ts)
        except NoMatchingDataError:
            log.warning("ENTSO-E %s: no data for %s -> %s", method, start_ts, end_ts)
            return _empty(["timestamp_utc", value_name])

        df = _to_hourly_utc(raw, column, value_name)
        log.info("ENTSO-E %s: %d hourly rows", method, len(df))
        return df

    def fetch_actual_load(self, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Hourly actual total load as ``[timestamp_utc, load_mw]`` (empty if no token)."""
        return self._query("query_load", _ACTUAL_COL, "load_mw", start, end)

    def fetch_dayahead_forecast(self, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """Hourly official day-ahead load forecast as ``[timestamp_utc, forecast_mw]``."""
        return self._query("query_load_forecast", _FORECAST_COL, "forecast_mw", start, end)
