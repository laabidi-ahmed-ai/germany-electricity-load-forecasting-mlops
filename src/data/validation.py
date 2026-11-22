"""Data-validation checks run on every ingested batch before it touches the DB.

Checks on hourly load frames (``[timestamp_utc, load_mw]``):

* not empty, and at least ``min_rows`` rows
* timestamps are tz-aware UTC, unique and sorted
* no NaN / non-positive load
* the *mean* load lies in a sane national range (default 20 000 - 90 000 MW)
* the share of individual hours outside that range stays below ``max_outlier_frac``
* time gaps > 1 hour are reported (as warnings: SMARD publishes with a lag and
  occasionally misses hours; gaps are handled downstream in feature engineering)

Gaps are measured on the UTC axis, so the 23-hour / 25-hour local DST days
are *not* gaps or duplicates - only genuinely missing hours are.

Per-city weather frames (``validate_weather``) are checked for the expected city set, the
share of missing values per variable, plausible ranges for every variable and, for
forecast rows, whether they reach far enough ahead.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass, field

import pandas as pd

from src.data.weather_client import GERMAN_CITIES, SOURCE_FORECAST

log = logging.getLogger(__name__)

# National German load is roughly 30 000 - 85 000 MW; these are generous bounds.
LOAD_MIN_MW = 20_000.0
LOAD_MAX_MW = 90_000.0
ONE_HOUR = pd.Timedelta(hours=1)

# Plausible hourly values in Germany, inclusive; anything outside is a unit or data error.
WEATHER_RANGES: dict[str, tuple[float, float]] = {
    "temperature_2m": (-40.0, 50.0),  # °C; German records -37.8 / +41.2
    "wind_speed_10m": (0.0, 200.0),  # km/h, hourly value at 10 m
    "shortwave_radiation": (0.0, 1200.0),  # W/m²; ~1361 at the top of the atmosphere
    "cloud_cover": (0.0, 100.0),  # %
    "relative_humidity_2m": (0.0, 100.0),  # %
}
# A dropped variable is 100 % missing, a missing city 20 % (of five); a few hours where
# one variable ends earlier than the others stay well below this.
WEATHER_MAX_NAN_SHARE = 0.10
CITY_NAMES: tuple[str, ...] = tuple(c.name for c in GERMAN_CITIES)


class DataValidationError(ValueError):
    """Raised when a batch fails a critical validation check."""


@dataclass
class Gap:
    start: pd.Timestamp  # last present hour before the gap
    end: pd.Timestamp  # first present hour after the gap
    missing_hours: int


@dataclass
class ValidationReport:
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    expected_hours: int = 0
    mean_load_mw: float | None = None
    min_load_mw: float | None = None
    max_load_mw: float | None = None
    outlier_rows: int = 0
    gaps: list[Gap] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def coverage(self) -> float:
        return self.rows / self.expected_hours if self.expected_hours else 0.0

    @property
    def missing_hours(self) -> int:
        return sum(g.missing_hours for g in self.gaps)

    def summary(self) -> str:
        lines = [
            f"rows={self.rows} range={self.start} -> {self.end} "
            f"coverage={self.coverage:.1%} ({self.missing_hours} missing hours in {len(self.gaps)} gaps)",
        ]
        if self.mean_load_mw is not None:
            lines.append(
                f"load_mw min/mean/max = {self.min_load_mw:,.0f} / "
                f"{self.mean_load_mw:,.0f} / {self.max_load_mw:,.0f}  outliers={self.outlier_rows}"
            )
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def find_gaps(timestamps: pd.Series, freq: pd.Timedelta = ONE_HOUR) -> list[Gap]:
    """Return the gaps in a sorted, unique series of tz-aware timestamps."""
    ts = pd.DatetimeIndex(timestamps).sort_values()
    if len(ts) < 2:
        return []
    deltas = ts[1:] - ts[:-1]
    gaps: list[Gap] = []
    for i in (deltas > freq).nonzero()[0]:
        missing = int(deltas[i] / freq) - 1
        gaps.append(Gap(start=ts[i], end=ts[i + 1], missing_hours=missing))
    return gaps


def validate_load(
    df: pd.DataFrame,
    *,
    min_rows: int = 1,
    max_outlier_frac: float = 0.01,
    max_gap_warning_hours: int = 24,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate an hourly load frame. Raises ``DataValidationError`` on critical issues."""
    report = ValidationReport(rows=len(df))

    if df.empty or len(df) < min_rows:
        report.errors.append(f"expected at least {min_rows} rows, got {len(df)}")
        return _finish(report, raise_on_error)

    missing_cols = {"timestamp_utc", "load_mw"} - set(df.columns)
    if missing_cols:
        report.errors.append(f"missing columns: {sorted(missing_cols)}")
        return _finish(report, raise_on_error)

    ts = df["timestamp_utc"]
    if not pd.api.types.is_datetime64_any_dtype(ts) or getattr(ts.dt, "tz", None) is None:
        report.errors.append("timestamp_utc must be tz-aware datetimes")
        return _finish(report, raise_on_error)
    if str(ts.dt.tz) != "UTC":
        report.errors.append(f"timestamp_utc must be UTC, got {ts.dt.tz}")
    if ts.duplicated().any():
        report.errors.append(f"{int(ts.duplicated().sum())} duplicate timestamps")
    if not ts.is_monotonic_increasing:
        report.errors.append("timestamps are not sorted ascending")

    load = pd.to_numeric(df["load_mw"], errors="coerce")
    n_nan = int(load.isna().sum())
    if n_nan:
        report.errors.append(f"{n_nan} NaN / non-numeric load values")
    n_nonpos = int((load <= 0).sum())
    if n_nonpos:
        report.errors.append(f"{n_nonpos} non-positive load values")

    report.start, report.end = ts.min(), ts.max()
    report.expected_hours = int((report.end - report.start) / pd.Timedelta(hours=1)) + 1
    report.mean_load_mw = float(load.mean())
    report.min_load_mw = float(load.min())
    report.max_load_mw = float(load.max())

    lo, hi = LOAD_MIN_MW, LOAD_MAX_MW
    if not (lo <= report.mean_load_mw <= hi):
        report.errors.append(
            f"mean load {report.mean_load_mw:,.0f} MW outside sane range [{lo:,.0f}, {hi:,.0f}]"
        )

    outliers = (load < lo) | (load > hi)
    report.outlier_rows = int(outliers.sum())
    frac = report.outlier_rows / len(df)
    if frac > max_outlier_frac:
        report.errors.append(
            f"{report.outlier_rows} rows ({frac:.2%}) outside [{lo:,.0f}, {hi:,.0f}] MW"
        )
    elif report.outlier_rows:
        report.warnings.append(f"{report.outlier_rows} rows outside [{lo:,.0f}, {hi:,.0f}] MW")

    report.gaps = find_gaps(ts.drop_duplicates())
    for gap in report.gaps:
        msg = f"gap of {gap.missing_hours}h between {gap.start} and {gap.end}"
        if gap.missing_hours > max_gap_warning_hours:
            report.warnings.append("LARGE " + msg)
        else:
            report.warnings.append(msg)

    return _finish(report, raise_on_error)


def _finish(report: ValidationReport, raise_on_error: bool) -> ValidationReport:
    if report.errors:
        log.error("load validation failed:\n%s", report.summary())
        if raise_on_error:
            raise DataValidationError("; ".join(report.errors))
    else:
        log.info("load validation ok:\n%s", report.summary())
    return report


@dataclass
class WeatherReport:
    rows: int = 0
    nan_share: dict[str, float] = field(default_factory=dict)  # per variable, on the grid
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_weather(
    df: pd.DataFrame,
    *,
    expected_cities: Collection[str] | None = None,
    forecast_until: pd.Timestamp | None = None,
    min_rows: int = 1,
    max_nan_share: float = WEATHER_MAX_NAN_SHARE,
    raise_on_error: bool = True,
) -> WeatherReport:
    """Validate a per-city weather frame (long format, ``CITY_COLUMNS``) before it is stored.

    Errors: no rows; timestamps not UTC; duplicate ``(timestamp_utc, city)``; city names
    that are not exactly ``expected_cities`` (default: ``GERMAN_CITIES``); a variable whose
    NaN share exceeds ``max_nan_share``; any value outside ``WEATHER_RANGES``; and, with
    ``forecast_until``, ``source=forecast`` rows that end before that hour (a stale or cut-off
    forecast). Any NaN at all is a warning.

    The NaN share is counted on the grid of every hour from the batch's own first to last
    hour times every expected city, a missing row counting as NaN in every variable. An
    answer that ends earlier than requested is therefore not "missing": lateness is judged
    by the archive's age (``src.data.ingest``), not here.
    """
    cities = set(expected_cities if expected_cities is not None else CITY_NAMES)
    report = WeatherReport(rows=len(df))
    if df.empty or len(df) < min_rows:
        report.errors.append(f"expected at least {min_rows} weather rows, got {len(df)}")
        return _finish_weather(report, raise_on_error)

    ts = df["timestamp_utc"]
    if getattr(ts.dt, "tz", None) is None or str(ts.dt.tz) != "UTC":
        report.errors.append("weather timestamp_utc must be tz-aware UTC")
        return _finish_weather(report, raise_on_error)
    dups = int(df.duplicated(subset=["timestamp_utc", "city"]).sum())
    if dups:
        report.errors.append(f"{dups} duplicate (timestamp_utc, city) rows")

    got = set(df["city"].unique())
    if got != cities:
        missing, unknown = sorted(cities - got), sorted(got - cities)
        report.errors.append(f"city set differs: missing {missing}, unknown {unknown}")

    hours = pd.date_range(ts.min(), ts.max(), freq="h", name="timestamp_utc")
    grid = pd.MultiIndex.from_product([hours, sorted(cities)], names=["timestamp_utc", "city"])
    known = df[df["city"].isin(cities)].drop_duplicates(subset=["timestamp_utc", "city"])
    known = known.set_index(["timestamp_utc", "city"]).reindex(grid)
    for var, (lo, hi) in WEATHER_RANGES.items():
        values = (
            pd.to_numeric(known[var], errors="coerce")
            if var in known
            else pd.Series(float("nan"), index=grid)
        )
        share = float(values.isna().mean())
        report.nan_share[var] = share
        if share > max_nan_share:
            report.errors.append(f"{var}: {share:.1%} missing (limit {max_nan_share:.0%})")
        elif share > 0:
            report.warnings.append(f"{var}: {share:.1%} missing")
        bad = int(((values < lo) | (values > hi)).sum())
        if bad:
            report.errors.append(f"{bad} {var} values outside [{lo:g}, {hi:g}]")

    if forecast_until is not None and "source" in df:
        forecast_ts = ts[df["source"] == SOURCE_FORECAST]
        last = forecast_ts.max() if not forecast_ts.empty else None
        if last is None or last < forecast_until:
            report.errors.append(
                f"forecast rows end at {last}, before {forecast_until} (stale or cut off)"
            )
    return _finish_weather(report, raise_on_error)


def _finish_weather(report: WeatherReport, raise_on_error: bool) -> WeatherReport:
    for warning in report.warnings:
        log.warning("weather validation: %s", warning)
    if report.errors:
        log.error("weather validation failed: %s", "; ".join(report.errors))
        if raise_on_error:
            raise DataValidationError("; ".join(report.errors))
    else:
        log.info("weather validation ok: %d rows", report.rows)
    return report
