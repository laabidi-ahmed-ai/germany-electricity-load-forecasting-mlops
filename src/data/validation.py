"""Data-validation checks run on every ingested batch before it touches the DB.

Checks on hourly load frames (``[timestamp_utc, load_mw]``):

* not empty, and at least ``min_rows`` rows
* timestamps are tz-aware UTC, unique and sorted
* no NaN / non-positive load
* the *mean* load lies in a sane national range (default 20 000 - 90 000 MW)
* the share of individual hours outside that range stays below ``max_outlier_frac``
* time gaps > 1 hour are reported (as warnings: SMARD publishes with a lag and
  occasionally misses hours; gaps are handled downstream in feature engineering)

Gaps are measured on the **UTC** axis, so the 23-hour / 25-hour local DST days
are *not* gaps or duplicates - only genuinely missing hours are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

# National German load is roughly 30 000 - 85 000 MW; these are generous bounds.
LOAD_MIN_MW = 20_000.0
LOAD_MAX_MW = 90_000.0
ONE_HOUR = pd.Timedelta(hours=1)


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
    mean_range: tuple[float, float] = (LOAD_MIN_MW, LOAD_MAX_MW),
    row_range: tuple[float, float] = (LOAD_MIN_MW, LOAD_MAX_MW),
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

    lo, hi = mean_range
    if not (lo <= report.mean_load_mw <= hi):
        report.errors.append(
            f"mean load {report.mean_load_mw:,.0f} MW outside sane range [{lo:,.0f}, {hi:,.0f}]"
        )

    rlo, rhi = row_range
    outliers = (load < rlo) | (load > rhi)
    report.outlier_rows = int(outliers.sum())
    frac = report.outlier_rows / len(df)
    if frac > max_outlier_frac:
        report.errors.append(
            f"{report.outlier_rows} rows ({frac:.2%}) outside [{rlo:,.0f}, {rhi:,.0f}] MW"
        )
    elif report.outlier_rows:
        report.warnings.append(f"{report.outlier_rows} rows outside [{rlo:,.0f}, {rhi:,.0f}] MW")

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


def validate_weather(df: pd.DataFrame, *, min_rows: int = 1, raise_on_error: bool = True) -> None:
    """Light checks on a per-city weather frame (non-empty, UTC, unique keys, sane temps)."""
    errors: list[str] = []
    if df.empty or len(df) < min_rows:
        errors.append(f"expected at least {min_rows} weather rows, got {len(df)}")
    else:
        ts = df["timestamp_utc"]
        if getattr(ts.dt, "tz", None) is None or str(ts.dt.tz) != "UTC":
            errors.append("weather timestamp_utc must be tz-aware UTC")
        dups = int(df.duplicated(subset=["timestamp_utc", "city"]).sum())
        if dups:
            errors.append(f"{dups} duplicate (timestamp_utc, city) rows")
        if "temperature_2m" in df:
            temp = pd.to_numeric(df["temperature_2m"], errors="coerce")
            bad = int(((temp < -40) | (temp > 50)).sum())
            if bad:
                errors.append(f"{bad} implausible temperature_2m values")
    if errors:
        log.error("weather validation failed: %s", "; ".join(errors))
        if raise_on_error:
            raise DataValidationError("; ".join(errors))
    else:
        log.info("weather validation ok: %d rows", len(df))
