"""Data-validation tests: row counts, gaps (UTC / DST-safe), duplicates, sane load range."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import (
    DataValidationError,
    find_gaps,
    validate_load,
    validate_weather,
)


def test_clean_frame_passes(load_frame) -> None:
    report = validate_load(load_frame)
    assert report.ok
    assert report.rows == 335
    assert report.expected_hours == 335
    assert report.coverage == pytest.approx(1.0)
    assert report.gaps == []
    assert 50_000 < report.mean_load_mw < 60_000
    assert "coverage=100.0%" in report.summary()


def test_empty_frame_fails() -> None:
    with pytest.raises(DataValidationError, match="at least 1 rows"):
        validate_load(pd.DataFrame(columns=["timestamp_utc", "load_mw"]))


def test_min_rows(load_frame) -> None:
    with pytest.raises(DataValidationError, match="at least 1000 rows"):
        validate_load(load_frame, min_rows=1000)


def test_duplicate_timestamps_fail(load_frame) -> None:
    dup = pd.concat([load_frame, load_frame.tail(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_load(dup)


def test_unsorted_fails(load_frame) -> None:
    with pytest.raises(DataValidationError, match="not sorted"):
        validate_load(load_frame.iloc[::-1].reset_index(drop=True))


def test_naive_timestamps_fail(load_frame) -> None:
    naive = load_frame.assign(timestamp_utc=load_frame["timestamp_utc"].dt.tz_localize(None))
    with pytest.raises(DataValidationError, match="tz-aware"):
        validate_load(naive)


def test_non_utc_timestamps_fail(load_frame) -> None:
    berlin = load_frame.assign(
        timestamp_utc=load_frame["timestamp_utc"].dt.tz_convert("Europe/Berlin")
    )
    with pytest.raises(DataValidationError, match="must be UTC"):
        validate_load(berlin)


def test_mean_outside_sane_range_fails(load_frame) -> None:
    with pytest.raises(DataValidationError, match="outside sane range"):
        validate_load(load_frame.assign(load_mw=load_frame["load_mw"] / 10))  # ~5.5 GW
    with pytest.raises(DataValidationError, match="outside sane range"):
        validate_load(load_frame.assign(load_mw=load_frame["load_mw"] * 2))  # ~110 GW


def test_few_outlier_rows_only_warn(load_frame) -> None:
    df = load_frame.copy()
    df.loc[10, "load_mw"] = 95_000  # a single spike (0.3% of rows)
    report = validate_load(df)
    assert report.ok
    assert report.outlier_rows == 1
    assert any("outside" in w for w in report.warnings)


def test_many_outlier_rows_fail(load_frame) -> None:
    df = load_frame.copy()
    df.loc[:10, "load_mw"] = 95_000  # 11 rows = 3.3%
    with pytest.raises(DataValidationError, match="outside \\["):
        validate_load(df)


def test_nan_and_nonpositive_fail(load_frame) -> None:
    df = load_frame.copy()
    df.loc[0, "load_mw"] = float("nan")
    df.loc[1, "load_mw"] = 0.0
    report = validate_load(df, raise_on_error=False)
    assert not report.ok
    assert any("NaN" in e for e in report.errors)
    assert any("non-positive" in e for e in report.errors)


def test_gaps_are_reported_as_warnings_not_errors(load_frame) -> None:
    df = load_frame.drop(index=range(100, 103)).reset_index(drop=True)  # 3 missing hours
    report = validate_load(df)
    assert report.ok
    assert len(report.gaps) == 1
    assert report.gaps[0].missing_hours == 3
    assert report.missing_hours == 3
    assert report.rows == 332 and report.expected_hours == 335
    assert report.gaps[0].start == load_frame["timestamp_utc"].iloc[99]
    assert report.gaps[0].end == load_frame["timestamp_utc"].iloc[103]
    assert any("gap of 3h" in w for w in report.warnings)


def test_large_gap_is_flagged(load_frame) -> None:
    df = load_frame.drop(index=range(50, 100)).reset_index(drop=True)  # 50 missing hours
    report = validate_load(df, max_gap_warning_hours=24)
    assert any(w.startswith("LARGE gap of 50h") for w in report.warnings)


def test_dst_day_is_not_a_gap() -> None:
    """A full Berlin-local DST day converted to UTC has uniform steps -> no gaps."""
    local = pd.date_range(
        "2021-03-28", "2021-03-29", freq="h", tz="Europe/Berlin", inclusive="left"
    )
    assert len(local) == 23  # spring-forward day
    ts = pd.Series(local.tz_convert("UTC"))
    assert find_gaps(ts) == []

    local_fall = pd.date_range(
        "2021-10-31", "2021-11-01", freq="h", tz="Europe/Berlin", inclusive="left"
    )
    assert len(local_fall) == 25  # fall-back day
    ts_fall = pd.Series(local_fall.tz_convert("UTC"))
    assert find_gaps(ts_fall) == []
    assert not ts_fall.duplicated().any()


def test_find_gaps_handles_short_inputs() -> None:
    assert find_gaps(pd.Series(pd.to_datetime([], utc=True))) == []
    assert find_gaps(pd.Series(pd.to_datetime(["2021-01-01T00:00Z"]))) == []


def test_validate_weather_ok_and_failures() -> None:
    ts = pd.date_range("2021-03-01", periods=3, freq="h", tz="UTC")
    ok = pd.DataFrame({"timestamp_utc": ts, "city": "Berlin", "temperature_2m": [1.0, 2.0, 3.0]})
    validate_weather(ok)

    with pytest.raises(DataValidationError, match="at least 1 weather rows"):
        validate_weather(ok.iloc[0:0])

    dup = pd.concat([ok, ok.tail(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_weather(dup)

    hot = ok.assign(temperature_2m=[1.0, 2.0, 99.0])
    with pytest.raises(DataValidationError, match="implausible temperature"):
        validate_weather(hot)

    naive = ok.assign(timestamp_utc=ts.tz_localize(None))
    with pytest.raises(DataValidationError, match="tz-aware UTC"):
        validate_weather(naive)
