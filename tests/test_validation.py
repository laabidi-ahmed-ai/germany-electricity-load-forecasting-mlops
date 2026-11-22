"""Data-validation tests: row counts, gaps (UTC / DST-safe), duplicates, sane load range,
and the weather checks (city set, missing values, ranges, forecast freshness)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import (
    CITY_NAMES,
    WEATHER_RANGES,
    DataValidationError,
    find_gaps,
    validate_load,
    validate_weather,
)
from src.data.weather_client import GERMAN_CITIES, WEATHER_VARS


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


def weather_frame(hours: int = 24, *, source: str = "archive", start: str = "2021-03-01"):
    """A valid long-format frame: every German city, every variable, mid-range values."""
    ts = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    rows = pd.MultiIndex.from_product([ts, CITY_NAMES], names=["timestamp_utc", "city"])
    df = rows.to_frame(index=False).assign(source=source)
    for var, (lo, hi) in WEATHER_RANGES.items():
        df[var] = (lo + hi) / 2
    return df


def test_weather_ranges_cover_every_variable() -> None:
    assert tuple(WEATHER_RANGES) == WEATHER_VARS
    assert tuple(c.name for c in GERMAN_CITIES) == CITY_NAMES


def test_validate_weather_ok_and_basic_failures() -> None:
    ok = weather_frame()
    report = validate_weather(ok)
    assert report.ok and report.warnings == []
    assert report.nan_share == dict.fromkeys(WEATHER_VARS, 0.0)

    with pytest.raises(DataValidationError, match="at least 1 weather rows"):
        validate_weather(ok.iloc[0:0])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_weather(pd.concat([ok, ok.tail(1)], ignore_index=True))
    naive = ok.assign(timestamp_utc=ok["timestamp_utc"].dt.tz_localize(None))
    with pytest.raises(DataValidationError, match="tz-aware UTC"):
        validate_weather(naive)


def test_validate_weather_checks_city_identities_not_counts() -> None:
    df = weather_frame()
    renamed = df.assign(city=df["city"].replace({"Munich": "München"}))  # still 5 cities
    with pytest.raises(DataValidationError, match=r"missing \['Munich'\], unknown \['München'\]"):
        validate_weather(renamed)
    with pytest.raises(DataValidationError, match=r"missing \['Berlin'\]"):
        validate_weather(df[df["city"] != "Berlin"])
    # the client's own cities are what is expected
    two = df[df["city"].isin(["Berlin", "Hamburg"])]
    assert validate_weather(two, expected_cities=["Berlin", "Hamburg"]).ok


def test_nan_share_is_an_error_above_ten_percent_and_a_warning_below() -> None:
    df = weather_frame(hours=100)  # 500 (hour, city) cells
    few = df.copy()
    few.loc[few.index[:25], "cloud_cover"] = float("nan")  # 5 %
    report = validate_weather(few)
    assert report.ok and report.nan_share["cloud_cover"] == pytest.approx(0.05)
    assert any("cloud_cover: 5.0% missing" in w for w in report.warnings)

    many = df.copy()
    many.loc[many.index[:60], "cloud_cover"] = float("nan")  # 12 %
    with pytest.raises(DataValidationError, match=r"cloud_cover: 12.0% missing \(limit 10%\)"):
        validate_weather(many)

    with pytest.raises(DataValidationError, match=r"wind_speed_10m: 100.0% missing"):
        validate_weather(df.drop(columns="wind_speed_10m"))  # a variable the API dropped


def test_missing_rows_count_as_missing_on_the_batch_grid() -> None:
    df = weather_frame(hours=10)
    # Berlin lacks 2 of the 10 hours in the middle: 2 of 50 cells = 4 % in every variable
    hole = (df["city"] == "Berlin") & df["timestamp_utc"].isin(df["timestamp_utc"].unique()[4:6])
    report = validate_weather(df[~hole])
    assert report.ok and report.nan_share == pytest.approx(dict.fromkeys(WEATHER_VARS, 0.04))


def test_a_shorter_answer_is_not_counted_as_missing() -> None:
    """The grid spans the batch's own first..last hour: an archive answer ending early is
    fine here (lateness is judged by the archive's age in the ingest)."""
    report = validate_weather(weather_frame(hours=5))
    assert report.ok and report.warnings == []


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("temperature_2m", 51.0),
        ("temperature_2m", -41.0),
        ("wind_speed_10m", -1.0),
        ("wind_speed_10m", 201.0),
        ("shortwave_radiation", -0.5),
        ("shortwave_radiation", 1300.0),
        ("cloud_cover", 101.0),
        ("relative_humidity_2m", 100.5),
    ],
)
def test_values_outside_the_plausible_range_fail(var: str, value: float) -> None:
    df = weather_frame()
    df.loc[3, var] = value
    with pytest.raises(DataValidationError, match=f"1 {var} values outside"):
        validate_weather(df)


def test_range_bounds_are_inclusive() -> None:
    df = weather_frame()
    for var, (lo, hi) in WEATHER_RANGES.items():
        df.loc[0, var], df.loc[1, var] = lo, hi
    assert validate_weather(df).ok


def test_forecast_rows_must_reach_far_enough() -> None:
    df = weather_frame(hours=72, source="forecast")
    last = df["timestamp_utc"].max()
    assert validate_weather(df, forecast_until=last).ok
    with pytest.raises(DataValidationError, match="stale or cut off"):
        validate_weather(df, forecast_until=last + pd.Timedelta(hours=1))
    # archive rows do not count as forecast coverage
    archive = df.assign(source="archive")
    with pytest.raises(DataValidationError, match="forecast rows end at None"):
        validate_weather(archive, forecast_until=last)
    assert validate_weather(archive).ok  # no forecast requirement for an archive batch
