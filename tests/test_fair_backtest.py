"""Fair backtest: issue times, which archived forecast each hour reads, no leakage, the block
bootstrap and the CLI - all offline (fake Open-Meteo session, synthetic SQLite database)."""

from __future__ import annotations

import functools
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from sqlalchemy import text

from src.data import _http, db
from src.data import weather_client as wc
from src.data.weather_client import GERMAN_CITIES, WeatherClient
from src.features.build_features import TARGET, WEATHER_COLUMNS, build_feature_frame
from src.features.horizons import D_PLUS_1, DAY_AHEAD, most_recent_load_lag, select_features
from src.models import train as train_mod
from src.monitoring import fair_backtest as fb
from src.monitoring import performance as perf
from tests.conftest import (
    FAST_LGBM,
    FakeResponse,
    FakeSession,
    load_curve,
    previous_runs_route,
    random_weather,
)

PREVIOUS_RUNS = "https://previous-runs.test/v1/forecast"
START = pd.Timestamp("2024-01-01T00:00Z")
N_DAYS = 400  # > 1 year of training rows + a 14-day holdout
HOLDOUT_DAYS = 14
FAST_BOOT = 200
# What the fake API returns, averaged over the five cities: 100 x offset + mean latitude / 100.
MEAN_LAT = float(np.mean([c.latitude for c in GERMAN_CITIES])) / 100


# --- Helpers ---
def berlin_day(day: str) -> pd.DatetimeIndex:
    """The UTC hours of Berlin calendar ``day`` (23, 24 or 25 of them)."""
    start = pd.Timestamp(day).tz_localize("Europe/Berlin")
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("Europe/Berlin")
    return pd.date_range(
        start.tz_convert("UTC"),
        end.tz_convert("UTC"),
        freq="h",
        inclusive="left",
        name="timestamp_utc",
    )


def seed(engine) -> pd.DataFrame:
    """Actuals + official forecast in the DB; returns the feature frame (archive weather)."""
    ts = pd.date_range(START, periods=24 * N_DAYS, freq="h", tz="UTC")
    load = load_curve(len(ts))
    rng = np.random.default_rng(1)
    db.upsert_dataframe(
        engine, db.LoadActual, pd.DataFrame({"timestamp_utc": ts, TARGET: load, "source": "smard"})
    )
    db.upsert_dataframe(
        engine,
        db.LoadForecastOfficial,
        pd.DataFrame(
            {
                "timestamp_utc": ts,
                "forecast_mw": load + rng.normal(0, 1_500, len(ts)),
                "source": "smard",
            }
        ),
    )
    load_df = pd.DataFrame({"timestamp_utc": ts, TARGET: load})
    return build_feature_frame(load_df, random_weather(load_df["timestamp_utc"], seed=2))


def fake_client(route=previous_runs_route) -> WeatherClient:
    session = FakeSession(routes={PREVIOUS_RUNS: route})
    return WeatherClient(session=session, previous_runs_url=PREVIOUS_RUNS)


def run(engine, frame, **kwargs) -> fb.FairBacktestReport:
    kwargs.setdefault("weather_client", fake_client())
    return fb.fair_backtest_vs_official(
        engine, days=HOLDOUT_DAYS, frame=frame, params=FAST_LGBM, n_boot=FAST_BOOT, **kwargs
    )


# --- Issue times and offsets ---
@pytest.mark.parametrize(
    ("day", "n_hours", "first_utc"),
    [
        ("2026-01-15", 24, "2026-01-14T23:00Z"),  # winter
        ("2026-07-15", 24, "2026-07-14T22:00Z"),  # summer
        ("2026-03-29", 23, "2026-03-28T23:00Z"),  # spring clock change
        ("2025-10-26", 25, "2025-10-25T22:00Z"),  # autumn clock change
    ],
)
def test_issue_is_0540_utc_on_the_day_before_the_berlin_target_day(day, n_hours, first_utc) -> None:
    hours = berlin_day(day)
    assert len(hours) == n_hours and hours[0] == pd.Timestamp(first_utc)
    issue = fb.issue_times(hours)
    day_before = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    assert set(issue) == {pd.Timestamp(f"{day_before}T05:40Z")}


def test_newest_reads_day1_only_where_its_run_was_out_by_the_issue() -> None:
    winter = fb.choose_offsets(berlin_day("2026-01-15"), fb.NEWEST)
    assert list(winter.index[winter["offset_days"] == 1]) == [pd.Timestamp("2026-01-14T23:00Z")]
    summer = fb.choose_offsets(berlin_day("2026-07-15"), fb.NEWEST)
    assert list(summer.index[summer["offset_days"] == 1]) == list(
        pd.to_datetime(["2026-07-14T22:00Z", "2026-07-14T23:00Z"])
    )
    for day in ("2026-01-15", "2026-07-15", "2025-10-26"):
        assert set(fb.choose_offsets(berlin_day(day), fb.DAY2)["offset_days"]) == {2}


def test_offsets_carry_when_their_run_was_out() -> None:
    hours = berlin_day("2026-01-15")
    out = fb.choose_offsets(hours, fb.DAY2)
    assert (out["run_init_bound"] == hours - pd.Timedelta(hours=48)).all()
    assert (out["available_by"] == out["run_init_bound"] + fb.PUBLICATION_LATENCY).all()
    assert (out["available_by"] <= out["issue_time"]).all()


def test_a_forecast_published_after_the_issue_is_refused() -> None:
    late = pd.Timedelta(hours=8)  # day2 of the last hour would be out at 06:00, after 05:40
    for mode in fb.MODES:
        with pytest.raises(ValueError, match="published after their issue"):
            fb.choose_offsets(berlin_day("2026-01-15"), mode, latency=late)
    with pytest.raises(ValueError, match="mode"):
        fb.choose_offsets(berlin_day("2026-01-15"), "day0")


def test_blocks_are_whole_weeks_of_berlin_target_days() -> None:
    hours = pd.DatetimeIndex(
        np.concatenate([berlin_day(d) for d in pd.date_range("2026-03-20", "2026-04-12").date])
    )  # 24 days, including the 23-hour 29 March
    blocks = fb.target_day_blocks(hours)
    days = fb.target_days(hours)
    per_day = pd.Series(blocks).groupby(days).nunique()
    assert (per_day == 1).all()  # a target day is never split
    days_per_block = pd.Series(days).groupby(blocks).nunique()
    assert list(days_per_block) == [7, 7, 7, 3]
    assert (np.diff(blocks) >= 0).all()


# --- Block bootstrap: known inputs -> known interval ---
def blocks_with_gains(*gains: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One 24-hour block per gain: official error 100 MW, model error 100 - gain (so gain %)."""
    official = np.full(24 * len(gains), 100.0)
    model = np.repeat(100.0 - np.asarray(gains, dtype="float64"), 24)
    return model, official, np.repeat(np.arange(len(gains)), 24)


def test_two_blocks_give_the_interval_between_their_gains() -> None:
    # Resamples: AA (p = 1/4) -> 10 %, AB / BA -> 20 %, BB (1/4) -> 30 %.
    ci = fb.block_bootstrap_ci(*blocks_with_gains(10, 30), n_boot=2_000, seed=3)
    assert ci.improvement_pct == pytest.approx(20.0)
    assert (ci.low, ci.high) == (pytest.approx(10.0), pytest.approx(30.0))
    assert ci.diff_mw == pytest.approx(20.0)
    assert (ci.diff_low, ci.diff_high) == (pytest.approx(10.0), pytest.approx(30.0))
    assert ci.n_blocks == 2


@pytest.mark.parametrize("gains", [(30,), (10, 30)], ids=["k=1", "k=2"])
def test_too_few_blocks_give_no_verdict(gains, caplog) -> None:
    # Every block is a clear gain, so the interval lies above 0 - but it rests on 1 or 2 blocks.
    with caplog.at_level(logging.WARNING, logger=fb.__name__):
        ci = fb.block_bootstrap_ci(*blocks_with_gains(*gains), n_boot=500)
    assert ci.n_blocks == len(gains) and ci.low > 0
    assert not ci.reliable and ci.verdict == fb.TOO_FEW_BLOCKS
    assert "clearly" not in ci.verdict
    assert f"only {len(gains)} block(s)" in caplog.text and "not reliable" in caplog.text


def test_a_constant_difference_collapses_the_interval() -> None:
    ci = fb.block_bootstrap_ci(
        np.full(70, 95.0), np.full(70, 100.0), np.repeat(np.arange(10), 7), n_boot=500
    )
    assert ci.low == pytest.approx(5.0) and ci.high == pytest.approx(5.0)


def test_the_bootstrap_is_reproducible_with_its_seed() -> None:
    rng = np.random.default_rng(7)
    official = rng.uniform(500, 3_000, 24 * 70)
    model = official * rng.uniform(0.5, 1.3, len(official))
    blocks = np.repeat(np.arange(10), 24 * 7)
    first = fb.block_bootstrap_ci(model, official, blocks, seed=11)
    assert fb.block_bootstrap_ci(model, official, blocks, seed=11) == first
    other = fb.block_bootstrap_ci(model, official, blocks, seed=12)
    assert (other.low, other.high) != (first.low, first.high)
    assert other.improvement_pct == first.improvement_pct  # the point estimate has no randomness


def test_verdicts_with_enough_blocks_and_input_checks() -> None:
    gains = np.linspace(10, 30, fb.MIN_BOOTSTRAP_BLOCKS)
    better = fb.block_bootstrap_ci(*blocks_with_gains(*gains), n_boot=500)
    assert better.reliable and better.low > 0 and better.verdict == "clearly beats"
    worse = fb.block_bootstrap_ci(*blocks_with_gains(*-gains), n_boot=500)
    assert worse.high < 0 and worse.verdict == "clearly loses to"
    mixed = fb.block_bootstrap_ci(*blocks_with_gains(*[20, -20] * (fb.MIN_BOOTSTRAP_BLOCKS // 2)))
    assert mixed.low < 0 < mixed.high and mixed.verdict == "does not clearly beat"
    model, official, blocks = blocks_with_gains(10, 30)
    with pytest.raises(ValueError, match="equally long"):
        fb.block_bootstrap_ci(model[:-1], official, blocks)


def test_autocorrelation_lags_are_calendar_days_even_with_a_missing_day() -> None:
    days = [str(d) for d in pd.date_range("2026-01-05", periods=12).date]
    daily = np.random.default_rng(5).normal(size=len(days))
    kept = [i for i in range(len(days)) if i != 5]  # 10 Jan has no scored hours
    hours = [berlin_day(days[i]) for i in kept]
    index = pd.DatetimeIndex(np.concatenate(hours))
    official = pd.Series(
        np.concatenate([np.full(len(h), daily[i]) for i, h in zip(kept, hours, strict=True)]),
        index=index,
    )
    model = pd.Series(0.0, index=index)  # loss difference = that day's value, every hour

    (lag1,) = fb.daily_loss_autocorr(model, official, lags=1)
    pairs = [(daily[i], daily[i + 1]) for i in kept if i + 1 in kept]  # days 1 apart only
    assert lag1 == pytest.approx(np.corrcoef(*zip(*pairs, strict=True))[0, 1])
    positional = pd.Series(daily[kept]).autocorr(1)  # would pair 9 Jan with 11 Jan
    assert lag1 != pytest.approx(positional)


# --- End to end, no leakage ---
@pytest.mark.parametrize(
    ("as_of", "train_end", "gap_minutes"),
    [
        # Training ends 17 h 20 min after the issue of target day 21 Jan (05:40 UTC on 20 Jan).
        (None, "2025-01-20T23:00Z", -1040),
        # Boundary: training ends 40 min *before* that issue - but its last hour was not yet
        # public at 05:40, so 21 Jan is still dropped.
        ("2025-02-03T05:00Z", "2025-01-20T05:00Z", 40),
    ],
    ids=["default", "train-end-40-min-before-an-issue"],
)
def test_every_holdout_weather_value_was_issued_before_its_forecasts_issue_time(
    engine, monkeypatch, as_of, train_end, gap_minutes
) -> None:
    frame = seed(engine)
    trained: list[pd.DataFrame] = []
    fed: list[pd.DataFrame] = []
    real_fit = train_mod.fit_model

    def spy_fit(df, horizon, params=None):
        trained.append(df)
        model, cols = real_fit(df, horizon, params)
        real_predict = model.predict

        def predict(X):
            fed.append(X.copy())
            return real_predict(X)

        model.predict = predict
        return model, cols

    monkeypatch.setattr(train_mod, "fit_model", spy_fit)
    report = run(engine, frame, as_of=pd.Timestamp(as_of) if as_of else None)
    hours = report.hours

    # Every archived forecast the replay read was public by its issue, never day0.
    for mode in fb.MODES:
        assert (hours[f"available_by_{mode}"] <= hours["issue_time"]).all()
        assert (hours[f"run_init_bound_{mode}"] < hours["issue_time"]).all()
        assert (hours[f"offset_{mode}"] >= 1).all()
    assert set(hours[f"offset_{fb.NEWEST}"]) == {1, 2} and set(hours[f"offset_{fb.DAY2}"]) == {2}

    # The rest of each feature row: its newest load value is older than the issue too.
    lags = [most_recent_load_lag(c) for c in select_features(frame.columns, D_PLUS_1)]
    newest_lag = pd.Timedelta(hours=min(lag for lag in lags if lag is not None))
    assert (hours.index - newest_lag < hours["issue_time"]).all()

    # Trained once, on the archive weather; the last training hour was public by every scored
    # issue, and the target day before the first scored one (21 Jan) was dropped.
    (train_df,) = trained
    assert train_df.index.max() == report.train_end == pd.Timestamp(train_end)
    assert (hours["issue_time"] - fb.ACTUALS_PUBLISHED_LAG >= report.train_end).all()
    assert hours.index.min() == pd.Timestamp("2025-01-21T23:00Z")  # 22 Jan, 00:00 Berlin
    dropped_issue = fb.issue_times(pd.DatetimeIndex([pd.Timestamp("2025-01-20T23:00Z")]))[0]
    assert dropped_issue - report.train_end == pd.Timedelta(minutes=gap_minutes)
    assert dropped_issue - fb.ACTUALS_PUBLISHED_LAG < report.train_end
    pd.testing.assert_frame_equal(
        train_df[list(WEATHER_COLUMNS)], frame.loc[train_df.index, list(WEATHER_COLUMNS)]
    )

    # What the model was fed: archive weather for the measured row, and for the issue-time
    # rows exactly the chosen offset's five-city average - never the archive.
    measured_x, newest_x, day2_x = fed
    pd.testing.assert_frame_equal(
        measured_x.loc[hours.index, list(WEATHER_COLUMNS)],
        frame.loc[hours.index, list(WEATHER_COLUMNS)],
    )
    for x, mode in ((newest_x, fb.NEWEST), (day2_x, fb.DAY2)):
        expected = 100.0 * hours[f"offset_{mode}"].to_numpy() + MEAN_LAT
        for col in WEATHER_COLUMNS:
            np.testing.assert_allclose(x.loc[hours.index, col].to_numpy(), expected)
        np.testing.assert_allclose(
            report.weather[mode][list(WEATHER_COLUMNS)].min(axis=1), expected
        )


def test_the_three_rows_score_the_same_hours(engine) -> None:
    report = run(engine, seed(engine))
    rows = report.rows
    assert set(rows) == {fb.NEWEST, fb.DAY2, fb.MEASURED}
    assert {r.n_hours for r in rows.values()} == {report.n_hours}
    assert len({r.official_mae for r in rows.values()}) == 1
    assert report.n_dropped == 0 and report.n_days == HOLDOUT_DAYS
    assert len(report.autocorr) == fb.AUTOCORR_LAGS

    text_out = report.summary()
    for needle in ("Fair backtest", "95% CI", "bracket", "seed 0", "1-day blocks", "lags 1-7"):
        assert needle in text_out
    # 14 target days are 2 blocks of 7: the summary must not give a verdict.
    assert report.headline.ci.n_blocks == 2
    assert f"-> {fb.TOO_FEW_BLOCKS}" in text_out and "no verdict" in text_out
    assert "clearly" not in text_out
    monthly = report.monthly()
    assert list(monthly.columns) == ["month", "hours", "official", "newest", "day2", "measured"]
    assert monthly["hours"].sum() == report.n_hours


def test_hours_without_complete_archived_forecasts_leave_every_row(engine) -> None:
    def route(url, params):
        payload = previous_runs_route(url, params)
        if float(params["latitude"]) == GERMAN_CITIES[0].latitude:  # one city, one UTC day
            day = [t.startswith("2025-01-25") for t in payload["hourly"]["time"]]
            for name, values in payload["hourly"].items():
                if name.endswith("_previous_day2"):
                    payload["hourly"][name] = [
                        None if d else v for d, v in zip(day, values, strict=True)
                    ]
        return payload

    report = run(engine, seed(engine), weather_client=fake_client(route))
    assert report.n_dropped == 24
    assert not report.hours.index.normalize().isin([pd.Timestamp("2025-01-25T00:00Z")]).any()
    assert {r.n_hours for r in report.rows.values()} == {report.n_hours}
    assert "24 dropped" in report.summary()


def test_an_open_meteo_outage_fails_clearly_after_the_measured_row_is_out(
    engine, monkeypatch
) -> None:
    frame = seed(engine)
    monkeypatch.setattr(wc, "get_json", functools.partial(_http.get_json, sleep=lambda s: None))
    outage = fake_client([FakeResponse({}, status_code=503) for _ in range(4)])
    lines: list[str] = []
    with pytest.raises(RuntimeError, match="Previous Runs weather fetch failed"):
        run(engine, frame, weather_client=outage, progress=lines.append)
    assert len(lines) == 1 and lines[0].startswith("Measured-weather backtest")
    assert "official" in lines[0] and "95% CI" in lines[0]


def test_only_a_calendar_day_horizon_has_an_issue_to_replay(engine) -> None:
    with pytest.raises(ValueError, match="D\\+1 issue"):
        fb.fair_backtest_vs_official(engine, horizon=DAY_AHEAD)


# --- CLI ---
def test_cli_runs_the_fair_backtest_with_read_only_rights(engine, monkeypatch, capsys) -> None:
    seed(engine)
    uri = Path(engine.url.database).resolve().as_uri() + "?mode=ro"
    read_only = sqlalchemy.create_engine(
        "sqlite://", creator=lambda: sqlite3.connect(uri, uri=True)
    )
    with (
        pytest.raises(sqlalchemy.exc.OperationalError, match="readonly"),
        read_only.begin() as conn,
    ):
        conn.execute(text("CREATE TABLE probe (a INTEGER)"))

    def no_schema_check(*args, **kwargs):
        raise AssertionError("the fair backtest must not run the schema check")

    real_fit = train_mod.fit_model
    monkeypatch.setattr(perf.db, "get_engine", lambda url=None: read_only)
    monkeypatch.setattr(db, "init_db", no_schema_check)
    monkeypatch.setattr(db, "_check_schema_version", no_schema_check)
    monkeypatch.setattr(fb, "WeatherClient", fake_client)
    monkeypatch.setattr(
        train_mod, "fit_model", lambda df, horizon, params=None: real_fit(df, horizon, FAST_LGBM)
    )

    assert perf.main(["--backtest-days", str(HOLDOUT_DAYS), "--fair-weather"]) == 0
    out = capsys.readouterr().out
    assert out.index("Measured-weather backtest") < out.index("Fair backtest:")
    assert "Per month" in out
