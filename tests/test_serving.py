"""Serving tests (offline): feature reconstruction parity, forecast window, storing every issue
with its weather inputs, the deadline margin, API."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError, OperationalError

from src.data import db
from src.data.weather_client import WEATHER_VARS, City
from src.features.build_features import TARGET, build_features
from src.features.horizons import D_PLUS_1, DAY_AHEAD, HORIZONS, NOWCAST, select_features
from src.models.registry import CorruptBundleError, LoadedModel
from src.serving import api, batch_forecast
from src.serving.forecast import (
    NoDataError,
    build_serving_features,
    calendar_day_window,
    ensure_weather,
    fill_short_gaps,
    forecast_window,
    make_day_ahead_forecast,
    store_forecast,
    target_window,
)
from tests.conftest import fake_loaded_model, load_curve

CITIES = (City("A", 50.0, 8.0), City("B", 52.0, 13.0))
N_DAYS = 30
LAST_ACTUAL = pd.Timestamp("2024-01-30T23:00Z")  # 30 full days from 2024-01-01


# --- Fixtures ---
def _weather_rows(ts: pd.DatetimeIndex, source: str) -> pd.DataFrame:
    frames = []
    for city in CITIES:
        h = np.arange(len(ts))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp_utc": ts,
                    "city": city.name,
                    "source": source,
                    "temperature_2m": 5 + city.latitude / 10 + 3 * np.sin(2 * np.pi * h / 24),
                    "wind_speed_10m": 10.0,
                    "shortwave_radiation": np.clip(200 * np.sin(2 * np.pi * h / 24), 0, None),
                    "cloud_cover": 50.0,
                    "relative_humidity_2m": 70.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class FakeWeatherClient:
    """Returns forecast rows for the two test cities; records calls."""

    cities = CITIES

    def __init__(self, start: pd.Timestamp) -> None:
        self.start = start
        self.calls: list[dict] = []

    def fetch_forecast(self, *, past_days: int = 2, forecast_days: int = 3) -> pd.DataFrame:
        self.calls.append({"past_days": past_days, "forecast_days": forecast_days})
        ts = pd.date_range(
            self.start - pd.Timedelta(days=past_days),
            periods=24 * (past_days + max(forecast_days, 3)),  # window independent of "now"
            freq="h",
            tz="UTC",
        )
        return _weather_rows(ts, "forecast")


@pytest.fixture
def seeded(engine):
    """30 days of load + archive weather in the DB."""
    ts = pd.date_range("2024-01-01", periods=24 * N_DAYS, freq="h", tz="UTC")
    load = load_curve(len(ts))
    db.upsert_dataframe(
        engine, db.LoadActual, pd.DataFrame({"timestamp_utc": ts, TARGET: load, "source": "smard"})
    )
    db.upsert_dataframe(engine, db.WeatherHourly, _weather_rows(ts, "archive"))
    return engine


def fake_model(features: list[str]) -> LoadedModel:
    """The champion stand-in: last week's load + 100 MW, version 7."""
    return fake_loaded_model(features, version="7", bias=100.0)


@pytest.fixture
def day_ahead_features(seeded) -> list[str]:
    served, _, _ = build_serving_features(seeded, as_of=LAST_ACTUAL - pd.Timedelta(days=5))
    return select_features(served.columns, DAY_AHEAD)


# --- Forecast window ---
def test_forecast_window_is_the_24_hours_after_the_last_actual() -> None:
    w = forecast_window(LAST_ACTUAL)
    assert len(w) == 24
    assert w[0] == LAST_ACTUAL + pd.Timedelta(hours=1)
    assert w[-1] == LAST_ACTUAL + pd.Timedelta(hours=24)
    assert str(w.tz) == "UTC"
    assert len(forecast_window(LAST_ACTUAL, hours=6)) == 6
    for bad in (0, 25, 48):
        with pytest.raises(ValueError, match="honest day-ahead"):
            forecast_window(LAST_ACTUAL, hours=bad)


# --- Train / serve parity ---
def test_served_features_equal_training_features_exactly(seeded) -> None:
    """Serve 'as of' a past hour; the same hours in the training frame must match bit-for-bit."""
    as_of = pd.Timestamp("2024-01-20T23:00Z")
    served, last_actual, _ = build_serving_features(seeded, as_of=as_of)
    assert last_actual == as_of
    assert list(served.index) == list(forecast_window(as_of))

    training = build_features(seeded, output=None)  # the training pipeline, from the same DB
    train_rows = training.loc[served.index]

    cols = select_features(training.columns, DAY_AHEAD)
    assert select_features(served.columns, DAY_AHEAD) == cols  # same names, same order
    pd.testing.assert_frame_equal(served[cols], train_rows[cols], check_exact=True)
    # ... and the target is (correctly) unknown at serving time.
    assert served[TARGET].isna().all()


def test_served_features_never_contain_information_after_the_cutoff(seeded) -> None:
    """Change the load *after* the as-of time: served day-ahead features must not move."""
    as_of = pd.Timestamp("2024-01-20T23:00Z")
    base, _, _ = build_serving_features(seeded, as_of=as_of)
    cols = select_features(base.columns, DAY_AHEAD)

    later = db.read_load(seeded, start=as_of + pd.Timedelta(hours=1))
    later["load_mw"] += 1e6
    db.upsert_dataframe(seeded, db.LoadActual, later.assign(source="smard"))

    after, _, _ = build_serving_features(seeded, as_of=as_of)
    pd.testing.assert_frame_equal(base[cols], after[cols], check_exact=True)


def test_nowcast_only_features_are_nan_for_the_served_hours(seeded) -> None:
    """lag_1 / rolling stats of the target hours depend on unobserved load -> NaN, never used."""
    served, _, _ = build_serving_features(seeded, as_of="2024-01-20T23:00Z")
    nowcast_only = set(select_features(served.columns, NOWCAST)) - set(
        select_features(served.columns, DAY_AHEAD)
    )
    assert "load_lag_1" in nowcast_only
    assert served.loc[served.index[1:], "load_lag_1"].isna().all()  # only hour 1 has lag_1
    last = served.loc[served.index[-1]]
    assert last[["load_lag_1", *[c for c in nowcast_only if c.endswith("_24")]]].isna().all()
    # (168h rolling stats still compute - min_periods is 75% of the window - but they
    # contain unobserved hours and are excluded for day-ahead just the same.)


def _delete_load_hours(engine, start: str, hours: int) -> pd.DatetimeIndex:
    from sqlalchemy import delete

    gap = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    with engine.begin() as conn:
        conn.execute(delete(db.LoadActual).where(db.LoadActual.timestamp_utc.in_(list(gap))))
    return gap


def test_fill_short_gaps_bridges_short_runs_only() -> None:
    ts = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    full = pd.DataFrame({"timestamp_utc": ts, TARGET: np.arange(20, dtype=float) * 10})
    history = full.drop(index=[3, 4, 5, 10, 11, 12, 13])  # a 3 h and a 4 h gap

    out = fill_short_gaps(history, end=ts[-1]).set_index("timestamp_utc")[TARGET]
    assert out.loc[ts[3:6]].tolist() == [30.0, 40.0, 50.0]  # linear between neighbours
    assert ts[10:14].isin(out.index).sum() == 0  # too long: left missing for the lag check
    assert fill_short_gaps(full, end=ts[-1]) is full  # nothing missing -> untouched


def test_one_missing_hour_is_interpolated_for_serving_only(seeded) -> None:
    (gap,) = _delete_load_hours(seeded, "2024-01-20T05:00Z", 1)  # t-24 for 2024-01-21T05
    served, _, _ = build_serving_features(seeded, as_of="2024-01-20T23:00Z")
    lag = served.loc[pd.Timestamp("2024-01-21T05:00Z"), "load_lag_24"]
    load = db.read_load(seeded).set_index("timestamp_utc")[TARGET]
    neighbours = load.loc[[gap - pd.Timedelta(hours=1), gap + pd.Timedelta(hours=1)]]
    assert lag == pytest.approx(neighbours.mean())
    assert gap not in load.index  # the interpolated value is never stored as an actual


def test_long_gap_in_recent_load_is_still_rejected(seeded) -> None:
    _delete_load_hours(seeded, "2024-01-20T03:00Z", 6)  # covers t-24 for the first targets
    with pytest.raises(NoDataError, match=r"gaps longer than 3h .*load_lag_24"):
        build_serving_features(seeded, as_of="2024-01-20T23:00Z")


# --- Calendar-day D+1 ---
@pytest.mark.parametrize(
    ("day", "hours", "first_utc"),
    [
        ("2024-01-21", 24, "2024-01-20T23:00Z"),  # winter: Berlin = UTC+1
        ("2024-03-31", 23, "2024-03-30T23:00Z"),  # clocks go forward: a 23-hour day
        ("2024-10-27", 25, "2024-10-26T22:00Z"),  # clocks go back: a 25-hour day
    ],
)
def test_calendar_day_window_follows_the_berlin_clock(day, hours, first_utc) -> None:
    w = calendar_day_window(pd.Timestamp(day))
    assert len(w) == hours and w[0] == pd.Timestamp(first_utc)
    assert (w.tz_convert("Europe/Berlin").date == pd.Timestamp(day).date()).all()


def test_d_plus_1_targets_tomorrow_and_refuses_stale_actuals() -> None:
    issued = pd.Timestamp("2024-10-26T05:40Z")  # Saturday morning; Sunday has 25 hours
    w = target_window(D_PLUS_1, pd.Timestamp("2024-10-26T03:00Z"), issued)
    assert len(w) == 25 and w[0] == pd.Timestamp("2024-10-26T22:00Z")
    # A rolling horizon ignores the calendar.
    assert target_window(DAY_AHEAD, pd.Timestamp("2024-10-26T03:00Z"), issued)[0] == (
        pd.Timestamp("2024-10-26T04:00Z")
    )
    # Actuals ending 50 h before the last target hour: lag_48 cannot cover the whole day.
    with pytest.raises(NoDataError, match="beyond the 48h cutoff of d_plus_1"):
        target_window(D_PLUS_1, pd.Timestamp("2024-10-25T20:00Z"), issued)


def test_d_plus_1_forecast_covers_the_next_local_day_without_the_24h_lag(seeded) -> None:
    as_of = pd.Timestamp("2024-01-20T05:00Z")  # issued Saturday morning, latest actual 05:00
    served, last_actual, _ = build_serving_features(seeded, horizon=D_PLUS_1, as_of=as_of)
    assert list(served.index) == list(calendar_day_window(pd.Timestamp("2024-01-21")))
    assert last_actual == as_of

    features = select_features(served.columns, D_PLUS_1)
    assert "load_lag_24" not in features and "load_lag_48" in features
    model = fake_loaded_model(features, version="9", horizon=D_PLUS_1)
    result = make_day_ahead_forecast(seeded, model, as_of=as_of)
    assert result.horizon == "d_plus_1" and len(result.frame) == 24
    assert result.target_start == pd.Timestamp("2024-01-20T23:00Z")


def test_empty_db_raises(engine) -> None:
    with pytest.raises(NoDataError, match="no actual load"):
        build_serving_features(engine)


# --- Weather at serving time ---
def test_ensure_weather_fetches_forecast_only_when_missing(seeded) -> None:
    target = forecast_window(LAST_ACTUAL)  # beyond the archive -> missing
    client = FakeWeatherClient(start=LAST_ACTUAL)
    ensure_weather(seeded, target, client)
    assert len(client.calls) == 1
    have = db.read_table(seeded, db.WeatherHourly, start=target.min(), end=target.max())
    assert len(have) == 24 * len(CITIES)
    assert (have["source"] == "forecast").all()

    ensure_weather(seeded, target, client)  # now covered -> no second fetch
    assert len(client.calls) == 1

    # A past window already covered by the archive never triggers a fetch.
    ensure_weather(seeded, forecast_window(pd.Timestamp("2024-01-10T23:00Z")), client)
    assert len(client.calls) == 1


def test_forecast_weather_does_not_overwrite_archive(seeded) -> None:
    before = db.read_table(seeded, db.WeatherHourly, start="2024-01-29T00:00Z")
    client = FakeWeatherClient(start=LAST_ACTUAL)  # past_days=2 overlaps the archive
    ensure_weather(seeded, forecast_window(LAST_ACTUAL), client)
    after = db.read_table(seeded, db.WeatherHourly, start="2024-01-29T00:00Z", end=LAST_ACTUAL)
    pd.testing.assert_frame_equal(
        before[before["timestamp_utc"] <= LAST_ACTUAL].reset_index(drop=True)[
            ["temperature_2m", "source"]
        ],
        after.reset_index(drop=True)[["temperature_2m", "source"]],
    )


def test_missing_weather_without_client_gives_nan_weather_not_an_error(seeded) -> None:
    served, _, _ = build_serving_features(seeded)  # target beyond the archive, no client
    assert served.filter(like="_de_avg").isna().all().all()
    assert served[["load_lag_24", "load_lag_168", "hour"]].notna().all().all()


# --- Forecast + batch storage ---
def test_make_day_ahead_forecast_uses_champion_and_served_features(
    seeded, day_ahead_features
) -> None:
    model = fake_model(day_ahead_features)
    result = make_day_ahead_forecast(seeded, model, weather_client=FakeWeatherClient(LAST_ACTUAL))

    assert result.model_version == "7" and result.horizon == "day_ahead"
    assert result.last_actual == LAST_ACTUAL
    assert list(result.frame.index) == list(forecast_window(LAST_ACTUAL))
    assert list(result.frame.columns) == ["forecast_mw", *day_ahead_features]
    np.testing.assert_allclose(result.frame["forecast_mw"], result.frame["load_lag_168"] + 100)
    assert result.issued_at.tzinfo is not None


def test_forecast_refuses_model_with_different_features(seeded, day_ahead_features) -> None:
    wrong = fake_model([*day_ahead_features, "load_lag_1"])
    with pytest.raises(ValueError, match="train/serve feature mismatch"):
        make_day_ahead_forecast(seeded, wrong)
    reordered = fake_model(list(reversed(day_ahead_features)))
    with pytest.raises(ValueError, match="train/serve feature mismatch"):
        make_day_ahead_forecast(seeded, reordered)


def test_store_forecast_keeps_every_issue_with_its_weather(seeded, day_ahead_features) -> None:
    model = fake_model(day_ahead_features)
    r1 = make_day_ahead_forecast(seeded, model, weather_client=FakeWeatherClient(LAST_ACTUAL))
    assert len(r1.weather) == len(CITIES) * 24
    assert store_forecast(seeded, r1) == 24
    assert store_forecast(seeded, r1) == 24  # the same issue again adds no rows
    assert db.count_rows(seeded, db.LoadForecastModel) == 24
    assert db.count_rows(seeded, db.LoadForecastModelWeather) == len(CITIES) * 24
    assert (db.read_table(seeded, db.LoadForecastModel)["horizon"] == "day_ahead").all()

    # A re-issue with the same version is added; the earlier issue stays.
    r2 = make_day_ahead_forecast(seeded, model, weather_client=FakeWeatherClient(LAST_ACTUAL))
    r2.issued_at = r1.issued_at + pd.Timedelta(minutes=5)
    store_forecast(seeded, r2)
    assert db.count_rows(seeded, db.LoadForecastModel) == 48
    assert db.count_rows(seeded, db.LoadForecastModel, issued_at=r1.issued_at) == 24
    assert db.count_rows(seeded, db.LoadForecastModelWeather) == 2 * len(CITIES) * 24
    latest = db.read_latest_model_forecast(seeded)
    assert len(latest) == 24 and (latest["issued_at"] == r2.issued_at).all()
    assert (latest["model_name"] == "germany-load-day-ahead").all()

    # A different model version adds its own issue.
    v8 = fake_model(day_ahead_features)
    v8.version = "8"
    store_forecast(
        seeded, make_day_ahead_forecast(seeded, v8, weather_client=FakeWeatherClient(LAST_ACTUAL))
    )
    assert db.count_rows(seeded, db.LoadForecastModel) == 72
    assert db.count_rows(seeded, db.LoadForecastModel, model_version="8") == 24


def test_stored_weather_inputs_are_what_was_read_and_survive_the_archive(
    seeded, day_ahead_features
) -> None:
    result = make_day_ahead_forecast(
        seeded, fake_model(day_ahead_features), weather_client=FakeWeatherClient(LAST_ACTUAL)
    )
    store_forecast(seeded, result)
    target = dict(start=result.target_start, end=result.target_end)
    cols = ["timestamp_utc", "city", "source", *WEATHER_VARS, "fetched_at"]
    keys = ["timestamp_utc", "city"]
    read = db.read_table(seeded, db.WeatherHourly, **target)
    inputs = db.read_table(seeded, db.LoadForecastModelWeather)
    assert set(inputs["issued_at"]) == {result.issued_at}
    assert (inputs["source"] == "forecast").all()
    pd.testing.assert_frame_equal(
        inputs[cols].sort_values(keys).reset_index(drop=True),
        read.rename(columns={"ingested_at": "fetched_at"})[cols]
        .sort_values(keys)
        .reset_index(drop=True),
    )

    # Days later the archive overwrites those hours in weather_hourly; the inputs stay.
    archive = _weather_rows(result.frame.index, "archive").assign(temperature_2m=-30.0)
    db.upsert_dataframe(seeded, db.WeatherHourly, archive, update_where=db.weather_update_where)
    assert (db.read_table(seeded, db.WeatherHourly, **target)["temperature_2m"] == -30.0).all()
    pd.testing.assert_frame_equal(db.read_table(seeded, db.LoadForecastModelWeather), inputs)


def test_a_failed_weather_write_stores_no_forecast_rows(seeded, day_ahead_features) -> None:
    """One transaction: an issue is stored with its weather inputs or not at all."""
    result = make_day_ahead_forecast(
        seeded, fake_model(day_ahead_features), weather_client=FakeWeatherClient(LAST_ACTUAL)
    )
    result.weather = result.weather.assign(source=None)  # NOT NULL in the inputs table
    with pytest.raises(IntegrityError):
        store_forecast(seeded, result)
    assert db.count_rows(seeded, db.LoadForecastModel) == 0
    assert db.count_rows(seeded, db.LoadForecastModelWeather) == 0


def test_deadline_margin_is_measured_to_the_earliest_deadline(seeded, day_ahead_features) -> None:
    result = make_day_ahead_forecast(
        seeded, fake_model(day_ahead_features), weather_client=FakeWeatherClient(LAST_ACTUAL)
    )
    # day_ahead: each hour is its own deadline, the first target hour comes first
    result.issued_at = result.target_start - pd.Timedelta(hours=2)
    assert result.deadline_margin_hours == pytest.approx(2.0)
    # d_plus_1 over 31 Jan 00:00-23:00 UTC: the 00:00 hour is 31 Jan in Berlin, due 30 Jan
    # 11:00 UTC; the 23:00 hour is already 1 Feb, due a day later. The earlier one counts.
    d1 = replace(result, horizon=D_PLUS_1.name, issued_at=pd.Timestamp("2024-01-30T05:40Z"))
    assert d1.deadline_margin_hours == pytest.approx(5 + 20 / 60)
    late = replace(d1, issued_at=pd.Timestamp("2024-01-30T12:00Z"))
    assert late.deadline_margin_hours == pytest.approx(-1.0)


def test_batch_cli_dry_run_and_write(
    seeded, day_ahead_features, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        batch_forecast, "load_champion", lambda engine: fake_model(day_ahead_features)
    )
    monkeypatch.setattr(batch_forecast, "WeatherClient", lambda: FakeWeatherClient(LAST_ACTUAL))
    url = str(seeded.url)

    assert batch_forecast.main(["--dry-run", "--database-url", url]) == 0
    assert db.count_rows(seeded, db.LoadForecastModel) == 0
    assert db.count_rows(seeded, db.LoadForecastModelWeather) == 0
    out = capsys.readouterr().out
    assert "germany-load-day-ahead v7" in out and "deadline margin" in out

    assert batch_forecast.main(["--database-url", url, "--hours", "12"]) == 0
    assert db.count_rows(seeded, db.LoadForecastModel) == 12
    assert db.count_rows(seeded, db.LoadForecastModelWeather) == 12 * len(CITIES)


def test_batch_refuses_to_store_an_as_of_replay(
    seeded, day_ahead_features, monkeypatch, capsys
) -> None:
    """Stored, a replay would be a late issue that shows as the latest forecast."""
    monkeypatch.setattr(
        batch_forecast, "load_champion", lambda engine: fake_model(day_ahead_features)
    )
    monkeypatch.setattr(batch_forecast, "WeatherClient", lambda: FakeWeatherClient(LAST_ACTUAL))
    url = str(seeded.url)
    as_of = "2024-01-20T23:00Z"

    with pytest.raises(SystemExit) as exit_info:
        batch_forecast.main(["--as-of", as_of, "--database-url", url])
    assert exit_info.value.code == 2
    assert "only allowed with --dry-run" in capsys.readouterr().err
    with pytest.raises(ValueError, match="only allowed with --dry-run"):
        batch_forecast.run(as_of=pd.Timestamp(as_of), database_url=url)
    assert db.count_rows(seeded, db.LoadForecastModel) == 0
    assert db.count_rows(seeded, db.LoadForecastModelWeather) == 0

    assert batch_forecast.main(["--as-of", as_of, "--dry-run", "--database-url", url]) == 0
    assert "target 2024-01-21 00:00:00+00:00" in capsys.readouterr().out
    assert db.count_rows(seeded, db.LoadForecastModel) == 0


def test_batch_once_per_day_skips_when_today_is_already_issued(
    seeded, day_ahead_features, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        batch_forecast, "load_champion", lambda engine: fake_model(day_ahead_features)
    )
    monkeypatch.setattr(batch_forecast, "WeatherClient", lambda: FakeWeatherClient(LAST_ACTUAL))
    url = str(seeded.url)

    assert batch_forecast.main(["--database-url", url, "--once-per-day"]) == 0  # first issue
    first = db.latest_issued_at(seeded)
    assert first is not None and db.count_rows(seeded, db.LoadForecastModel) == 24

    capsys.readouterr()
    assert batch_forecast.main(["--database-url", url, "--once-per-day"]) == 0  # late backup
    assert "already issued" in capsys.readouterr().out
    assert db.latest_issued_at(seeded) == first  # nothing re-issued

    assert batch_forecast.main(["--database-url", url]) == 0  # a forced run re-issues
    assert db.latest_issued_at(seeded) > first


# --- API ---
NOW = LAST_ACTUAL + pd.Timedelta(hours=2)  # the API's wall clock in these tests
SHA7 = "7" * 64


class FakeClock:
    """Monotonic clock for the status TTL and reload retries; advanced by hand."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeChampionLoader:
    """``registry.load_champion`` without MLflow: builds the model from the DB champion row.

    ``error`` makes the next loads raise; ``gate`` (an Event) makes a load wait for it.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None
        self.gate: threading.Event | None = None

    def __call__(self, engine) -> LoadedModel:
        self.calls += 1
        if self.gate is not None:
            assert self.gate.wait(timeout=10)
        if self.error is not None:
            raise self.error
        row = db.get_model_artifact(engine, "germany-load-day-ahead", champion=True)
        model = fake_loaded_model(
            json.loads(row["features"]),
            version=row["version"],
            bias=100.0,
            horizon=HORIZONS[row["horizon"]],
        )
        return replace(model, bundle_sha256=row["bundle_sha256"])


def export(
    engine, version: str, features: list[str], *, sha: str, horizon: str = "day_ahead"
) -> None:
    """A ``model_artifacts`` row as ``registry.export_model_version`` writes it (fake bundle)."""
    db.store_model_artifact(
        engine,
        name="germany-load-day-ahead",
        version=version,
        run_id=f"run-{version}",
        horizon=horizon,
        features=json.dumps(features),
        bundle=b"zip",
        bundle_sha256=sha,
        bundle_bytes=3,
    )


def promote(engine, version: str, features: list[str], **kwargs) -> None:
    export(engine, version, features, **kwargs)
    db.set_champion(engine, "germany-load-day-ahead", version)


def issue_forecast(engine, features: list[str], issued_at: pd.Timestamp, version="7"):
    """Store one batch issue made by ``version`` at ``issued_at``."""
    model = fake_loaded_model(features, version=version, bias=100.0)
    result = make_day_ahead_forecast(engine, model, weather_client=FakeWeatherClient(LAST_ACTUAL))
    result = replace(result, issued_at=issued_at)
    store_forecast(engine, result)
    return result


@dataclass
class Api:
    client: TestClient
    clock: FakeClock
    loader: FakeChampionLoader
    wall: list[pd.Timestamp]  # [now]; replace the item to move the wall clock

    @property
    def state(self) -> api.AppState:
        return self.client.app.state.ctx

    def get(self, path: str, **kwargs):
        return self.client.get(path, **kwargs)

    def after_ttl(self) -> None:
        self.clock.advance(api.STATUS_TTL_SECONDS + 1)


def make_api(engine, features: list[str], *, background_reload: bool = False):
    """A client serving v7 (exported and DB champion), with a fake clock and loader."""
    promote(engine, "7", features, sha=SHA7)
    clock, loader, wall = FakeClock(), FakeChampionLoader(), [NOW]
    initial = replace(fake_model(features), bundle_sha256=SHA7)

    def state_loader() -> api.AppState:
        return api.AppState(
            engine=engine,
            model=initial,
            weather_client=FakeWeatherClient(LAST_ACTUAL),
            started_at=datetime.now(UTC),
            model_loader=loader,
        )

    config = api.ServingConfig(
        background_reload=background_reload, now=lambda: wall[0], monotonic=clock
    )
    return TestClient(api.create_app(state_loader, config=config)), clock, loader, wall


@pytest.fixture
def served(seeded, day_ahead_features):
    client, clock, loader, wall = make_api(seeded, day_ahead_features)
    with client:
        yield Api(client, clock, loader, wall)


@pytest.fixture
def client(served):
    return served.client


def test_health_ok_when_data_is_fresh(served, seeded, day_ahead_features) -> None:
    issue_forecast(seeded, day_ahead_features, NOW - pd.Timedelta(hours=1))
    served.after_ttl()
    r = served.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["reasons"] == []
    assert body["database"] == "sqlite"
    assert body["model"] == {
        "name": "germany-load-day-ahead",
        "version": "7",
        "alias": "champion",
        "horizon": "day_ahead",
        "n_features": len(day_ahead_features),
        "features": day_ahead_features,
    }
    assert body["champion_db_version"] == "7"
    assert body["last_actual_utc"].startswith("2024-01-30T23:00:00")
    assert body["last_actual_age_hours"] == 2.0
    assert body["last_forecast_age_hours"] == 1.0


@pytest.mark.parametrize(
    ("actual_age", "forecast_age", "reasons"),
    [
        (6.0, 25.0, []),  # at the thresholds: still ok
        (6.05, 25.0, ["latest actual older than 6 h"]),
        (6.0, 25.05, ["latest forecast older than 25 h"]),
        (30.0, 48.0, ["latest actual older than 6 h", "latest forecast older than 25 h"]),
    ],
)
def test_health_degraded_when_data_is_stale(
    served, seeded, day_ahead_features, actual_age, forecast_age, reasons
) -> None:
    now = LAST_ACTUAL + pd.Timedelta(hours=actual_age)
    issue_forecast(seeded, day_ahead_features, now - pd.Timedelta(hours=forecast_age))
    served.wall[0] = now
    served.after_ttl()
    r = served.get("/health")
    assert r.status_code == 200  # degraded is not dead
    assert r.json()["reasons"] == reasons
    assert r.json()["status"] == ("degraded" if reasons else "ok")


def test_health_degraded_without_forecast_or_champion(served, seeded) -> None:
    body = served.get("/health").json()
    assert body["reasons"] == ["no forecast issued yet"]
    with seeded.begin() as conn:
        conn.execute(db.ModelArtifact.__table__.update().values(is_champion=False))
    served.after_ttl()
    body = served.get("/health").json()
    assert body["reasons"] == ["no forecast issued yet", "no champion in database"]
    assert body["champion_db_version"] is None
    assert body["model"]["version"] == "7"  # still serving what it has


def test_health_status_is_cached_for_the_ttl(served, seeded, day_ahead_features) -> None:
    assert served.get("/health").json()["last_forecast_issued_utc"] is None
    issue_forecast(seeded, day_ahead_features, NOW - pd.Timedelta(hours=1))
    assert served.get("/health").json()["last_forecast_issued_utc"] is None  # cached
    served.after_ttl()
    assert served.get("/health").json()["last_forecast_issued_utc"] is not None


def test_status_is_read_in_one_round_trip(seeded, day_ahead_features) -> None:
    promote(seeded, "7", day_ahead_features, sha=SHA7)
    issue_forecast(seeded, day_ahead_features, NOW - pd.Timedelta(hours=1))
    statements: list[str] = []

    def count(conn, cursor, statement, *args) -> None:
        statements.append(statement)

    event.listen(seeded, "before_cursor_execute", count)
    try:
        status = db.serving_status(seeded, "germany-load-day-ahead")
    finally:
        event.remove(seeded, "before_cursor_execute", count)
    assert len(statements) == 1
    assert status == db.ServingStatus(
        last_actual=LAST_ACTUAL,
        last_issued=NOW - pd.Timedelta(hours=1),
        champion_version="7",
        champion_sha256=SHA7,
    )
    assert db.serving_status(seeded, "another-model").champion_version is None


def test_unreachable_database_gives_a_label_not_the_error(served, monkeypatch, caplog) -> None:
    secret = "connection to server at ep-secret-host.neon.tech failed: password hunter2"

    def boom(engine, name):
        raise OperationalError("SELECT 1", {}, Exception(secret))

    monkeypatch.setattr(api.db, "serving_status", boom)
    served.after_ttl()
    with caplog.at_level(logging.ERROR, logger="src.serving.api"):
        r = served.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["reasons"] == ["database unreachable"]
    assert body["database"] == "unreachable"
    assert "ep-secret-host" not in r.text and "hunter2" not in r.text
    assert "ep-secret-host" in caplog.text  # the full error goes to the log only


def test_api_engine_has_a_short_connect_timeout(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(db, "create_engine", lambda url, **kw: captured.update(kw))
    db.get_engine("postgresql://u:p@h/d", connect_timeout=api.DB_CONNECT_TIMEOUT_SECONDS)
    assert captured["connect_args"] == {"connect_timeout": api.DB_CONNECT_TIMEOUT_SECONDS}
    assert api.DB_CONNECT_TIMEOUT_SECONDS < 5  # inside the Docker healthcheck's timeout
    captured.clear()
    db.get_engine("sqlite:///:memory:", connect_timeout=3)  # Postgres only
    assert "connect_args" not in captured


def test_api_follows_a_new_champion_after_the_ttl(served, seeded, day_ahead_features) -> None:
    promote(seeded, "8", day_ahead_features, sha="8" * 64)
    assert served.get("/forecast").json()["model"]["version"] == "7"  # within the TTL
    assert served.loader.calls == 0
    served.after_ttl()
    assert served.get("/forecast").json()["model"]["version"] == "8"
    body = served.get("/health").json()
    assert body["model"]["version"] == "8" and body["champion_db_version"] == "8"
    assert "champion" not in " ".join(body["reasons"])
    served.after_ttl()
    served.get("/forecast")
    assert served.loader.calls == 1  # no reload while the pointer is unchanged

    # a rollback is the same move back
    db.set_champion(seeded, "germany-load-day-ahead", "7")
    served.after_ttl()
    assert served.get("/forecast").json()["model"]["version"] == "7"


def test_a_new_bundle_under_the_same_version_reloads(served, seeded, day_ahead_features) -> None:
    export(seeded, "7", day_ahead_features, sha="e" * 64)  # re-export keeps the champion flag
    served.after_ttl()
    served.get("/health")
    assert served.loader.calls == 1
    assert served.state.served.model.bundle_sha256 == "e" * 64


def test_corrupt_champion_keeps_the_old_model_and_is_not_retried(
    served, seeded, day_ahead_features
) -> None:
    served.loader.error = CorruptBundleError("corrupt model bundle at /secret/path")
    promote(seeded, "8", day_ahead_features, sha="8" * 64)
    served.after_ttl()
    assert served.get("/forecast").json()["model"]["version"] == "7"
    r = served.get("/health")
    assert r.json()["status"] == "degraded"
    assert "champion reload failed (CorruptBundleError)" in r.json()["reasons"]
    assert "/secret/path" not in r.text
    served.clock.advance(api.RELOAD_RETRY_SECONDS * 10)
    served.get("/forecast")
    assert served.loader.calls == 1  # the same bundle will never load
    # a new export (another SHA) is tried again
    served.loader.error = None
    promote(seeded, "9", day_ahead_features, sha="9" * 64)
    served.after_ttl()
    assert served.get("/forecast").json()["model"]["version"] == "9"
    assert served.get("/health").json()["reasons"] == ["no forecast issued yet"]


def test_a_transient_reload_failure_is_retried_after_the_wait(
    served, seeded, day_ahead_features
) -> None:
    served.loader.error = RuntimeError("timeout talking to host-x")
    promote(seeded, "8", day_ahead_features, sha="8" * 64)
    served.after_ttl()
    assert "champion reload failed (RuntimeError)" in served.get("/health").json()["reasons"]
    served.after_ttl()
    served.get("/forecast")
    assert served.loader.calls == 1  # not every TTL: each try downloads the bundle
    served.loader.error = None
    served.clock.advance(api.RELOAD_RETRY_SECONDS)
    assert served.get("/forecast").json()["model"]["version"] == "8"
    assert served.loader.calls == 2


def test_reload_in_background_never_blocks_requests(seeded, day_ahead_features) -> None:
    client, clock, loader, _ = make_api(seeded, day_ahead_features, background_reload=True)
    with client:
        state: api.AppState = client.app.state.ctx
        loader.gate = threading.Event()
        promote(seeded, "8", day_ahead_features, sha="8" * 64)
        clock.advance(api.STATUS_TTL_SECONDS + 1)
        before = state.served
        assert client.get("/forecast").json()["model"]["version"] == "7"  # served meanwhile
        assert "champion reload in progress" in client.get("/health").json()["reasons"]
        clock.advance(api.STATUS_TTL_SECONDS + 1)
        assert client.get("/forecast").json()["model"]["version"] == "7"
        assert loader.calls == 1  # the lock is held: no second reload started
        assert state.served is before  # nothing half-swapped
        loader.gate.set()
        state.reload_thread.join(timeout=10)
        after = state.served
        assert after is not before
        assert after.pointer == ("8", "8" * 64)  # model, version and SHA swapped together
        assert client.get("/forecast").json()["model"]["version"] == "8"


def test_forecast_endpoint_returns_24_hours(client) -> None:
    r = client.get("/forecast")
    assert r.status_code == 200
    body = r.json()
    assert body["horizon"] == "day_ahead"
    assert body["model"]["version"] == "7"
    assert body["last_actual_utc"].startswith("2024-01-30T23:00:00")
    assert body["target_start_utc"].startswith("2024-01-31T00:00:00")
    assert body["target_end_utc"].startswith("2024-01-31T23:00:00")
    pts = body["forecast"]
    assert len(pts) == 24
    stamps = pd.to_datetime([p["timestamp_utc"] for p in pts], utc=True)
    assert list(stamps) == list(forecast_window(LAST_ACTUAL))
    assert all(30_000 < p["forecast_mw"] < 90_000 for p in pts)


def test_forecast_endpoint_hours_param(client) -> None:
    assert len(client.get("/forecast", params={"hours": 6}).json()["forecast"]) == 6
    assert client.get("/forecast", params={"hours": 25}).status_code == 422
    assert client.get("/forecast", params={"hours": 0}).status_code == 422


def test_forecast_latest_404_then_serves_batch_output(served, seeded, day_ahead_features) -> None:
    assert served.get("/forecast/latest").status_code == 404

    result = issue_forecast(seeded, day_ahead_features, NOW - pd.Timedelta(hours=1))
    r = served.get("/forecast/latest")
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 24
    assert body["model"]["version"] == "7" and body["model"]["alias"] == "champion"
    assert pd.Timestamp(body["issued_at"]) == result.issued_at.floor("us")


def test_forecast_latest_describes_the_model_that_issued_it(
    served, seeded, day_ahead_features
) -> None:
    issue_forecast(seeded, day_ahead_features, NOW - pd.Timedelta(hours=1))
    d1_features = select_features(day_ahead_features, D_PLUS_1)
    assert d1_features != day_ahead_features
    promote(seeded, "8", d1_features, sha="8" * 64, horizon="d_plus_1")
    served.after_ttl()
    served.get("/health")
    assert served.state.served.model.horizon.name == "d_plus_1"  # the new champion is loaded

    body = served.get("/forecast/latest").json()
    assert body["horizon"] == "day_ahead"  # the stored issue's, not the loaded model's
    assert body["model"] == {
        "name": "germany-load-day-ahead",
        "version": "7",
        "alias": None,  # no longer the champion
        "horizon": "day_ahead",
        "n_features": len(day_ahead_features),
        "features": day_ahead_features,
    }

    # v7's export pruned: its features are unknown, not borrowed from the loaded model
    with seeded.begin() as conn:
        conn.execute(db.ModelArtifact.__table__.delete().where(db.ModelArtifact.version == "7"))
    body = served.get("/forecast/latest").json()
    assert body["model"]["features"] is None and body["model"]["n_features"] is None
    assert body["model"]["horizon"] == "day_ahead"


def test_forecast_endpoint_503_without_data(tmp_path, day_ahead_features) -> None:
    empty = db.get_engine(f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    db.init_db(empty)

    def loader() -> api.AppState:
        return api.AppState(
            engine=empty,
            model=fake_model(day_ahead_features),
            weather_client=None,
            started_at=datetime.now(UTC),
        )

    with TestClient(api.create_app(loader)) as c:
        r = c.get("/forecast")
        health = c.get("/health").json()
    assert r.status_code == 503
    assert "no actual load" in r.json()["detail"]
    assert health["reasons"] == [
        "no actual load stored",
        "no forecast issued yet",
        "no champion in database",
    ]


def test_openapi_lists_the_endpoints() -> None:
    paths = api.create_app(lambda: None).openapi()["paths"]
    assert {"/health", "/forecast", "/forecast/latest"} <= set(paths)
