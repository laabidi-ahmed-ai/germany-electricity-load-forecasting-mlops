"""Serving tests (offline): feature reconstruction parity, forecast window, batch upsert, API."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data import db
from src.data.weather_client import City
from src.features.build_features import TARGET, build_features
from src.features.horizons import DAY_AHEAD, NOWCAST, select_features
from src.models.registry import LoadedModel
from src.serving import api, batch_forecast
from src.serving.forecast import (
    NoDataError,
    build_serving_features,
    ensure_weather,
    forecast_window,
    make_day_ahead_forecast,
    store_forecast,
)

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
    rng = np.random.default_rng(0)
    h = np.arange(len(ts))
    load = 55_000 + 10_000 * np.sin(2 * np.pi * (h - 6) / 24) + 3_000 * np.sin(2 * np.pi * h / 168)
    load += rng.normal(0, 300, len(ts))
    db.upsert_dataframe(
        engine, db.LoadActual, pd.DataFrame({"timestamp_utc": ts, TARGET: load, "source": "smard"})
    )
    db.upsert_dataframe(engine, db.WeatherHourly, _weather_rows(ts, "archive"))
    return engine


class StubPyfunc:
    """Stands in for an MLflow pyfunc model: predicts last week's load + 100."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["load_lag_168"].to_numpy() + 100.0


def fake_model(features: list[str]) -> LoadedModel:
    return LoadedModel(
        name="germany-load-day-ahead",
        version="7",
        alias="champion",
        run_id="run-xyz",
        features=features,
        horizon=DAY_AHEAD,
        pyfunc=StubPyfunc(),
    )


@pytest.fixture
def day_ahead_features(seeded) -> list[str]:
    served, _ = build_serving_features(seeded, as_of=LAST_ACTUAL - pd.Timedelta(days=5))
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
    served, last_actual = build_serving_features(seeded, as_of=as_of)
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
    base, _ = build_serving_features(seeded, as_of=as_of)
    cols = select_features(base.columns, DAY_AHEAD)

    later = db.read_load(seeded, start=as_of + pd.Timedelta(hours=1))
    later["load_mw"] += 1e6
    db.upsert_dataframe(seeded, db.LoadActual, later.assign(source="smard"))

    after, _ = build_serving_features(seeded, as_of=as_of)
    pd.testing.assert_frame_equal(base[cols], after[cols], check_exact=True)


def test_nowcast_only_features_are_nan_for_the_served_hours(seeded) -> None:
    """lag_1 / rolling stats of the target hours depend on unobserved load -> NaN, never used."""
    served, _ = build_serving_features(seeded, as_of="2024-01-20T23:00Z")
    nowcast_only = set(select_features(served.columns, NOWCAST)) - set(
        select_features(served.columns, DAY_AHEAD)
    )
    assert "load_lag_1" in nowcast_only
    assert served.loc[served.index[1:], "load_lag_1"].isna().all()  # only hour 1 has lag_1
    last = served.loc[served.index[-1]]
    assert last[["load_lag_1", *[c for c in nowcast_only if c.endswith("_24")]]].isna().all()
    # (168h rolling stats still compute - min_periods is 75% of the window - but they
    # contain unobserved hours and are excluded for day-ahead just the same.)


def test_gap_in_recent_load_is_rejected(seeded) -> None:
    from sqlalchemy import delete

    gap = pd.Timestamp("2024-01-20T05:00Z")  # t-24 for the target hour 2024-01-21T05
    with seeded.begin() as conn:
        conn.execute(delete(db.LoadActual).where(db.LoadActual.timestamp_utc == gap))
    with pytest.raises(NoDataError, match="load_lag_24"):
        build_serving_features(seeded, as_of="2024-01-20T23:00Z")


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
    served, _ = build_serving_features(seeded)  # target beyond the archive, no client
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
    recs = result.to_records()
    assert len(recs) == 24 and recs[0]["timestamp_utc"].endswith("+00:00")


def test_forecast_refuses_model_with_different_features(seeded, day_ahead_features) -> None:
    wrong = fake_model([*day_ahead_features, "load_lag_1"])
    with pytest.raises(ValueError, match="train/serve feature mismatch"):
        make_day_ahead_forecast(seeded, wrong)
    reordered = fake_model(list(reversed(day_ahead_features)))
    with pytest.raises(ValueError, match="train/serve feature mismatch"):
        make_day_ahead_forecast(seeded, reordered)


def test_store_forecast_is_idempotent_and_versioned(seeded, day_ahead_features) -> None:
    model = fake_model(day_ahead_features)
    r1 = make_day_ahead_forecast(seeded, model, weather_client=FakeWeatherClient(LAST_ACTUAL))
    assert store_forecast(seeded, r1) == 24
    assert store_forecast(seeded, r1) == 24
    assert db.count_rows(seeded, db.LoadForecastModel) == 24

    # Re-issue with the same version -> rows updated in place (new issued_at).
    r2 = make_day_ahead_forecast(seeded, model, weather_client=FakeWeatherClient(LAST_ACTUAL))
    r2.issued_at = r1.issued_at + pd.Timedelta(minutes=5)
    store_forecast(seeded, r2)
    assert db.count_rows(seeded, db.LoadForecastModel) == 24
    latest = db.read_latest_model_forecast(seeded)
    assert len(latest) == 24 and (latest["issued_at"] == r2.issued_at).all()
    assert (latest["model_name"] == "germany-load-day-ahead").all()

    # A different model version keeps its own rows.
    v8 = fake_model(day_ahead_features)
    v8.version = "8"
    store_forecast(
        seeded, make_day_ahead_forecast(seeded, v8, weather_client=FakeWeatherClient(LAST_ACTUAL))
    )
    assert db.count_rows(seeded, db.LoadForecastModel) == 48
    assert db.count_rows(seeded, db.LoadForecastModel, model_version="8") == 24


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
    assert "germany-load-day-ahead v7" in capsys.readouterr().out

    assert batch_forecast.main(["--database-url", url, "--hours", "12"]) == 0
    assert db.count_rows(seeded, db.LoadForecastModel) == 12


# --- API ---
@pytest.fixture
def client(seeded, day_ahead_features):
    def loader() -> api.AppState:
        return api.AppState(
            engine=seeded,
            model=fake_model(day_ahead_features),
            weather_client=FakeWeatherClient(LAST_ACTUAL),
            started_at=datetime.now(UTC),
        )

    with TestClient(api.create_app(loader)) as c:
        yield c


def test_health(client, day_ahead_features) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "sqlite"
    assert body["model"] == {
        "name": "germany-load-day-ahead",
        "version": "7",
        "alias": "champion",
        "horizon": "day_ahead",
        "n_features": len(day_ahead_features),
        "features": day_ahead_features,
    }
    assert body["last_actual_utc"].startswith("2024-01-30T23:00:00")
    assert body["last_forecast_issued_utc"] is None


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


def test_forecast_latest_404_then_serves_batch_output(client, seeded, day_ahead_features) -> None:
    assert client.get("/forecast/latest").status_code == 404

    result = make_day_ahead_forecast(
        seeded, fake_model(day_ahead_features), weather_client=FakeWeatherClient(LAST_ACTUAL)
    )
    store_forecast(seeded, result)
    r = client.get("/forecast/latest")
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 24
    assert body["model"]["version"] == "7"
    assert pd.Timestamp(body["issued_at"]) == result.issued_at.floor("us")
    assert client.get("/health").json()["last_forecast_issued_utc"] is not None


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
    assert r.status_code == 503
    assert "no actual load" in r.json()["detail"]


def test_openapi_lists_the_endpoints() -> None:
    paths = api.create_app(lambda: None).openapi()["paths"]
    assert {"/health", "/forecast", "/forecast/latest"} <= set(paths)
