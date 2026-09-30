"""Storage-layer tests on a temporary SQLite database.

Focus: idempotent upserts (no duplicates on re-run), UTC round-trips across DST,
archive-over-forecast priority for weather, and the readers.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import select

from src.data import db


def _rows(df: pd.DataFrame, source: str = "smard") -> pd.DataFrame:
    return df.assign(source=source)


def test_init_db_creates_tables(engine) -> None:
    from sqlalchemy import inspect

    names = set(inspect(engine).get_table_names())
    assert {"load_actual", "load_forecast_official", "weather_hourly"} <= names


def test_upsert_is_idempotent(engine, load_frame) -> None:
    n1 = db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame))
    n2 = db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame))
    assert n1 == n2 == len(load_frame)
    assert db.count_rows(engine, db.LoadActual) == len(load_frame)


def test_upsert_updates_existing_values(engine, load_frame) -> None:
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame))
    corrected = load_frame.copy()
    corrected["load_mw"] = corrected["load_mw"] + 1000
    db.upsert_dataframe(engine, db.LoadActual, _rows(corrected))

    out = db.read_load(engine, source="smard")
    assert len(out) == len(load_frame)
    assert out["load_mw"].iloc[0] == pytest.approx(load_frame["load_mw"].iloc[0] + 1000)


def test_sources_are_independent_keys(engine, load_frame) -> None:
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame, "smard"))
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame, "entsoe"))
    assert db.count_rows(engine, db.LoadActual) == 2 * len(load_frame)
    assert db.count_rows(engine, db.LoadActual, source="smard") == len(load_frame)


def test_timestamps_round_trip_as_utc_across_dst(engine, load_frame) -> None:
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame))
    out = db.read_load(engine, source="smard")

    assert str(out["timestamp_utc"].dt.tz) == "UTC"
    pd.testing.assert_series_equal(
        out["timestamp_utc"], load_frame["timestamp_utc"], check_names=False
    )
    # Uniform hourly UTC steps straight through the DST switch.
    assert set(out["timestamp_utc"].diff().dropna()) == {pd.Timedelta(hours=1)}


def test_naive_timestamps_are_rejected(engine) -> None:
    rows = [{"timestamp_utc": datetime(2021, 3, 1, 0, 0), "source": "smard", "load_mw": 50_000.0}]
    with pytest.raises(Exception, match="naive datetime rejected"):
        db.upsert_rows(engine, db.LoadActual, rows)


def test_non_utc_aware_timestamps_are_normalised(engine) -> None:
    berlin = pd.Timestamp("2021-07-01T02:00", tz="Europe/Berlin")  # CEST = UTC+2
    db.upsert_rows(
        engine, db.LoadActual, [{"timestamp_utc": berlin, "source": "smard", "load_mw": 1.0}]
    )
    out = db.read_load(engine)
    assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2021-07-01T00:00Z")


def test_latest_timestamp_and_empty_table(engine, load_frame) -> None:
    assert db.latest_timestamp(engine, db.LoadActual) is None
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame))
    latest = db.latest_timestamp(engine, db.LoadActual, source="smard")
    assert latest == load_frame["timestamp_utc"].max()
    assert str(latest.tz) == "UTC"
    assert db.latest_timestamp(engine, db.LoadActual, source="entsoe") is None


def test_read_table_window(engine, load_frame) -> None:
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame))
    start = pd.Timestamp("2021-03-28T00:00Z")
    end = pd.Timestamp("2021-03-28T23:00Z")
    out = db.read_load(engine, start=start, end=end)
    assert len(out) == 24
    assert out["timestamp_utc"].min() == start
    assert out["timestamp_utc"].max() == end


def test_read_empty_returns_typed_columns(engine) -> None:
    out = db.read_load(engine)
    assert out.empty
    assert list(out.columns) == ["timestamp_utc", "load_mw"]


def test_ingested_at_is_set_and_utc(engine, load_frame) -> None:
    before = datetime.now(UTC).replace(microsecond=0)
    db.upsert_dataframe(engine, db.LoadActual, _rows(load_frame.head(3)))
    with engine.connect() as conn:
        stamps = conn.execute(select(db.LoadActual.ingested_at)).scalars().all()
    assert all(s.tzinfo is not None and s >= before for s in stamps)


def test_nan_becomes_null_for_optional_weather_columns(engine) -> None:
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(["2021-03-01T00:00Z"]),
            "city": ["Berlin"],
            "source": ["archive"],
            "temperature_2m": [float("nan")],
            "wind_speed_10m": [3.0],
        }
    )
    db.upsert_dataframe(engine, db.WeatherHourly, df)
    out = db.read_table(engine, db.WeatherHourly)
    assert pd.isna(out["temperature_2m"].iloc[0])
    assert out["wind_speed_10m"].iloc[0] == 3.0


def test_weather_archive_wins_over_forecast(engine) -> None:
    ts = pd.to_datetime(["2021-03-01T00:00Z"])
    base = {"timestamp_utc": ts, "city": ["Berlin"]}
    forecast = pd.DataFrame({**base, "source": ["forecast"], "temperature_2m": [1.0]})
    archive = pd.DataFrame({**base, "source": ["archive"], "temperature_2m": [2.0]})
    newer_forecast = pd.DataFrame({**base, "source": ["forecast"], "temperature_2m": [3.0]})

    # forecast -> archive: archive overwrites
    db.upsert_dataframe(engine, db.WeatherHourly, forecast, update_where=db.weather_update_where)
    db.upsert_dataframe(engine, db.WeatherHourly, archive, update_where=db.weather_update_where)
    out = db.read_table(engine, db.WeatherHourly)
    assert out["temperature_2m"].iloc[0] == 2.0 and out["source"].iloc[0] == "archive"

    # archive -> forecast: archive is kept
    db.upsert_dataframe(
        engine, db.WeatherHourly, newer_forecast, update_where=db.weather_update_where
    )
    out = db.read_table(engine, db.WeatherHourly)
    assert len(out) == 1
    assert out["temperature_2m"].iloc[0] == 2.0 and out["source"].iloc[0] == "archive"


def test_forecast_refreshes_forecast(engine) -> None:
    ts = pd.to_datetime(["2021-03-01T00:00Z"])
    base = {"timestamp_utc": ts, "city": ["Berlin"], "source": ["forecast"]}
    db.upsert_dataframe(
        engine,
        db.WeatherHourly,
        pd.DataFrame({**base, "temperature_2m": [1.0]}),
        update_where=db.weather_update_where,
    )
    db.upsert_dataframe(
        engine,
        db.WeatherHourly,
        pd.DataFrame({**base, "temperature_2m": [5.0]}),
        update_where=db.weather_update_where,
    )
    out = db.read_table(engine, db.WeatherHourly)
    assert len(out) == 1 and out["temperature_2m"].iloc[0] == 5.0


def test_large_batches_are_chunked(engine) -> None:
    ts = pd.date_range("2021-03-01", periods=1_500, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": ts, "load_mw": 55_000.0, "source": "smard"})
    assert db.upsert_dataframe(engine, db.LoadActual, df) == 1_500
    assert db.count_rows(engine, db.LoadActual) == 1_500


def test_get_engine_creates_sqlite_parent_dir(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "x.db"
    eng = db.get_engine(f"sqlite:///{path.as_posix()}")
    db.init_db(eng)
    assert path.exists()


def test_postgres_urls_are_routed_to_psycopg3() -> None:
    assert db.normalize_database_url("postgresql://u:p@h:5432/d?sslmode=require") == (
        "postgresql+psycopg://u:p@h:5432/d?sslmode=require"
    )
    assert db.normalize_database_url("postgres://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert (
        db.normalize_database_url("postgresql+psycopg://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    )
    assert db.normalize_database_url("sqlite:///x.db") == "sqlite:///x.db"
    eng = db.get_engine("postgresql://u:p@localhost:1/d")  # lazy: no connection made
    assert eng.url.drivername == "postgresql+psycopg" and eng.pool._pre_ping is True


def test_model_artifacts_store_champion_pointer_and_prune(engine) -> None:
    for v in ("1", "2", "3"):
        db.store_model_artifact(
            engine,
            name="m",
            version=v,
            run_id=f"run{v}",
            horizon="day_ahead",
            features='["a","b"]',
            bundle=b"zip" + v.encode(),
            bundle_sha256="x" * 64,
            bundle_bytes=4,
        )
    assert db.get_model_artifact(engine, "m", champion=True) is None
    db.set_champion(engine, "m", "2")
    champ = db.get_model_artifact(engine, "m", champion=True)
    assert (
        champ["version"] == "2" and champ["bundle"] == b"zip2" and champ["promoted_at"] is not None
    )
    db.set_champion(engine, "m", "3")  # exactly one champion at a time
    listing = db.list_model_artifacts(engine, "m")
    assert listing["is_champion"].sum() == 1 and "bundle" not in listing.columns
    assert db.get_model_artifact(engine, "m", champion=True)["version"] == "3"
    with pytest.raises(LookupError, match="not exported"):
        db.set_champion(engine, "m", "99")
    assert db.get_model_artifact(engine, "m", champion=True)["version"] == "3"  # untouched
    # re-export of an existing version replaces it
    db.store_model_artifact(
        engine,
        name="m",
        version="1",
        horizon="day_ahead",
        features="[]",
        bundle=b"new",
        bundle_sha256="y" * 64,
        bundle_bytes=3,
    )
    assert db.get_model_artifact(engine, "m", "1")["bundle"] == b"new"
    assert db.count_rows(engine, db.ModelArtifact) == 3
    # prune keeps the champion + the newest ones
    assert db.prune_model_artifacts(engine, "m", keep=2) == 1
    remaining = set(db.list_model_artifacts(engine, "m")["version"])
    assert "3" in remaining and len(remaining) == 2
    assert db.prune_model_artifacts(engine, "m", keep=5) == 0
