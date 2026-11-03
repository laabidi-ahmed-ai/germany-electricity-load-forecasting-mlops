"""Database layer: SQLAlchemy models, engine factory, idempotent upserts, readers.

* All timestamps are stored in UTC. ``UTCDateTime`` guarantees that whatever
  goes in is converted to UTC and whatever comes out is tz-aware UTC - on both
  Postgres (``timestamptz``) and SQLite (naive text, interpreted as UTC). Conversion
  to ``Europe/Berlin`` happens only in feature engineering (calendar features).
* Upserts are idempotent. Re-running any ingestion never creates duplicates:
  every table has a natural primary key and ``upsert_dataframe`` uses
  ``INSERT ... ON CONFLICT DO UPDATE`` on both dialects.
* Histories are kept, never overwritten: every model forecast issue
  (``load_forecast_model`` is keyed by ``issued_at`` too), with the per-city weather it
  used (``load_forecast_model_weather``), and every value the official forecast has had
  (``load_forecast_official_revision``).
* Postgres is the production store (README §10); SQLite is the zero-config local
  fallback that the default ``DATABASE_URL`` points at.
* The schema is versioned with Alembic (``src/data/migrations``). ``init_db`` creates it
  on an empty database and otherwise only checks its version: jobs never migrate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    or_,
    select,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Connection, Dialect, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import ColumnElement, Executable
from sqlalchemy.types import TypeDecorator

from config.settings import get_settings

log = logging.getLogger(__name__)

UPSERT_CHUNK_ROWS = 500

DateLike = date | datetime | str | pd.Timestamp

# ``source`` value of the load tables: both the actual load and the official day-ahead
# forecast come from SMARD (the column stays in the key so another source can sit beside it).
SOURCE_SMARD = "smard"

# ``kind`` of a monitoring event: one per pass - a clear pass is a check, a pass that ran the
# champion/challenger (triggered or forced) is a retrain. Here so the dashboard can share them.
EVENT_CHECK = "check"
EVENT_RETRAIN = "retrain"


# --- UTC handling ---
def to_utc(value: DateLike) -> pd.Timestamp:
    """Coerce a date-like value to a tz-aware UTC ``pd.Timestamp``; naive input counts as UTC."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class UTCDateTime(TypeDecorator[datetime]):
    """Timestamp column that is always UTC, on every backend.

    * bind: any tz-aware datetime is converted to UTC; naive datetimes are
      rejected (a naive value is a latent DST bug, so we fail loudly).
    * result: values come back tz-aware in UTC.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.TIMESTAMP(timezone=True))
        return dialect.type_descriptor(DateTime(timezone=False))

    def process_bind_param(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if not isinstance(value, datetime):
            raise TypeError(f"expected datetime, got {type(value).__name__}")
        if value.tzinfo is None:
            raise ValueError("naive datetime rejected: timestamps must be tz-aware (UTC)")
        value = value.astimezone(UTC)
        if dialect.name == "postgresql":
            return value
        return value.replace(tzinfo=None)  # SQLite stores naive text = UTC by convention

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):  # SQLite may hand back ISO text
            value = datetime.fromisoformat(value)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


# --- Models ---
class Base(DeclarativeBase):
    pass


class LoadActual(Base):
    """Hourly actual total load (``source`` = ``smard``)."""

    __tablename__ = "load_actual"

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    load_mw: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class LoadForecastOfficial(Base):
    """Hourly official (grid-operator) day-ahead load forecast - the benchmark."""

    __tablename__ = "load_forecast_official"

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    forecast_mw: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class LoadForecastOfficialRevision(Base):
    """Every value the official forecast has had: a row per hour each time it is new or changed.

    ``load_forecast_official`` keeps the latest value (the scores use it); this table keeps
    the history, written by ``store_official_forecast``. ``ingested_at`` is the run that
    first saw the value - except for the rows migration 0003 seeded: a snapshot of the values
    held at the upgrade, with ``ingested_at`` = when each row was last written. The history
    is complete only from that upgrade onward.
    """

    __tablename__ = "load_forecast_official_revision"

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    source: Mapped[str] = mapped_column(String(16), primary_key=True)
    forecast_mw: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)


class WeatherHourly(Base):
    """Hourly weather per city. ``source`` = ``archive`` (reanalysis) | ``forecast``."""

    __tablename__ = "weather_hourly"

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    city: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    temperature_2m: Mapped[float | None] = mapped_column(Float)
    wind_speed_10m: Mapped[float | None] = mapped_column(Float)
    shortwave_radiation: Mapped[float | None] = mapped_column(Float)
    cloud_cover: Mapped[float | None] = mapped_column(Float)
    relative_humidity_2m: Mapped[float | None] = mapped_column(Float)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class LoadForecastModel(Base):
    """Hourly forecast produced by *our* champion model (batch job output), one row per issue.

    Keyed by ``(timestamp_utc, model_version, issued_at)``: a re-issue for the same hours
    adds rows instead of overwriting, so every forecast can be scored as it was issued.
    ``horizon`` is the model's (``src.features.horizons``) and sets each issue's deadline;
    scoring takes the latest issue made by it (``src.monitoring.metrics``).
    """

    __tablename__ = "load_forecast_model"
    # Named as Postgres names it, on SQLite too, so a migration can replace it by name.
    __table_args__ = (
        PrimaryKeyConstraint(
            "timestamp_utc", "model_version", "issued_at", name="load_forecast_model_pkey"
        ),
    )

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime)
    model_version: Mapped[str] = mapped_column(String(64))
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_mw: Mapped[float] = mapped_column(Float, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class LoadForecastModelWeather(Base):
    """The per-city weather each stored model forecast used, as it was read at issue time.

    ``weather_hourly`` keeps one row per hour and city, and the archive overwrites forecast
    values there; this table keeps the inputs of every issue (``issued_at`` links it to
    ``load_forecast_model``). ``fetched_at`` is when that weather row had been ingested.
    """

    __tablename__ = "load_forecast_model_weather"

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    city: Mapped[str] = mapped_column(String(64), primary_key=True)
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    temperature_2m: Mapped[float | None] = mapped_column(Float)
    wind_speed_10m: Mapped[float | None] = mapped_column(Float)
    shortwave_radiation: Mapped[float | None] = mapped_column(Float)
    cloud_cover: Mapped[float | None] = mapped_column(Float)
    relative_humidity_2m: Mapped[float | None] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class MonitoringEvent(Base):
    """One row per monitoring / retraining decision (the dashboard's event timeline).

    ``kind`` = ``EVENT_CHECK`` (performance + drift evaluation) | ``EVENT_RETRAIN`` (the same
    plus the champion/challenger outcome). ``details`` holds the full JSON summary.
    """

    __tablename__ = "monitoring_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    as_of: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    triggered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision: Mapped[str | None] = mapped_column(String(32))
    reason: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String(64))
    model_mape_7d: Mapped[float | None] = mapped_column(Float)
    official_mape_7d: Mapped[float | None] = mapped_column(Float)
    drift_share: Mapped[float | None] = mapped_column(Float)
    details: Mapped[str | None] = mapped_column(Text)


class ModelArtifact(Base):
    """A model version exported for serving: zipped MLflow pyfunc bundle + metadata.

    Written by ``registry.promote``; exactly one row per ``name`` is ``is_champion``.
    """

    __tablename__ = "model_artifacts"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64))
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    features: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list, in model order
    train_start: Mapped[datetime | None] = mapped_column(UTCDateTime)
    train_end: Mapped[datetime | None] = mapped_column(UTCDateTime)
    metrics: Mapped[str | None] = mapped_column(Text)  # JSON: CV metrics of the training run
    bundle: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bundle_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    is_champion: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    promoted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


# --- Engine / schema ---
def get_engine(database_url: str | None = None, **kwargs: Any) -> Engine:
    """Create an engine for ``database_url`` (default: ``settings.database_url``).

    For SQLite the parent directory is created so a fresh clone "just works".
    """
    url = normalize_database_url(database_url or get_settings().database_url)
    if url.startswith("sqlite:///") and not url.endswith(":memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    elif url.startswith("postgresql"):
        # Cloud Postgres (Neon / Supabase) closes idle connections; re-check before use.
        kwargs.setdefault("pool_pre_ping", True)
    return create_engine(url, future=True, **kwargs)


def normalize_database_url(url: str) -> str:
    """Route plain ``postgresql://`` / ``postgres://`` URLs (as issued by Neon, Supabase,
    Heroku…) to the psycopg 3 driver this project ships; leave everything else alone."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


# Schema version (Alembic, ``src/data/migrations``). Not the default ``alembic_version``:
# MLflow's own migrations keep their revision there when it tracks into the same database.
SCHEMA_VERSION_TABLE = "alembic_version_app"
# Every revision, oldest first; tests/test_migrations.py checks it against the scripts.
SCHEMA_REVISIONS: tuple[str, ...] = ("0001", "0002", "0003")
SCHEMA_REVISION = SCHEMA_REVISIONS[-1]

# The version table exactly as Alembic creates it, so ``init_db`` can record the revision of
# a new database without importing Alembic (the ingest job and the dashboard run without it).
_schema_version = Table(
    SCHEMA_VERSION_TABLE,
    MetaData(),
    Column("version_num", String(32), nullable=False),
    PrimaryKeyConstraint("version_num", name=f"{SCHEMA_VERSION_TABLE}_pkc"),
)


class SchemaError(RuntimeError):
    """The database schema is not at the revision this code needs."""


def init_db(engine: Engine) -> None:
    """Create the schema on an empty database; otherwise check its version - never migrate.

    Called once by each entry point. A database with none of the application's tables (a
    fresh local or test database, the first cloud backfill) gets them, recorded at
    ``SCHEMA_REVISION``. Any other database must already be at ``SCHEMA_REVISION``, or
    ``SchemaError`` names the Alembic command to run. Tables of other applications in the
    same database (MLflow) are ignored.
    """
    tables = set(inspect(engine).get_table_names())
    versions = _schema_versions(engine) if SCHEMA_VERSION_TABLE in tables else []
    if not versions and not tables & set(Base.metadata.tables):
        with engine.begin() as conn:
            Base.metadata.create_all(conn)
            _schema_version.create(conn, checkfirst=True)
            conn.execute(insert(_schema_version).values(version_num=SCHEMA_REVISION))
        return
    _check_schema_version(versions)


def _schema_versions(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        return list(conn.execute(select(_schema_version.c.version_num)).scalars())


def _check_schema_version(versions: list[str]) -> None:
    see = "(docs/operations.md, 'Schema migrations')"
    if not versions:
        raise SchemaError(
            "the database has the application's tables but no schema version - it predates "
            f"migrations. Record it once with `alembic stamp {SCHEMA_REVISIONS[0]}` (the "
            f"baseline), then run `alembic upgrade head` {see}"
        )
    if len(versions) > 1:
        raise SchemaError(
            f"the database has several schema versions {sorted(versions)}; this code expects "
            f"exactly one, {SCHEMA_REVISION} {see}"
        )
    current = versions[0]
    if current == SCHEMA_REVISION:
        return
    if current in SCHEMA_REVISIONS:
        raise SchemaError(
            f"the database schema is at revision {current}, this code needs "
            f"{SCHEMA_REVISION} - run `alembic upgrade head` {see}"
        )
    raise SchemaError(
        f"the database schema is at revision {current}, which this code does not know (it "
        f"knows up to {SCHEMA_REVISION}) - most likely newer code migrated it. Deploy that "
        f"code, or downgrade the database with it (`alembic downgrade {SCHEMA_REVISION}`) {see}"
    )


# --- Upsert ---
UpdateWhere = Callable[[Any, Any], ColumnElement[bool]]


def _build_upsert(
    dialect_name: str,
    model: type[Base],
    rows: Sequence[dict[str, Any]],
    where: UpdateWhere | None,
) -> Executable:
    table = model.__table__
    pk_cols = [c.name for c in table.primary_key.columns]
    update_cols = [c.name for c in table.columns if c.name not in pk_cols]

    if dialect_name == "postgresql":
        stmt = postgresql.insert(table).values(list(rows))
    elif dialect_name == "sqlite":
        stmt = sqlite.insert(table).values(list(rows))
    else:  # pragma: no cover
        raise NotImplementedError(f"upsert not implemented for dialect {dialect_name!r}")

    set_ = {col: getattr(stmt.excluded, col) for col in update_cols}
    return stmt.on_conflict_do_update(
        index_elements=pk_cols,
        set_=set_,
        where=where(table.c, stmt.excluded) if where is not None else None,
    )


def upsert_rows(
    engine: Engine,
    model: type[Base],
    rows: Sequence[dict[str, Any]],
    *,
    update_where: UpdateWhere | None = None,
    chunk_rows: int = UPSERT_CHUNK_ROWS,
) -> int:
    """Insert-or-update ``rows`` into ``model``'s table by primary key.

    ``update_where(existing_cols, excluded)`` may return a condition restricting
    *when* an existing row is overwritten (e.g. "only if the incoming row is
    archive data"). Returns the number of rows sent.
    """
    if not rows:
        return 0
    payload = _stamped(model, rows)
    with engine.begin() as conn:
        _upsert(conn, model, payload, update_where=update_where, chunk_rows=chunk_rows)
    return len(payload)


def _stamped(model: type[Base], rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """``rows`` with ``ingested_at`` = now where ``model`` has that column and a row lacks it."""
    if "ingested_at" not in model.__table__.columns:
        return list(rows)
    now = datetime.now(UTC)
    return [{**row, "ingested_at": row.get("ingested_at", now)} for row in rows]


def _upsert(
    conn: Connection,
    model: type[Base],
    rows: Sequence[dict[str, Any]],
    *,
    update_where: UpdateWhere | None = None,
    chunk_rows: int = UPSERT_CHUNK_ROWS,
) -> None:
    """Upsert ``rows`` in the transaction ``conn`` is in, ``chunk_rows`` at a time."""
    for i in range(0, len(rows), chunk_rows):
        chunk = rows[i : i + chunk_rows]
        conn.execute(_build_upsert(conn.dialect.name, model, chunk, update_where))


def frame_rows(model: type[Base], df: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as rows of ``model``: its columns only, UTC datetimes, NaN -> NULL."""
    if df.empty:
        return []
    table_cols = {c.name for c in model.__table__.columns}
    cols = [c for c in df.columns if c in table_cols]
    if "timestamp_utc" in cols:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.assign(timestamp_utc=ts)
    records = df.loc[:, cols].astype(object).where(df.loc[:, cols].notna(), None)
    return [
        {k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in rec.items()}
        for rec in records.to_dict(orient="records")
    ]


def upsert_dataframe(
    engine: Engine,
    model: type[Base],
    df: pd.DataFrame,
    *,
    update_where: UpdateWhere | None = None,
) -> int:
    """Upsert a DataFrame whose columns match ``model``'s columns (NaN -> NULL)."""
    return upsert_rows(engine, model, frame_rows(model, df), update_where=update_where)


def store_issue(engine: Engine, forecast: pd.DataFrame, weather: pd.DataFrame) -> int:
    """Store one model forecast issue and the per-city weather it used, in one transaction.

    ``forecast`` rows go to ``load_forecast_model``, ``weather`` rows to
    ``load_forecast_model_weather``. Both carry the issue's ``issued_at``, which is in both
    keys: storing the same issue again adds no rows, a re-issue adds its own. Returns the
    number of forecast rows.
    """
    rows = _stamped(LoadForecastModel, frame_rows(LoadForecastModel, forecast))
    inputs = frame_rows(LoadForecastModelWeather, weather)
    with engine.begin() as conn:
        _upsert(conn, LoadForecastModel, rows)
        _upsert(conn, LoadForecastModelWeather, inputs)
    return len(rows)


def store_official_forecast(engine: Engine, df: pd.DataFrame) -> tuple[int, int]:
    """Upsert official forecast rows and record a revision for every hour new or changed.

    ``df`` has ``timestamp_utc, source, forecast_mw``. One transaction: read the latest
    revision of each hour in ``df``'s span, upsert ``load_forecast_official`` as before, and
    append to ``load_forecast_official_revision`` - with this run's ``ingested_at`` - the
    hours whose value it has not recorded yet. Returns ``(rows, revisions)``.
    """
    rows = _stamped(LoadForecastOfficial, frame_rows(LoadForecastOfficial, df))
    if not rows:
        return 0, 0
    revision = LoadForecastOfficialRevision
    hours = [row["timestamp_utc"] for row in rows]
    with engine.begin() as conn:
        latest = {
            (to_utc(ts), source): value
            for ts, source, value in conn.execute(
                select(revision.timestamp_utc, revision.source, revision.forecast_mw)
                .where(revision.timestamp_utc.between(min(hours), max(hours)))
                .order_by(revision.ingested_at)  # later revisions overwrite earlier ones here
            )
        }
        changed = [
            {k: row[k] for k in ("timestamp_utc", "source", "forecast_mw", "ingested_at")}
            for row in rows
            if latest.get((to_utc(row["timestamp_utc"]), row["source"])) != row["forecast_mw"]
        ]
        _upsert(conn, LoadForecastOfficial, rows)
        _upsert(conn, revision, changed)
    return len(rows), len(changed)


def weather_update_where(existing: Any, excluded: Any) -> ColumnElement[bool]:
    """Archive rows always win; forecast rows only fill in / refresh forecast rows."""
    return or_(excluded.source == "archive", existing.source == "forecast")


# --- Readers ---
def _frame(rows: Sequence[Any], columns: Sequence[Any]) -> pd.DataFrame:
    """Query rows -> DataFrame: UTC timestamps tz-aware, booleans as bool (SQLite says 0/1).

    Typed by the table definition, so no reader keeps its own list of datetime columns.
    """
    names = [c.name for c in columns]
    if not rows:
        return pd.DataFrame(columns=names)
    df = pd.DataFrame(rows, columns=names)
    for c in columns:
        if isinstance(c.type, UTCDateTime):
            df[c.name] = pd.to_datetime(df[c.name], utc=True)
        elif isinstance(c.type, Boolean):
            df[c.name] = df[c.name].astype(bool)
    return df


def latest_timestamp(engine: Engine, model: type[Base], **filters: Any) -> pd.Timestamp | None:
    """Most recent ``timestamp_utc`` in the table (optionally filtered), UTC, or None."""
    return _timestamp_bound(engine, model.timestamp_utc, func.max, filters)


def earliest_timestamp(engine: Engine, model: type[Base], **filters: Any) -> pd.Timestamp | None:
    """Oldest ``timestamp_utc`` in the table (optionally filtered), UTC, or None."""
    return _timestamp_bound(engine, model.timestamp_utc, func.min, filters)


def latest_issued_at(engine: Engine) -> pd.Timestamp | None:
    """When the most recent model forecast was issued (any version), UTC, or None."""
    return _timestamp_bound(engine, LoadForecastModel.issued_at, func.max, {})


def latest_event_at(engine: Engine) -> pd.Timestamp | None:
    """When the most recent monitoring event was recorded, UTC, or None."""
    return _timestamp_bound(engine, MonitoringEvent.created_at, func.max, {})


def happened_today(ts: pd.Timestamp | None, now: pd.Timestamp | None = None) -> bool:
    """True if ``ts`` falls on the same UTC calendar day as ``now`` (default: the current time)."""
    if ts is None:
        return False
    now = pd.Timestamp.now(tz="UTC") if now is None else to_utc(now)
    return to_utc(ts).normalize() == now.normalize()


def _timestamp_bound(
    engine: Engine, column: Any, agg: Any, filters: dict[str, Any]
) -> pd.Timestamp | None:
    stmt = select(agg(column))
    for col, val in filters.items():
        stmt = stmt.where(getattr(column.class_, col) == val)
    with engine.connect() as conn:
        value = conn.execute(stmt).scalar()
    return None if value is None else to_utc(value)


def count_rows(engine: Engine, model: type[Base], **filters: Any) -> int:
    stmt = select(func.count()).select_from(model)
    for col, val in filters.items():
        stmt = stmt.where(getattr(model, col) == val)
    with engine.connect() as conn:
        return int(conn.execute(stmt).scalar_one())


def read_table(
    engine: Engine,
    model: type[Base],
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    **filters: Any,
) -> pd.DataFrame:
    """Read rows in ``[start, end]`` as a DataFrame with tz-aware UTC timestamps."""
    stmt = select(model.__table__)
    if start is not None:
        stmt = stmt.where(model.timestamp_utc >= pd.Timestamp(start).to_pydatetime())
    if end is not None:
        stmt = stmt.where(model.timestamp_utc <= pd.Timestamp(end).to_pydatetime())
    for col, val in filters.items():
        stmt = stmt.where(getattr(model, col) == val)
    stmt = stmt.order_by(model.timestamp_utc)
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return _frame(rows, model.__table__.columns)


def read_load(
    engine: Engine, *, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Hourly actual load ``[timestamp_utc, load_mw]``."""
    df = read_table(engine, LoadActual, start=start, end=end, source=SOURCE_SMARD)
    return df.loc[:, ["timestamp_utc", "load_mw"]].reset_index(drop=True)


def read_latest_model_forecast(engine: Engine) -> pd.DataFrame:
    """Rows of the most recently *issued* model forecast (all hours of that issue), on time
    or not - what was last published; scoring picks on-time issues itself."""
    latest = latest_issued_at(engine)
    if latest is None:
        return _frame([], LoadForecastModel.__table__.columns)
    return read_table(engine, LoadForecastModel, issued_at=latest)


def insert_monitoring_event(engine: Engine, **fields: Any) -> int:
    """Append one monitoring event; returns its id."""
    fields.setdefault("created_at", datetime.now(UTC))
    with engine.begin() as conn:
        result = conn.execute(insert(MonitoringEvent).values(**fields))
        return int(result.inserted_primary_key[0])


def read_monitoring_events(engine: Engine, *, limit: int = 100) -> pd.DataFrame:
    stmt = (
        select(MonitoringEvent.__table__)
        .order_by(MonitoringEvent.created_at.desc(), MonitoringEvent.id.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return _frame(rows, MonitoringEvent.__table__.columns)


# --- Model artifacts (champion persistence) ---
def store_model_artifact(engine: Engine, **fields: Any) -> None:
    """Insert-or-replace one exported model version (keyed by name + version)."""
    fields.setdefault("created_at", datetime.now(UTC))
    existing = get_model_artifact(engine, fields["name"], str(fields["version"]))
    if existing is not None:  # re-export keeps the champion pointer
        fields.setdefault("is_champion", existing["is_champion"])
        fields.setdefault("promoted_at", existing["promoted_at"])
    upsert_rows(engine, ModelArtifact, [fields])


def set_champion(engine: Engine, name: str, version: str) -> None:
    """Point the champion flag of ``name`` at ``version`` (exactly one row flagged)."""
    with engine.begin() as conn:  # one transaction: an unknown version changes nothing
        result = conn.execute(
            update(ModelArtifact)
            .where(ModelArtifact.name == name, ModelArtifact.version == str(version))
            .values(is_champion=True, promoted_at=datetime.now(UTC))
        )
        if result.rowcount != 1:
            raise LookupError(f"model {name!r} version {version!r} is not exported")
        conn.execute(
            update(ModelArtifact)
            .where(ModelArtifact.name == name, ModelArtifact.version != str(version))
            .values(is_champion=False)
        )


def get_model_artifact(
    engine: Engine, name: str, version: str | None = None, *, champion: bool = False
) -> dict[str, Any] | None:
    """One exported version (by ``version`` or the champion) as a dict, or None."""
    stmt = select(ModelArtifact.__table__).where(ModelArtifact.name == name)
    if champion:
        stmt = stmt.where(ModelArtifact.is_champion.is_(True))
    elif version is not None:
        stmt = stmt.where(ModelArtifact.version == str(version))
    else:
        raise ValueError("give a version or champion=True")
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    if row is None:
        return None
    out = dict(row)
    out["is_champion"] = bool(out["is_champion"])
    return out


def list_model_artifacts(engine: Engine, name: str) -> pd.DataFrame:
    """All exported versions of ``name`` (newest first) without the bundle bytes."""
    cols = [c for c in ModelArtifact.__table__.columns if c.name != "bundle"]
    stmt = (
        select(*cols)
        .where(ModelArtifact.name == name)
        .order_by(ModelArtifact.created_at.desc(), ModelArtifact.version.desc())
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return _frame(rows, cols)


def prune_model_artifacts(engine: Engine, name: str, *, keep: int = 5) -> int:
    """Delete the oldest exported versions beyond ``keep``; the champion is never deleted."""
    df = list_model_artifacts(engine, name)
    if len(df) <= keep:
        return 0
    doomed = df[~df["is_champion"]].iloc[keep - 1 :] if keep > 0 else df[~df["is_champion"]]
    victims = [str(v) for v in doomed["version"]]
    if not victims:
        return 0
    with engine.begin() as conn:
        conn.execute(
            delete(ModelArtifact).where(
                ModelArtifact.name == name, ModelArtifact.version.in_(victims)
            )
        )
    return len(victims)
