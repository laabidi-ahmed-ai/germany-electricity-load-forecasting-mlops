"""Database layer: SQLAlchemy models, engine factory, idempotent upserts, readers.

* All timestamps are stored in UTC. ``UTCDateTime`` guarantees that whatever
  goes in is converted to UTC and whatever comes out is tz-aware UTC - on both
  Postgres (``timestamptz``) and SQLite (naive text, interpreted as UTC). Conversion
  to ``Europe/Berlin`` happens only in feature engineering (calendar features).
* Upserts are idempotent. Re-running any ingestion never creates duplicates:
  every table has a natural primary key and ``upsert_dataframe`` uses
  ``INSERT ... ON CONFLICT DO UPDATE`` on both dialects.
* Postgres is the production store (README §10); SQLite is the zero-config local
  fallback that the default ``DATABASE_URL`` points at.
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
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    delete,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Dialect, Engine
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
    """Hourly day-ahead forecast produced by *our* champion model (batch job output).

    Keyed by ``(timestamp_utc, model_version)`` so re-issuing a forecast for the
    same hours with the same model updates in place; a new model version keeps
    its own rows, which lets monitoring compare versions side by side.
    """

    __tablename__ = "load_forecast_model"

    timestamp_utc: Mapped[datetime] = mapped_column(UTCDateTime, primary_key=True)
    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    forecast_mw: Mapped[float] = mapped_column(Float, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class MonitoringEvent(Base):
    """One row per monitoring / retraining decision (the dashboard's event timeline).

    ``kind`` = ``check`` (performance + drift evaluation) | ``retrain`` (champion/
    challenger outcome). ``details`` holds the full JSON summary.
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


def init_db(engine: Engine) -> None:
    """Create all tables (no-op if they exist). Called once by each entry point."""
    Base.metadata.create_all(engine)


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
    now = datetime.now(UTC)
    if "ingested_at" in model.__table__.columns:
        payload = [{**row, "ingested_at": row.get("ingested_at", now)} for row in rows]
    else:
        payload = list(rows)
    dialect = engine.dialect.name
    with engine.begin() as conn:
        for i in range(0, len(payload), chunk_rows):
            chunk = payload[i : i + chunk_rows]
            conn.execute(_build_upsert(dialect, model, chunk, update_where))
    return len(payload)


def upsert_dataframe(
    engine: Engine,
    model: type[Base],
    df: pd.DataFrame,
    *,
    update_where: UpdateWhere | None = None,
) -> int:
    """Upsert a DataFrame whose columns match ``model``'s columns (NaN -> NULL)."""
    if df.empty:
        return 0
    table_cols = {c.name for c in model.__table__.columns}
    cols = [c for c in df.columns if c in table_cols]
    if "timestamp_utc" in cols:
        ts = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.assign(timestamp_utc=ts)
    records = df.loc[:, cols].astype(object).where(df.loc[:, cols].notna(), None)
    rows = [
        {k: (v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in rec.items()}
        for rec in records.to_dict(orient="records")
    ]
    return upsert_rows(engine, model, rows, update_where=update_where)


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
    return _timestamp_bound(engine, model, func.max, filters)


def earliest_timestamp(engine: Engine, model: type[Base], **filters: Any) -> pd.Timestamp | None:
    """Oldest ``timestamp_utc`` in the table (optionally filtered), UTC, or None."""
    return _timestamp_bound(engine, model, func.min, filters)


def _timestamp_bound(
    engine: Engine, model: type[Base], agg: Any, filters: dict[str, Any]
) -> pd.Timestamp | None:
    stmt = select(agg(model.timestamp_utc))
    for col, val in filters.items():
        stmt = stmt.where(getattr(model, col) == val)
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
    """Rows of the most recently *issued* model forecast (all hours of that issue)."""
    with engine.connect() as conn:
        latest = conn.execute(select(func.max(LoadForecastModel.issued_at))).scalar()
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
