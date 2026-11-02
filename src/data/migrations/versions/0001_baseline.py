"""Baseline: the schema as ``Base.metadata.create_all`` built it before migrations existed.

Revision ID: 0001
Revises:
Create Date: 2026-11-02

A database created before migrations is recorded at this revision once with
``alembic stamp 0001`` (docs/operations.md); a new database can be built from it.

Production (Neon) also has something no migration created: its four time-series tables -
``load_actual``, ``load_forecast_official``, ``weather_hourly``, ``load_forecast_model`` -
are TimescaleDB hypertables, converted by an early setup before migrations existed (the
``timescaledb`` extension is still installed). Alembic cannot see that, except through the
index ``create_hypertable`` gave each table, ``<table>_timestamp_utc_idx`` on
``timestamp_utc DESC``: the baseline includes those four indexes so it describes
production. The code never creates hypertables; a database built from this revision has
plain tables with the same indexes.

Plain SQLAlchemy types, not the application's ``UTCDateTime``, so this file stays valid
whatever ``src/data/db.py`` becomes. ``DateTime(timezone=True)`` creates what
``UTCDateTime`` creates: ``TIMESTAMP WITH TIME ZONE`` on Postgres, ``DATETIME`` on SQLite.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _utc(name: str, *, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


# The tables production made hypertables; each got TimescaleDB's default time index.
TIME_SERIES_TABLES = (
    "load_actual",
    "load_forecast_official",
    "weather_hourly",
    "load_forecast_model",
)


def upgrade() -> None:
    op.create_table(
        "load_actual",
        _utc("timestamp_utc"),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("load_mw", sa.Float(), nullable=False),
        _utc("ingested_at"),
        sa.PrimaryKeyConstraint("timestamp_utc", "source"),
    )
    op.create_table(
        "load_forecast_official",
        _utc("timestamp_utc"),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("forecast_mw", sa.Float(), nullable=False),
        _utc("ingested_at"),
        sa.PrimaryKeyConstraint("timestamp_utc", "source"),
    )
    op.create_table(
        "weather_hourly",
        _utc("timestamp_utc"),
        sa.Column("city", sa.String(64), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("temperature_2m", sa.Float(), nullable=True),
        sa.Column("wind_speed_10m", sa.Float(), nullable=True),
        sa.Column("shortwave_radiation", sa.Float(), nullable=True),
        sa.Column("cloud_cover", sa.Float(), nullable=True),
        sa.Column("relative_humidity_2m", sa.Float(), nullable=True),
        _utc("ingested_at"),
        sa.PrimaryKeyConstraint("timestamp_utc", "city"),
    )
    op.create_table(
        "load_forecast_model",
        _utc("timestamp_utc"),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("forecast_mw", sa.Float(), nullable=False),
        _utc("issued_at"),
        _utc("ingested_at"),
        sa.PrimaryKeyConstraint("timestamp_utc", "model_version"),
    )
    op.create_table(
        "monitoring_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        _utc("created_at"),
        _utc("as_of"),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("triggered", sa.Boolean(), nullable=False),
        sa.Column("decision", sa.String(32), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("model_mape_7d", sa.Float(), nullable=True),
        sa.Column("official_mape_7d", sa.Float(), nullable=True),
        sa.Column("drift_share", sa.Float(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_artifacts",
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=True),
        sa.Column("horizon", sa.String(32), nullable=False),
        sa.Column("features", sa.Text(), nullable=False),
        _utc("train_start", nullable=True),
        _utc("train_end", nullable=True),
        sa.Column("metrics", sa.Text(), nullable=True),
        sa.Column("bundle", sa.LargeBinary(), nullable=False),
        sa.Column("bundle_sha256", sa.String(64), nullable=False),
        sa.Column("bundle_bytes", sa.Integer(), nullable=False),
        sa.Column("is_champion", sa.Boolean(), nullable=False),
        _utc("created_at"),
        _utc("promoted_at", nullable=True),
        sa.PrimaryKeyConstraint("name", "version"),
    )
    for table in TIME_SERIES_TABLES:
        op.create_index(f"{table}_timestamp_utc_idx", table, [sa.text("timestamp_utc DESC")])


def downgrade() -> None:
    # Below the baseline there is no schema: going there would drop every table and its data.
    raise RuntimeError(
        "0001 is the baseline and cannot be downgraded: that would drop every application "
        "table and all of its data."
    )
