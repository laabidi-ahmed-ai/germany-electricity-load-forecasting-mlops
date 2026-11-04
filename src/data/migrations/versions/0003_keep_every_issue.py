"""Keep every issued forecast: issue time in the key, official revisions, weather inputs.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03

Up to 0002, three histories could be overwritten (README §16.9):

* ``load_forecast_model`` was keyed ``(timestamp_utc, model_version)``, so a re-issue replaced
  the earlier issue of the same hours;
* ``load_forecast_official`` keeps one value per hour, so its revisions were invisible;
* the per-city weather a forecast used was only in ``weather_hourly``, where the archive
  overwrites forecast values.

Upgrade, in Alembic's transaction. On Postgres the two tables it reads are first locked
against writes; reads still work.

1. The guard, before any change: every stored forecast's ``(model_name, model_version)``
   must have its export in ``model_artifacts``, the source of the new ``horizon``. If one is
   missing, nothing changes. This matters on SQLite, where Alembic's DDL is not transactional.
2. ``load_forecast_model`` gains ``horizon`` (NOT NULL, backfilled from the exports) and the
   primary key ``(timestamp_utc, model_version, issued_at)`` under its usual name. On
   Postgres these are in-place ``ALTER``s, so the table, its rows and its grants stay; SQLite
   rebuilds the table.
3. New ``load_forecast_official_revision``, seeded with one row per hour of
   ``load_forecast_official`` (any row-count difference aborts), and new
   ``load_forecast_model_weather``, empty. The seed is a snapshot of the values held at the
   upgrade: its ``ingested_at`` is when each row was last written (every ingest rewrites the
   hours it re-reads), not when the value was first seen, and earlier revisions were never
   kept. The revision history is complete only from the upgrade onward.

The dashboard role's grants on the two new tables are a runbook step (docs/operations.md,
"Keep every issued forecast (0003)").

Downgrade to 0002 deletes data for good: both new tables - every official revision and
weather input recorded since the upgrade - and every forecast issue except the latest per
hour and version (what 0002's code kept). No later upgrade brings them back, so a downgrade
is for a code problem right after the deploy; later, fix forward.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

FORECASTS = "load_forecast_model"
PRIMARY_KEY = f"{FORECASTS}_pkey"
KEY_0002 = ("timestamp_utc", "model_version")
KEY_0003 = ("timestamp_utc", "model_version", "issued_at")
WEATHER_VARS = (
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
    "relative_humidity_2m",
)

# Stored forecasts whose model version has no export - and so no horizon to backfill.
MISSING_EXPORTS = sa.text(
    "SELECT DISTINCT f.model_name, f.model_version FROM load_forecast_model f"
    " WHERE NOT EXISTS (SELECT 1 FROM model_artifacts a"
    " WHERE a.name = f.model_name AND a.version = f.model_version)"
    " ORDER BY 1, 2"
)
BACKFILL_HORIZON = sa.text(
    "UPDATE load_forecast_model SET horizon = (SELECT a.horizon FROM model_artifacts a"
    " WHERE a.name = load_forecast_model.model_name"
    " AND a.version = load_forecast_model.model_version)"
)
HORIZONS = sa.text(
    "SELECT model_version, horizon, count(*) FROM load_forecast_model GROUP BY 1, 2 ORDER BY 1"
)
SEED_REVISIONS = sa.text(
    "INSERT INTO load_forecast_official_revision (timestamp_utc, source, forecast_mw, ingested_at)"
    " SELECT timestamp_utc, source, forecast_mw, ingested_at FROM load_forecast_official"
)
# Every issue that a later issue of the same hour and version replaced in 0002's table.
DELETE_REPLACED_ISSUES = sa.text(
    "DELETE FROM load_forecast_model WHERE EXISTS (SELECT 1 FROM load_forecast_model later"
    " WHERE later.timestamp_utc = load_forecast_model.timestamp_utc"
    " AND later.model_version = load_forecast_model.model_version"
    " AND later.issued_at > load_forecast_model.issued_at)"
)


def _utc(name: str, *, primary_key: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, primary_key=primary_key)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # A lock that cannot be had soon aborts the run instead of queueing every reader behind it.
        bind.execute(sa.text("SET LOCAL lock_timeout = '30s'"))
        bind.execute(sa.text(f"LOCK TABLE {FORECASTS}, load_forecast_official IN EXCLUSIVE MODE"))
    check_exports(bind)

    op.add_column(FORECASTS, sa.Column("horizon", sa.String(32), nullable=True))
    bind.execute(BACKFILL_HORIZON)
    unset = bind.execute(
        sa.text(f"SELECT count(*) FROM {FORECASTS} WHERE horizon IS NULL")
    ).scalar_one()
    if unset:
        raise RuntimeError(f"{FORECASTS}: {unset} rows got no horizon; aborted")
    with op.batch_alter_table(FORECASTS, copy_from=forecasts_before_upgrade()) as batch:
        batch.alter_column("horizon", existing_type=sa.String(32), nullable=False)
        batch.drop_constraint(PRIMARY_KEY, type_="primary")
        batch.create_primary_key(PRIMARY_KEY, list(KEY_0003))
    for version, horizon, rows in bind.execute(HORIZONS):
        log.info("%s: version %s -> horizon %s (%d rows)", FORECASTS, version, horizon, rows)

    op.create_table(
        "load_forecast_official_revision",
        _utc("timestamp_utc", primary_key=True),
        sa.Column("source", sa.String(16), nullable=False, primary_key=True),
        sa.Column("forecast_mw", sa.Float(), nullable=False),
        _utc("ingested_at", primary_key=True),
    )
    op.create_table(
        "load_forecast_model_weather",
        _utc("timestamp_utc", primary_key=True),
        sa.Column("city", sa.String(64), nullable=False, primary_key=True),
        _utc("issued_at", primary_key=True),
        sa.Column("source", sa.String(16), nullable=False),
        *(sa.Column(var, sa.Float(), nullable=True) for var in WEATHER_VARS),
        _utc("fetched_at"),
    )
    official = count_rows(bind, "load_forecast_official")
    inserted = bind.execute(SEED_REVISIONS).rowcount
    check_copied(official, inserted, count_rows(bind, "load_forecast_official_revision"))
    log.info("load_forecast_official_revision: seeded with %d rows", official)


def downgrade() -> None:
    bind = op.get_bind()
    op.drop_table("load_forecast_model_weather")
    op.drop_table("load_forecast_official_revision")
    deleted = bind.execute(DELETE_REPLACED_ISSUES).rowcount
    log.info("%s: deleted %d issues replaced by a later one", FORECASTS, deleted)
    # At 0003 the key is named on every backend: SQLite's rebuild above, or the model's name.
    with op.batch_alter_table(FORECASTS) as batch:
        batch.drop_constraint(PRIMARY_KEY, type_="primary")
        batch.create_primary_key(PRIMARY_KEY, list(KEY_0002))
        batch.drop_column("horizon")


def check_exports(conn: Connection) -> None:
    """The guard: refuse, before changing anything, if a horizon cannot be backfilled."""
    missing = [f"{name} v{version}" for name, version in conn.execute(MISSING_EXPORTS)]
    if missing:
        raise RuntimeError(
            f"{FORECASTS} has forecasts of model versions with no export in model_artifacts, "
            f"so their horizon is unknown: {missing}. Nothing changed."
        )


def forecasts_before_upgrade() -> sa.Table:
    """``load_forecast_model`` as it stands when the upgrade swaps its key: 0002's columns,
    the new nullable ``horizon``, and 0002's key under its Postgres name.

    Only SQLite's rebuild uses it (Postgres alters in place). SQLite's own reflection would
    find the key unnamed - ``create_all`` and 0001 did not name it - and Alembic can only
    drop a named key.
    """
    return sa.Table(
        FORECASTS,
        sa.MetaData(),
        _utc("timestamp_utc"),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(128), nullable=False),
        sa.Column("forecast_mw", sa.Float(), nullable=False),
        _utc("issued_at"),
        _utc("ingested_at"),
        sa.Column("horizon", sa.String(32), nullable=True),
        sa.PrimaryKeyConstraint(*KEY_0002, name=PRIMARY_KEY),
    )


def count_rows(conn: Connection, table: str) -> int:
    return int(conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one())


def check_copied(source: int, inserted: int, copied: int) -> None:
    if not source == inserted == copied:
        raise RuntimeError(
            f"load_forecast_official: {source} rows, {inserted} inserted, {copied} in "
            "load_forecast_official_revision - aborted"
        )
