"""Convert production's leftover TimescaleDB hypertables back to plain tables.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03

``init_db`` made the four time-series tables hypertables (``create_hypertable``, when the
``timescaledb`` extension was available) until commit c7e7058 removed that setup without
converting them back. Production kept them, with ~900 chunks between them and the
``<table>_timestamp_utc_idx`` index on ``timestamp_utc DESC`` that ``create_hypertable``
gave each one - a duplicate of the primary key, which starts with ``timestamp_utc``.

Upgrade:

* Conversion, only on Postgres with TimescaleDB and only for the tables still listed in
  ``timescaledb_information.hypertables`` (so a re-run converts nothing twice). All of it
  runs in Alembic's single transaction: the tables are locked against writes (reads still
  allowed) in a fixed order, then each one is copied into a plain table with an explicit
  column list - any row-count difference aborts and rolls everything back - its grants are
  copied (the dashboard reads through its own role), the hypertable is dropped with its
  chunks, the copy takes its name and its ``<table>_pkey`` primary key, and is analyzed.
* The index step, on every backend: drop the four DESC indexes (a converted table no longer
  has its index; on SQLite or plain Postgres this is all the upgrade does).

Dropping every chunk in one transaction takes a lock per chunk and per chunk index. If that
exceeds the lock table ("out of shared memory", ``max_locks_per_transaction``), nothing has
changed; ``alembic -x per_table=true upgrade head`` then converts each table in its own
transaction before the index step. Runbook: docs/operations.md, "Convert the leftover
hypertables (0002)".

Downgrade recreates the four indexes, the schema of 0001 as Alembic sees it. It does not
recreate hypertables: no code needs them.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.engine import Connection

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

log = logging.getLogger("alembic.runtime.migration")

# Every column copied, by name, and the primary key restored, per table - in the order the
# tables are locked and converted.
TABLES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "load_actual": (
        ("timestamp_utc", "source", "load_mw", "ingested_at"),
        ("timestamp_utc", "source"),
    ),
    "load_forecast_official": (
        ("timestamp_utc", "source", "forecast_mw", "ingested_at"),
        ("timestamp_utc", "source"),
    ),
    "weather_hourly": (
        (
            "timestamp_utc",
            "city",
            "source",
            "temperature_2m",
            "wind_speed_10m",
            "shortwave_radiation",
            "cloud_cover",
            "relative_humidity_2m",
            "ingested_at",
        ),
        ("timestamp_utc", "city"),
    ),
    "load_forecast_model": (
        ("timestamp_utc", "model_version", "model_name", "forecast_mw", "issued_at", "ingested_at"),
        ("timestamp_utc", "model_version"),
    ),
}
TIME_SERIES_TABLES = tuple(TABLES)

# Privileges on a table held by roles other than its owner (grantee 0 is PUBLIC).
GRANTS = sa.text(
    "SELECT CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE quote_ident(pg_get_userbyid(a.grantee))"
    " END, a.privilege_type, a.is_grantable"
    " FROM pg_class c, aclexplode(c.relacl) a"
    " WHERE c.oid = CAST(:table AS regclass) AND a.grantee <> c.relowner"
)


def upgrade() -> None:
    bind = op.get_bind()
    if context.get_x_argument(as_dictionary=True).get("per_table") == "true":
        # The lock-limit fallback: each table in its own committed transaction, on
        # connections of its own; Alembic's transaction then runs the index step.
        with bind.engine.connect() as conn:
            tables = hypertables(conn)
        for table in tables:
            with bind.engine.begin() as conn:
                convert(conn, [table])
    else:
        convert(bind, hypertables(bind))
    for table in TIME_SERIES_TABLES:
        op.drop_index(f"{table}_timestamp_utc_idx", table_name=table, if_exists=True)


def downgrade() -> None:
    for table in TIME_SERIES_TABLES:
        op.create_index(f"{table}_timestamp_utc_idx", table, [sa.text("timestamp_utc DESC")])


def hypertables(conn: Connection) -> list[str]:
    """The application's tables that are still hypertables, in conversion order."""
    if conn.dialect.name != "postgresql":
        return []
    if conn.execute(sa.text("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")).first():
        listed = set(
            conn.execute(
                sa.text(
                    "SELECT hypertable_name FROM timescaledb_information.hypertables"
                    " WHERE hypertable_schema = current_schema()"
                )
            ).scalars()
        )
        return [table for table in TIME_SERIES_TABLES if table in listed]
    return []


def convert(conn: Connection, tables: list[str]) -> None:
    """Convert ``tables`` in the transaction ``conn`` is in; any failure rolls it all back."""
    if not tables:
        return
    # A lock that cannot be had soon aborts the run instead of queueing every reader behind it.
    conn.execute(sa.text("SET LOCAL lock_timeout = '30s'"))
    conn.execute(sa.text(f"LOCK TABLE {', '.join(tables)} IN EXCLUSIVE MODE"))
    for table in tables:
        rows = convert_table(conn, table)
        log.info("Converted %s to a plain table: %d rows", table, rows)
    locks = conn.execute(
        sa.text("SELECT count(*) FROM pg_locks WHERE pid = pg_backend_pid()")
    ).scalar_one()
    log.info("Locks held by the conversion before commit: %d", locks)


def convert_table(conn: Connection, table: str) -> int:
    """Swap hypertable ``table`` for a plain copy; returns the number of rows copied."""
    columns, key = TABLES[table]
    actual = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns"
            " WHERE table_schema = current_schema() AND table_name = :table"
        ),
        {"table": table},
    ).scalars()
    if set(actual) != set(columns):
        raise RuntimeError(f"{table} does not have the expected columns {columns}; nothing changed")
    owner = conn.execute(
        sa.text(
            "SELECT pg_get_userbyid(relowner) = current_user FROM pg_class"
            " WHERE oid = CAST(:table AS regclass)"
        ),
        {"table": table},
    ).scalar_one()
    if not owner:
        raise RuntimeError(f"{table} is owned by another role: connect as the owner role")

    plain = f"{table}__plain"
    cols = ", ".join(columns)
    conn.execute(sa.text(f"CREATE TABLE {plain} (LIKE {table} INCLUDING DEFAULTS)"))
    source = count_rows(conn, table)
    inserted = conn.execute(sa.text(f"INSERT INTO {plain} ({cols}) SELECT {cols} FROM {table}"))
    check_copied(table, source, inserted.rowcount, count_rows(conn, plain))

    grants = table_grants(conn, table)
    for grantee, privilege, grantable in grants:
        option = " WITH GRANT OPTION" if grantable else ""
        conn.execute(sa.text(f"GRANT {privilege} ON TABLE {plain} TO {grantee}{option}"))
    if table_grants(conn, plain) != grants:
        raise RuntimeError(f"{table}: the copy's grants differ from the original; nothing changed")

    conn.execute(sa.text(f"DROP TABLE {table}"))  # with its chunks; no CASCADE
    conn.execute(sa.text(f"ALTER TABLE {plain} RENAME TO {table}"))
    conn.execute(
        sa.text(f"ALTER TABLE {table} ADD CONSTRAINT {table}_pkey PRIMARY KEY ({', '.join(key)})")
    )
    conn.execute(sa.text(f"ANALYZE {table}"))
    return source


def count_rows(conn: Connection, table: str) -> int:
    return int(conn.execute(sa.text(f"SELECT count(*) FROM {table}")).scalar_one())


def check_copied(table: str, source: int, inserted: int, copied: int) -> None:
    if not source == inserted == copied:
        raise RuntimeError(
            f"{table}: {source} rows, {inserted} inserted, {copied} in the copy - aborted, "
            "nothing changed"
        )


def table_grants(conn: Connection, table: str) -> set[tuple[str, str, bool]]:
    return {tuple(row) for row in conn.execute(GRANTS, {"table": table})}
