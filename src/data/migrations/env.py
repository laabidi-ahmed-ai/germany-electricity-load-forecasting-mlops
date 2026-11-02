"""Alembic environment for the application tables (``src.data.db.Base``).

The URL is ``sqlalchemy.url`` when a caller sets it (the tests), else ``DATABASE_URL`` from
``config.settings`` - so ``$env:DATABASE_URL`` picks the target for one shell session.

Two settings keep this safe on a database MLflow shares (the cloud default, where
``MLFLOW_TRACKING_URI`` falls back to ``DATABASE_URL``):

* the version table is ``db.SCHEMA_VERSION_TABLE``: MLflow's own migrations keep their
  revision in the default ``alembic_version`` of the same database;
* only the application's tables are visible (``include_name``), so ``alembic check`` and
  autogenerate never compare, alter or propose to drop MLflow's tables.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool

from config.settings import get_settings
from src.data import db

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = db.Base.metadata


def include_name(name: str | None, type_: str, parent_names: dict[str, str | None]) -> bool:
    """Tables of other applications sharing the database (MLflow) do not exist for us."""
    return type_ != "table" or name in target_metadata.tables


OPTIONS = {
    "target_metadata": target_metadata,
    "version_table": db.SCHEMA_VERSION_TABLE,
    "include_name": include_name,
    "render_as_batch": True,  # SQLite has no ALTER for most changes; ignored on Postgres
    "compare_type": True,
}


def database_url() -> str:
    return db.normalize_database_url(
        config.get_main_option("sqlalchemy.url") or get_settings().database_url
    )


def run_migrations_offline() -> None:
    """``--sql``: print the SQL instead of running it."""
    context.configure(
        url=database_url(), literal_binds=True, dialect_opts={"paramstyle": "named"}, **OPTIONS
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = db.get_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, **OPTIONS)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
