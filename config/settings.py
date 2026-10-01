"""Typed application configuration (README §12).

All secrets and environment-specific values are read from environment variables
or a local ``.env`` file via ``pydantic-settings``. Nothing is hardcoded here.

Every field has a safe default so the project runs *without* any secret set: all data
sources are keyless, and the database falls back to a local SQLite file for development.

Usage::

    from config.settings import get_settings

    settings = get_settings()
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (this file lives in <root>/config/settings.py).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings, loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- external APIs ------------------------------------------------------
    open_meteo_base_url: str = Field(
        default="https://api.open-meteo.com/v1",
        description="Open-Meteo forecast API base URL (no key required).",
    )
    open_meteo_archive_url: str = Field(
        default="https://archive-api.open-meteo.com/v1/archive",
        description="Open-Meteo historical archive endpoint.",
    )
    smard_base_url: str = Field(
        default="https://www.smard.de/app/chart_data",
        description="SMARD (Bundesnetzagentur) chart-data endpoint (no key required).",
    )

    # --- storage ------------------------------------------------------------
    database_url: str = Field(
        default=f"sqlite:///{(PROJECT_ROOT / 'data' / 'load_forecasting.db').as_posix()}",
        description="SQLAlchemy URL. Postgres in the cloud; SQLite locally.",
    )
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns/mlflow.db",
        description=(
            "MLflow tracking URI. Default: local SQLite store under mlruns/ "
            "(relative paths are anchored at the project root)."
        ),
    )

    # --- target geography & time window ------------------------------------
    smard_region: str = Field(
        default="DE",
        description="SMARD region code for Germany.",
    )
    data_start_date: date = Field(
        default=date(2021, 3, 1),
        description="First day of usable data (post-COVID cutoff: pandemic demand patterns are unrepresentative).",
    )

    # --- misc ---------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # --- validators ---------------------------------------------------------
    @field_validator("smard_region")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached after first load)."""
    return Settings()
