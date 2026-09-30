"""Tests for config/settings.py (env isolation comes from the autouse fixture in conftest)."""

from datetime import date

import pytest

from config.settings import Settings, get_settings


def test_runs_without_any_secrets() -> None:
    """The whole system must boot without the ENTSO-E token (it arrives days after signup)."""
    s = Settings(_env_file=None)
    assert s.entsoe_api_token is None
    assert s.has_entsoe_token is False
    assert s.database_url.startswith("sqlite:///")


def test_target_geography_and_start_date_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.bidding_zone == "DE_LU"
    assert s.smard_region == "DE"
    assert s.data_start_date == date(2021, 3, 1)
    assert s.open_meteo_base_url == "https://api.open-meteo.com/v1"
    assert s.mlflow_tracking_uri == "sqlite:///mlruns/mlflow.db"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "abc-123")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")
    monkeypatch.setenv("BIDDING_ZONE", "de_lu")
    monkeypatch.setenv("DATA_START_DATE", "2022-01-01")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    s = Settings(_env_file=None)
    assert s.has_entsoe_token is True
    assert s.entsoe_api_token is not None
    assert s.entsoe_api_token.get_secret_value() == "abc-123"
    assert s.database_url.startswith("postgresql://")
    assert s.bidding_zone == "DE_LU"  # normalised to upper-case
    assert s.data_start_date == date(2022, 1, 1)
    assert s.log_level == "DEBUG"


def test_placeholder_token_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Copying .env.example verbatim must not look like a configured token."""
    monkeypatch.setenv("ENTSOE_API_TOKEN", "your_entsoe_security_token_here")
    assert Settings(_env_file=None).has_entsoe_token is False

    monkeypatch.setenv("ENTSOE_API_TOKEN", "   ")
    assert Settings(_env_file=None).has_entsoe_token is False


def test_secret_is_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTSOE_API_TOKEN", "super-secret")
    assert "super-secret" not in repr(Settings(_env_file=None))


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
