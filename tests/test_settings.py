"""Tests for config/settings.py.

These build ``Settings`` with ``_env_file=None`` so a developer's real ``.env``
never leaks into the test run, and clear the relevant env vars explicitly.
"""

from datetime import date

import pytest

from config.settings import Settings, get_settings

_ENV_KEYS = [
    "ENTSOE_API_TOKEN",
    "DATABASE_URL",
    "OPEN_METEO_BASE_URL",
    "MLFLOW_TRACKING_URI",
    "BIDDING_ZONE",
    "SMARD_REGION",
    "DATA_START_DATE",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


def test_runs_without_any_secrets() -> None:
    """The whole system must boot without the ENTSO-E token (CLAUDE.md)."""
    s = Settings(_env_file=None)
    assert s.entsoe_api_token is None
    assert s.has_entsoe_token is False
    assert s.database_url.startswith("sqlite:///")
    assert s.is_postgres is False


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
    assert s.is_postgres is True
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
