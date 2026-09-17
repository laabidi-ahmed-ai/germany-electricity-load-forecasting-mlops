"""Logging setup shared by every CLI entry point."""

from __future__ import annotations

import logging

from config.settings import get_settings


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=level or get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
