"""Small HTTP helper shared by the keyless JSON clients (SMARD, Open-Meteo).

Kept deliberately tiny and injectable: every client accepts a ``requests.Session``
(or any object with a compatible ``.get()``), so unit tests can swap in a fake
session and never touch the network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

log = logging.getLogger(__name__)

USER_AGENT = "germany-load-forecasting-mlops/0.1 (student project)"
DEFAULT_TIMEOUT = 60


def make_session() -> requests.Session:
    """Create a ``requests.Session`` carrying the project's User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    retries: int = 4,
    backoff: float = 1.5,
    timeout: float = DEFAULT_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET ``url`` and return the parsed JSON body, retrying with exponential backoff.

    Raises ``RuntimeError`` after ``retries`` failed attempts.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as err:  # retry on anything transient
            last_err = err
            if attempt == retries:
                break
            wait = backoff**attempt
            log.warning(
                "request failed (%s/%s) for %s: %s - retrying in %.1fs",
                attempt,
                retries,
                url,
                err,
                wait,
            )
            sleep(wait)
    raise RuntimeError(f"giving up on {url} after {retries} attempts: {last_err}")
