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

USER_AGENT = "germany-load-forecasting-mlops/0.1"
DEFAULT_TIMEOUT = 60
RETRYABLE_4XX = {408, 429}  # request timeout, rate limited: worth another try


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
    backoff: float = 3.0,
    timeout: float = DEFAULT_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET ``url`` and return the parsed JSON body, retrying with exponential backoff.

    Transient failures (5xx, 408/429, timeouts, connection errors, a non-JSON body) are
    retried after ``backoff``, ``backoff**2``, ... seconds (3 s, 9 s, 27 s by default: long
    enough to ride out a provider hiccup); any other 4xx is final. Raises
    ``RuntimeError`` when giving up.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as err:  # say what came back instead of "Expecting value"
                raise ValueError(f"non-JSON response ({_describe(resp)})") from err
        except requests.HTTPError as err:
            status = getattr(err.response, "status_code", None)
            if status is not None and status < 500 and status not in RETRYABLE_4XX:
                raise RuntimeError(f"{url} returned HTTP {status}") from err
            last_err = err
        except Exception as err:
            last_err = err
        if attempt == retries:
            break
        wait = backoff**attempt
        log.warning(
            "request failed (%s/%s) for %s: %s - retrying in %.1fs",
            attempt,
            retries,
            url,
            last_err,
            wait,
        )
        sleep(wait)
    raise RuntimeError(f"giving up on {url} after {retries} attempts: {last_err}")


def _describe(resp: Any) -> str:
    """Status, content type and the start of the body - enough to tell an outage page apart."""
    status = getattr(resp, "status_code", "?")
    headers = getattr(resp, "headers", None) or {}
    body = (getattr(resp, "text", "") or "").strip().replace("\n", " ")
    return f"HTTP {status}, {headers.get('Content-Type', 'no content type')}, body {body[:120]!r}"
