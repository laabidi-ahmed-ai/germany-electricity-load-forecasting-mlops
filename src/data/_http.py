"""Small HTTP helper shared by the keyless JSON clients (SMARD, Open-Meteo).

Kept deliberately tiny and injectable: every client accepts a ``requests.Session``
(or any object with a compatible ``.get()``), so unit tests can swap in a fake
session and never touch the network.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

log = logging.getLogger(__name__)

USER_AGENT = "germany-load-forecasting-mlops/0.1"
DEFAULT_TIMEOUT = 60
RETRYABLE_4XX = {408, 429}  # request timeout, rate limited: worth another try

Timeout = float | tuple[float, float]  # seconds; a pair is (connect, read), as in requests


@dataclass(frozen=True)
class RetryPolicy:
    """How one call waits for an answer and how often it tries again.

    ``timeout`` goes to ``requests`` as is: a float caps the connect and the read wait each,
    a pair sets them apart. The read timeout bounds the gap between bytes, not the whole
    download. After failed attempt ``n`` the call sleeps ``backoff**n`` seconds, at most
    ``max_wait`` when set.
    """

    retries: int = 4  # attempts in total
    backoff: float = 3.0
    max_wait: float | None = None
    timeout: Timeout = DEFAULT_TIMEOUT

    def wait(self, attempt: int) -> float:
        """Seconds to sleep after failed attempt ``attempt`` (1-based)."""
        wait = self.backoff**attempt
        return wait if self.max_wait is None else min(wait, self.max_wait)

    def worst_case_seconds(self) -> float:
        """Every attempt stalls until it times out, then the call gives up.

        Assumes a stalled server sends nothing (what Open-Meteo does); a body trickling in
        slower than the read timeout could take longer.
        """
        connect, read = self.timeout if isinstance(self.timeout, tuple) else (self.timeout,) * 2
        return self.retries * (connect + read) + sum(map(self.wait, range(1, self.retries)))


DEFAULT_RETRY = RetryPolicy()


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
    policy: RetryPolicy = DEFAULT_RETRY,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET ``url`` and return the parsed JSON body, retrying with exponential backoff.

    Transient failures (5xx, 408/429, timeouts, connection errors, a non-JSON body) are
    retried after ``policy.wait(1)``, ``policy.wait(2)``, ... seconds (3 s, 9 s, 27 s by
    default: long enough to ride out a provider hiccup); any other 4xx is final. Raises
    ``RuntimeError`` when giving up.
    """
    retries = policy.retries
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=policy.timeout)
            resp.raise_for_status()
            try:
                body = resp.json()
            except ValueError as err:  # say what came back instead of "Expecting value"
                raise ValueError(f"non-JSON response ({_describe(resp)})") from err
            if attempt > 1:
                log.info("request succeeded on attempt %s/%s for %s", attempt, retries, url)
            return body
        except requests.HTTPError as err:
            status = getattr(err.response, "status_code", None)
            if status is not None and status < 500 and status not in RETRYABLE_4XX:
                raise RuntimeError(f"{url} returned HTTP {status}") from err
            last_err = err
        except Exception as err:
            last_err = err
        if attempt == retries:
            break
        wait = policy.wait(attempt)
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
