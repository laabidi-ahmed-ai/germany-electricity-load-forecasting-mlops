"""Job metrics pushed to Datadog over HTTP (optional; README §16.4, docs/observability.md).

GitHub runners are ephemeral, so there is no Datadog agent: each job buffers a few gauges
while it runs and POSTs them once when it ends (``/api/v2/series``). Monitors in Datadog
then alert on what a job *did* (a failed step, stale data, a rising error) and - through
"notify if data is missing" - on a job that never ran at all, which GitHub cannot report.

Without ``DD_API_KEY`` every call is a no-op, so local runs, tests and forks need nothing.
Sending is best-effort: a telemetry failure is logged and never fails the job.

Usage::

    with job_metrics("ingest") as m:
        m.gauge("ingest.rows", 24, source="smard")
        if something_failed:
            m.fail()
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import requests

from config.settings import get_settings

log = logging.getLogger(__name__)

PREFIX = "germany_load."
GAUGE = 3  # metric type code of the v2 series intake
TIMEOUT = 10


class JobMetrics:
    """Buffer of gauges for one job run, tagged ``job:<name>`` and ``runner:github|local``."""

    def __init__(
        self,
        job: str,
        *,
        api_key: str | None = None,
        site: str = "datadoghq.eu",
        session: Any = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.job = job
        self.api_key = api_key
        self.site = site
        self.session = session or requests
        self.clock = clock
        self.status = "ok"
        self._series: list[dict[str, Any]] = []

    @classmethod
    def from_settings(cls, job: str) -> JobMetrics:
        settings = get_settings()
        key = settings.dd_api_key.get_secret_value() if settings.dd_api_key else None
        return cls(job, api_key=key, site=settings.dd_site)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _tags(self, extra: dict[str, Any]) -> list[str]:
        runner = "github" if os.getenv("GITHUB_ACTIONS") == "true" else "local"
        return [f"job:{self.job}", f"runner:{runner}"] + [
            f"{k}:{v}" for k, v in sorted(extra.items())
        ]

    def gauge(self, name: str, value: float | None, **tags: Any) -> None:
        """Record ``germany_load.<name>``; ``None`` values (unjudged windows) are skipped."""
        if value is None:
            return
        self._series.append(
            {
                "metric": PREFIX + name,
                "type": GAUGE,
                "points": [{"timestamp": int(self.clock()), "value": float(value)}],
                "tags": self._tags(tags),
            }
        )

    def fail(self) -> None:
        """Mark the run as failed even though no exception escaped (e.g. one step failed)."""
        self.status = "failed"

    def flush(self) -> bool:
        """POST the buffered gauges; ``True`` if Datadog accepted them."""
        series, self._series = self._series, []
        if not self.enabled or not series:
            return False
        try:
            resp = self.session.post(
                f"https://api.{self.site}/api/v2/series",
                json={"series": series},
                headers={"DD-API-KEY": self.api_key, "Content-Type": "application/json"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as err:  # telemetry must never break the job
            log.warning("could not send %d metric(s) to Datadog: %s", len(series), err)
            return False
        log.info("sent %d metric(s) to Datadog", len(series))
        return True


@contextmanager
def job_metrics(job: str, metrics: JobMetrics | None = None) -> Iterator[JobMetrics]:
    """Time the block and always report ``job.run`` / ``job.failed`` / ``job.duration_seconds``."""
    m = metrics or JobMetrics.from_settings(job)
    start = time.monotonic()
    try:
        yield m
    except BaseException:
        m.fail()
        raise
    finally:
        m.gauge("job.run", 1, status=m.status)
        m.gauge("job.failed", int(m.status == "failed"))
        m.gauge("job.duration_seconds", time.monotonic() - start)
        m.flush()
