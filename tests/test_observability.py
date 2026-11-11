"""Job metrics -> Datadog (offline): payload shape, no-op without a key, never breaks a job."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from config.settings import Settings, get_settings
from src import observability
from src.data import db
from src.data.ingest import IngestResult
from src.data.ingest import report_metrics as ingest_metrics
from src.monitoring.retrain import report_metrics as monitor_metrics
from src.observability import JobMetrics, job_metrics
from src.serving.batch_forecast import report_metrics as forecast_metrics


class Recorder:
    """Stands in for ``requests``: records POSTs, optionally fails them."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        return SimpleNamespace(raise_for_status=lambda: None)

    def metrics(self) -> dict[str, list[dict[str, Any]]]:
        """metric name (without prefix) -> its series, across all POSTs."""
        out: dict[str, list[dict[str, Any]]] = {}
        for call in self.calls:
            for s in call["json"]["series"]:
                out.setdefault(s["metric"].removeprefix(observability.PREFIX), []).append(s)
        return out


def enabled(job: str = "test", **kwargs: Any) -> tuple[JobMetrics, Recorder]:
    rec = Recorder(**kwargs)
    return JobMetrics(job, api_key="k", site="datadoghq.eu", session=rec, clock=lambda: 1e9), rec


def value(series: dict, name: str, **tags: str) -> float:
    want = {f"{k}:{v}" for k, v in tags.items()}
    (match,) = [s for s in series[name] if want <= set(s["tags"])]
    return match["points"][0]["value"]


def test_without_a_key_nothing_is_sent() -> None:
    rec = Recorder()
    m = JobMetrics("ingest", session=rec)
    with job_metrics("ingest", m):
        m.gauge("ingest.rows", 5)
    assert not m.enabled and rec.calls == []


def test_payload_is_one_post_of_tagged_gauges(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    m, rec = enabled("ingest")
    m.gauge("ingest.rows", 24, step="smard")
    m.gauge("model.mape", None)  # unjudged window -> skipped, not sent as 0
    assert m.flush() is True

    (call,) = rec.calls
    assert call["url"] == "https://api.datadoghq.eu/api/v2/series"
    assert call["headers"]["DD-API-KEY"] == "k"
    (series,) = call["json"]["series"]
    assert series == {
        "metric": "germany_load.ingest.rows",
        "type": observability.GAUGE,
        "points": [{"timestamp": 1_000_000_000, "value": 24.0}],
        "tags": ["job:ingest", "runner:github", "step:smard"],
    }
    assert m.flush() is False  # buffer emptied


def test_job_metrics_reports_success_failure_and_duration() -> None:
    m, rec = enabled("forecast")
    with job_metrics("forecast", m):
        pass
    got = rec.metrics()
    assert value(got, "job.run", status="ok") == 1
    assert value(got, "job.failed") == 0
    assert value(got, "job.duration_seconds") >= 0

    m, rec = enabled("forecast")
    with pytest.raises(RuntimeError), job_metrics("forecast", m):
        raise RuntimeError("boom")
    got = rec.metrics()
    assert value(got, "job.run", status="failed") == 1 and value(got, "job.failed") == 1


def test_a_datadog_outage_never_fails_the_job(caplog) -> None:
    m, rec = enabled(error=ConnectionError("intake down"))
    with job_metrics("ingest", m):
        m.gauge("ingest.rows", 1)
    assert len(rec.calls) == 1
    assert "could not send" in caplog.text


def test_settings_read_the_key_as_a_secret_and_default_an_empty_site(monkeypatch) -> None:
    monkeypatch.setenv("DD_API_KEY", "abc123")
    monkeypatch.setenv("DD_SITE", "")  # an unset GitHub variable
    s = Settings(_env_file=None)
    assert s.dd_api_key.get_secret_value() == "abc123"
    assert "abc123" not in repr(s)
    assert s.dd_site == "datadoghq.eu"
    get_settings.cache_clear()
    assert JobMetrics.from_settings("ingest").enabled


def test_ingest_metrics_rows_failed_steps_and_freshness_of_data_and_forecast(engine) -> None:
    latest = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
    db.upsert_dataframe(
        engine,
        db.LoadActual,
        pd.DataFrame({"timestamp_utc": [latest], "load_mw": [50_000.0], "source": "smard"}),
    )
    db.upsert_dataframe(
        engine,
        db.LoadForecastModel,
        pd.DataFrame(
            {
                "timestamp_utc": [latest + pd.Timedelta(hours=1)],
                "model_version": "1",
                "model_name": "m",
                "horizon": "d_plus_1",
                "forecast_mw": 50_000.0,
                "issued_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5),
            }
        ),
    )
    m, rec = enabled("ingest")
    results = [IngestResult("smard", rows=3), IngestResult("weather_forecast", status="failed")]
    ingest_metrics(m, engine, results)
    m.flush()

    got = rec.metrics()
    assert value(got, "ingest.rows", step="smard") == 3
    assert value(got, "ingest.step_failed", step="weather_forecast") == 1
    assert 2 <= value(got, "data.freshness_hours") < 3.1
    assert value(got, "forecast.age_hours") == pytest.approx(5, abs=0.1)
    assert m.status == "failed"  # one failed step marks the run


def test_forecast_metrics_only_count_a_stored_forecast() -> None:
    m, rec = enabled("forecast")
    forecast_metrics(m, None, dry_run=False)  # skipped: already issued today
    result = SimpleNamespace(
        model_version="1",
        frame=pd.DataFrame(index=range(24)),
        issued_at=pd.Timestamp("2026-10-16T05:41Z"),
        last_actual=pd.Timestamp("2026-10-16T03:00Z"),
        deadline_margin_hours=4.3,  # due 10:00 UTC (summer time on 16 Oct)
    )
    forecast_metrics(m, result, dry_run=True)  # a dry run is not an issue
    m.flush()
    assert set(rec.metrics()) == {"forecast.skipped"}

    m, rec = enabled("forecast")
    forecast_metrics(m, result, dry_run=False)
    m.flush()
    got = rec.metrics()
    assert value(got, "forecast.issued", model_version="v1") == 1
    assert value(got, "forecast.hours") == 24
    assert value(got, "forecast.data_age_hours") == pytest.approx(2 + 41 / 60)
    assert value(got, "forecast.deadline_margin_hours", model_version="v1") == 4.3


def test_monitor_metrics_accuracy_drift_triggers_and_decision() -> None:
    window = SimpleNamespace(
        window_days=7, model_mape=2.8, official_mape=3.04, improvement_pct=10.5, on_time_pct=97.6
    )
    decision = SimpleNamespace(
        champion_version="1",
        performance=SimpleNamespace(windows=[window]),
        drift=SimpleNamespace(share_drifted=0.375),
        checks=[
            SimpleNamespace(name="error", fired=False),
            SimpleNamespace(name="drift", fired=True),
        ],
    )
    outcome = SimpleNamespace(decision="promoted (dry-run)", cv_gate="passed")
    m, rec = enabled("monitor")
    monitor_metrics(m, decision, outcome)
    m.flush()

    got = rec.metrics()
    assert value(got, "model.mape", window="7d", model_version="v1") == 2.8
    assert value(got, "official.mape", window="7d") == 3.04
    assert value(got, "forecast.on_time_pct", window="7d", model_version="v1") == 97.6
    assert value(got, "drift.share") == 0.375
    assert value(got, "drift.unavailable") == 0
    assert value(got, "trigger.fired", trigger="drift") == 1
    assert value(got, "trigger.fired", trigger="error") == 0
    assert value(got, "retrain.ran") == 1
    assert value(got, "retrain.cv_gate_failed") == 0
    # a dry-run "promotion" never counts as a real one
    assert value(got, "retrain.promoted", decision="promoted", dry_run="true") == 0


def test_monitor_metrics_send_the_cv_gate_on_every_pass() -> None:
    """0 or 1 on every pass: a gauge sent only when it is 1 would leave a monitor in alert."""
    decision = SimpleNamespace(champion_version="1", performance=None, drift=None, checks=[])
    for outcome, expected in [
        (None, 0),  # a clear pass: no challenger at all
        (SimpleNamespace(decision="skipped", cv_gate="not run"), 0),
        (SimpleNamespace(decision="kept", cv_gate="not run"), 0),  # the margin failed first
        (SimpleNamespace(decision="kept", cv_gate="failed"), 1),
        (SimpleNamespace(decision="kept (dry-run)", cv_gate="failed"), 1),
        (SimpleNamespace(decision="promoted", cv_gate="passed"), 0),
    ]:
        m, rec = enabled("monitor")
        monitor_metrics(m, decision, outcome)
        m.flush()
        assert value(rec.metrics(), "retrain.cv_gate_failed", model_version="v1") == expected


def test_monitor_metrics_flag_a_drift_check_that_could_not_run() -> None:
    decision = SimpleNamespace(champion_version="1", performance=None, drift=None, checks=[])
    m, rec = enabled("monitor")
    monitor_metrics(m, decision, None)
    m.flush()
    got = rec.metrics()
    assert value(got, "drift.unavailable") == 1 and "drift.share" not in got
    assert value(got, "retrain.ran") == 0
