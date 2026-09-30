"""GitHub Actions workflow contracts (offline): structure, secrets, schedules, lean ingest."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml

from config.settings import PROJECT_ROOT

WORKFLOWS = PROJECT_ROOT / ".github" / "workflows"
SCHEDULED = {"ingest.yml", "forecast.yml", "monitor_retrain.yml"}
HEAVY_MODULES = (
    "lightgbm",
    "mlflow",
    "evidently",
    "sklearn",
    "scipy",
    "torch",
    "streamlit",
    "fastapi",
    "optuna",
)


def load(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # PyYAML parses the bare key `on` as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def steps(wf: dict) -> list[dict]:
    (job,) = wf["jobs"].values()
    return job["steps"]


def run_lines(wf: dict) -> str:
    return "\n".join(s.get("run", "") for s in steps(wf))


@pytest.mark.parametrize("name", sorted(SCHEDULED | {"bootstrap.yml", "ci.yml"}))
def test_workflow_parses_and_has_a_single_job_with_timeout(name: str) -> None:
    wf = load(name)
    assert "jobs" in wf and len(wf["jobs"]) == 1
    (job,) = wf["jobs"].values()
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] > 0
    assert any(s.get("uses", "").startswith("actions/checkout@") for s in job["steps"])
    assert any(s.get("uses", "").startswith("actions/setup-python@") for s in job["steps"])


@pytest.mark.parametrize("name", sorted(SCHEDULED))
def test_scheduled_workflows_have_cron_and_manual_dispatch(name: str) -> None:
    on = load(name)["on"]
    assert "workflow_dispatch" in on
    crons = [s["cron"] for s in on["schedule"]]
    assert len(crons) == 1 and len(crons[0].split()) == 5


def test_bootstrap_is_manual_only() -> None:
    on = load("bootstrap.yml")["on"]
    assert list(on) == ["workflow_dispatch"]
    assert {"start", "skip_backfill", "skip_training"} <= set(on["workflow_dispatch"]["inputs"])


@pytest.mark.parametrize("name", sorted(SCHEDULED | {"bootstrap.yml"}))
def test_jobs_use_secrets_and_never_hardcode_them(name: str) -> None:
    wf = load(name)
    (job,) = wf["jobs"].values()
    env = job["env"]
    assert env["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert env["ENTSOE_API_TOKEN"] == "${{ secrets.ENTSOE_API_TOKEN }}"  # optional at runtime
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert "postgresql://" not in text and "postgres://" not in text  # no inline URLs
    assert any("DATABASE_URL secret is not set" in s.get("run", "") for s in steps(wf))
    assert job.get("permissions", wf.get("permissions")) == {"contents": "read"}


def test_schedule_order_ingest_then_forecast_then_monitor() -> None:
    def cron(name: str) -> list[str]:
        return load(name)["on"]["schedule"][0]["cron"].split()

    ingest, forecast, monitor = (
        cron("ingest.yml"),
        cron("forecast.yml"),
        cron("monitor_retrain.yml"),
    )
    assert ingest[1] == "*" and forecast[1] != "*" and monitor[1] != "*"  # hourly vs daily
    # the daily jobs run after that hour's ingest, and the monitor after the forecast
    assert int(ingest[0]) < int(forecast[0])
    assert (int(forecast[1]), int(forecast[0])) < (int(monitor[1]), int(monitor[0]))


def test_ingest_workflow_installs_only_the_lean_requirements() -> None:
    wf = load("ingest.yml")
    text = run_lines(wf)
    assert "pip install -r requirements/ingest.txt" in text
    assert "pip install ." not in text and "pip install -e" not in text
    (job,) = wf["jobs"].values()
    assert job["env"]["PYTHONPATH"] == "."
    assert "python -m src.data.ingest --backfill" in text
    assert "python -m src.data.ingest --sources" in text
    setup = next(s for s in steps(wf) if s.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["cache-dependency-path"] == "requirements/ingest.txt"


def test_heavy_workflows_install_the_project_and_run_the_right_entrypoints() -> None:
    fc = run_lines(load("forecast.yml"))
    assert "pip install ." in fc
    assert "python -m src.data.ingest" in fc and "python -m src.serving.batch_forecast" in fc

    mon = load("monitor_retrain.yml")
    text = run_lines(mon)
    assert (
        "python -m src.monitoring.retrain" in text and "--check-only" in text and "--force" in text
    )
    (job,) = mon["jobs"].values()
    assert job["env"]["MLFLOW_TRACKING_URI"] == (
        "${{ secrets.MLFLOW_TRACKING_URI || secrets.DATABASE_URL }}"
    )

    boot = run_lines(load("bootstrap.yml"))
    for cmd in (
        "src.data.ingest --backfill",
        "src.features.build_features",
        "src.models.train",
        "src.models.registry --promote-latest",
        "src.serving.batch_forecast",
        "src.monitoring.retrain --check-only",
    ):
        assert f"python -m {cmd}" in boot, cmd


def test_lean_requirements_cover_the_ingest_cli_imports() -> None:
    """Import the ingest CLI in a fresh interpreter: no heavy module may be pulled in."""
    code = (
        "import sys, json; import src.data.ingest; "
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    heavy = [m for m in HEAVY_MODULES if m in loaded]
    assert not heavy, f"ingest CLI imports heavy modules: {heavy}"

    listed = {
        line.split(">=")[0].split("[")[0].strip().lower()
        for line in (PROJECT_ROOT / "requirements" / "ingest.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "requests",
        "pandas",
        "sqlalchemy",
        "psycopg",
        "pydantic-settings",
        "entsoe-py",
    } <= listed
    assert not ({"lightgbm", "mlflow", "evidently", "torch"} & listed)


def test_dockerfile_and_compose_do_not_depend_on_local_mlruns() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert "DATABASE_URL" in api["environment"]
    assert not any("mlruns" in v for v in api.get("volumes", []))
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "libgomp1" in dockerfile and 'CMD ["uvicorn", "src.serving.api:app"' in dockerfile
