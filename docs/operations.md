# Operations runbook

How the system is kept running day to day. Why it is set up this way, and the incident
history, are in README §16; the one-time cloud setup is in `docs/cloud-setup.md`.

## Daily timeline (UTC)

| When | What starts it | Job |
|---|---|---|
| every hour, :20 | external scheduler (backup: GitHub cron at :15) | `ingest` |
| 05:40 | external scheduler (backup: GitHub cron at 05:40) | `forecast` (ingests first, then forecasts the champion's horizon: currently D+1, tomorrow 00:00–24:00 Berlin) |
| when `forecast` finishes | `workflow_run` chain (backup: GitHub cron at 08:30) | `monitor-retrain` |

GitHub's own cron ran these jobs 4–8 hours late every day (README §16.1), so it is only a
backup. Late backups are harmless: `forecast` and `monitor-retrain` pass `--once-per-day`
on automatic runs and exit without doing anything once today's forecast / monitoring
pass exists, and `ingest` is incremental.

## Setting up the external scheduler (once)

Any service that can send a scheduled HTTP POST works; these steps use
[cron-job.org](https://cron-job.org) (free).

1. **Create a token that can only start this repository's workflows.**
   GitHub → *Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token*:
   - *Repository access*: **Only select repositories** → this repository.
   - *Permissions → Repository permissions → Actions*: **Read and write** (nothing else;
     *Metadata: read* is added automatically).
   - *Expiration*: e.g. one year, with a calendar reminder to renew it.

   The token can start, re-run and cancel this repository's workflows. It cannot read
   secrets or change code, but a leaked token can still dispatch any workflow - including
   `bootstrap`, which retrains and promotes a new champion - and disable workflows, which
   also stops the GitHub cron backups. If it leaks, revoke it immediately in the same
   settings page, then check the Actions tab: the cron backups keep the system running
   only if no workflow was disabled.

2. **Create two cron jobs on cron-job.org** (*Create cronjob*, time zone **UTC**):

   | Title | URL | Schedule |
   |---|---|---|
   | forecast | `https://api.github.com/repos/laabidi-ahmed-ai/germany-electricity-load-forecasting-mlops/actions/workflows/forecast.yml/dispatches` | every day at 05:40 |
   | ingest | `https://api.github.com/repos/laabidi-ahmed-ai/germany-electricity-load-forecasting-mlops/actions/workflows/ingest.yml/dispatches` | every hour at minute 20 |

   In *Advanced* for both:
   - Request method: **POST**
   - Headers:
     - `Accept: application/vnd.github+json`
     - `Authorization: Bearer <the token>`
     - `X-GitHub-Api-Version: 2022-11-28`
     - `Content-Type: application/json`
   - Request body: `{"ref":"main"}`

   `monitor-retrain` needs no external job: it follows the forecast.

3. **Test it.** Use *Test run* on each job: the response must be **HTTP 204**, and a new run
   marked *workflow_dispatch* appears in the Actions tab within seconds. Turn on
   cron-job.org's failure notifications so a broken token is noticed.

## Routine procedures

- **Health check.** Datadog alerts by e-mail on a stopped ingest, stale data, a missing
  forecast, a rising error or a failed job (`docs/observability.md`). By hand: the Actions
  tab (all green), and the dashboard's *Data coverage* (latest actual) and *last model
  forecast* timestamps. `python -m src.models.registry --show` prints the champion.
- **Forecast again today on purpose.** Actions → *forecast* → *Run workflow* → tick
  **force**. (Without it, the run exits because today's forecast exists.)
- **Rehearse a retrain without touching the champion.** Actions → *monitor-retrain* →
  *Run workflow* → mode **force**, **dry_run** ticked.
- **Evaluate the calendar-day (D+1) forecast without touching production.** Actions →
  *bootstrap* → *Run workflow* → tick **skip_backfill**, horizon **d_plus_1**, untick
  **promote**. The log shows the cross-validation report (LightGBM vs the 48 h and 168 h
  baselines) and a 365-day backtest against the official forecast; the run is in MLflow,
  nothing is registered or forecast.
- **Switch production to D+1** (after a good evaluation): the same with **promote** ticked.
  It trains, registers and promotes the D+1 champion and issues its first forecast; from
  then on the daily jobs, monitoring and retraining follow the champion's horizon. Roll
  back with the command below.
- **Roll back the model.** With `DATABASE_URL` **and** `MLFLOW_TRACKING_URI` in `.env`
  (the same store the workflows use: the `MLFLOW_TRACKING_URI` secret if set, else the
  `DATABASE_URL` value - the local default `sqlite:///mlruns/mlflow.db` is the wrong
  registry): `python -m src.models.registry --promote <previous version>`. The MLflow alias
  moves first; if it fails, the champion serving reads is left unchanged.
- **A bootstrap or monitor-retrain run shows as cancelled.** Both workflows share the
  `promotion` concurrency group, and GitHub keeps only one pending run per group, so a newer
  queued run cancels the waiting one - either a bootstrap or that day's monitor-retrain
  (which then records no monitoring event). Re-run it.
- **Rotate the database password.** Neon → *Roles* → the owner role → *Reset password*;
  paste the new connection string into the GitHub secret `DATABASE_URL` (plain URL, **no
  quotes**); run *ingest* manually and check it is green. The dashboard uses its own
  read-only role, configured in Streamlit's secrets (TOML: `DATABASE_URL = "..."`, **with**
  quotes).
- **After a push that changes `dashboard/`.** Streamlit → *Manage app* → ⋮ → *Reboot app*,
  then check the page loads (a hot reload can keep an old module in memory).
- **Keep the schedules alive.** GitHub disables scheduled workflows in a repository with no
  activity for 60 days; push at least once every two months.
- **Renew the scheduler token** before it expires (step 1 above, then update the
  `Authorization` header in both cron jobs).

## Dependencies: adding and upgrading packages

Ranges live in `pyproject.toml`; the exact versions every job installs live in
`requirements/constraints.txt` (the lock, generated by `uv`). Never edit the lock by hand.
`uv` is a separate tool: `pip install uv` inside the activated venv. Without `make` (plain Git
Bash on Windows), run the command shown in the `Makefile` under `LOCK_CMD`.

- **Add or change a dependency.** Edit `pyproject.toml` (and `requirements/ingest.txt` if the
  ingest job needs it), then `make lock`. Existing pins are kept; only what is needed changes.
  Commit both files together. `tests/test_lock.py` fails if you forget.
- **Upgrade (about once a month, or for a security fix).** `make lock-upgrade`, then
  `pip install -e ".[dev]" -c requirements/constraints.txt`, `python -m pytest -m "not
  integration"` and ruff. Read `git diff requirements/constraints.txt`: a new *major* version
  (pandas 2 → 3) deserves a look at its changelog. Push and wait for CI on 3.11 and 3.12.
- **Upgrade one package only.** Add `--upgrade-package <name>` to the lock command instead of
  `--upgrade`.
- **An upgrade broke something.** `git revert` the lock commit: production goes back to the
  previous versions on the next run. If the new version is the problem, add an upper bound in
  `pyproject.toml` (with a comment saying why, as for pandas and NumPy) and `make lock`.
