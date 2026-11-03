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
- **`forecast` is red but the forecast step ran.** The refresh (ingest) failed, e.g. Open-Meteo down, and the job forecast from the stored data anyway; it stays red so the failure is seen. Check the refresh step's log; the next hourly `ingest` catches up.
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
- **Evaluate a model-code change against production** (anything that changes how the model
  is trained; README §16.7 is the worked example):
  1. Commit the change on a branch and push the branch (not `main`: the daily
     monitor-retrain would start training challengers with it).
  2. Actions → *bootstrap* → *Run workflow*, evaluate-only: tick **skip_backfill**, the
     champion's horizon (`registry --show`; currently **d_plus_1**), untick **promote**. Run it
     from `main` (A), then immediately from the branch (B, *Use workflow from*), so both see
     the same data. Avoid roughly 05:30–07:00 and 08:30 UTC: both workflows share the
     `promotion` queue with monitor-retrain, and a newer queued run cancels a waiting one.
  3. Compare B with A: CV mean MAE, the paired folds (same validation windows, how many B
     wins), the both-baselines verdict, and the backtest's gain over the official forecast.
     `best_iteration` (MLflow) should match if the data was the same; the official
     forecast's backtest MAE must be identical.
  4. Merge only if B holds. Then promote: *bootstrap* from `main` with **skip_backfill** and
     **promote** ticked (the promoting run skips the backtest).
  5. Record the numbers in README §16, including the backtest: it is only in the Actions log,
     and those logs expire.

  **While evaluation runs exist, don't run `python -m src.models.registry --promote-latest`
  by hand:** it registers and promotes the newest *training run* in MLflow, which may be an
  evaluation run - from the branch, or from `main` before the merge. Promote with the
  bootstrap, which trains first, or name the run: `--run-id <run id>`.
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

## Schema migrations

The application's tables are versioned with Alembic (`alembic.ini`, `src/data/migrations/`).
Jobs never migrate. `ingest`, `batch_forecast`, `registry`, `retrain` and the API call
`init_db`, which builds the schema only on a database that has none of the application's
tables, and otherwise checks that the database is at the revision the code expects
(`SCHEMA_REVISION` in `src/data/db.py`). If it is not, they stop with a `SchemaError` that
names the command to run, before writing anything. `build_features`, `train`, `performance`,
`drift` and the dashboard only read and don't check. Every scheduled workflow starts with
`ingest`, so each scheduled run is checked before anything writes.

- **Our version table is `alembic_version_app`.** The `alembic_version` table in the same
  database belongs to MLflow (its own migrations; the workflows' `MLFLOW_TRACKING_URI` falls
  back to `DATABASE_URL`): never stamp or edit it. Alembic here sees only our tables.
- **Production's time-series tables were TimescaleDB hypertables until 0002.** `init_db`
  made `load_actual`, `load_forecast_official`, `weather_hourly` and `load_forecast_model`
  hypertables (when the `timescaledb` extension was available) until commit c7e7058
  removed that setup without converting them back. Alembic could not see that, except
  through the index `create_hypertable` gave each table: `<table>_timestamp_utc_idx` on
  `timestamp_utc DESC`. The baseline `0001` includes those four indexes, so it describes
  production at that revision. `0002` converts the hypertables to plain tables and drops
  the indexes, which duplicated the primary keys (see "Convert the leftover hypertables
  (0002)" below).
- **Run Alembic** from the repository root, with the target database set for the current
  PowerShell session only (without it, Alembic uses `.env`'s value or the local SQLite file):

  ```powershell
  $env:DATABASE_URL = "<connection string>"
  .venv\Scripts\alembic current
  Remove-Item Env:DATABASE_URL
  ```

- **A local database created before migrations** fails the same check: with
  `DATABASE_URL` unset, record it at the baseline and upgrade it (0002 drops the four time
  indexes if the file has them, and skips them if not), or delete the file and run the
  backfill again:

  ```powershell
  Remove-Item Env:DATABASE_URL -ErrorAction SilentlyContinue   # target the local file
  .venv\Scripts\alembic stamp 0001
  .venv\Scripts\alembic upgrade head
  .venv\Scripts\alembic check
  ```
- **Stamp the live database (once, before merging the change that introduced migrations).**
  1. Neon SQL editor, read-only: `SELECT table_name FROM information_schema.tables WHERE
     table_schema = 'public' ORDER BY 1;` and `SELECT version_num FROM alembic_version;`
     (MLflow's revision - note it).
  2. Neon → *Branches* → *Create branch* `backup-before-stamp` from `main` (current data).
  3. In PowerShell, from the repository root:

     ```powershell
     $env:DATABASE_URL = "<Neon connection string of main>"
     .venv\Scripts\alembic current      # no revision yet
     .venv\Scripts\alembic stamp 0001
     .venv\Scripts\alembic current      # 0001 (head at the time)
     .venv\Scripts\alembic check        # No new upgrade operations detected.
     Remove-Item Env:DATABASE_URL
     ```

     `alembic check` compares the live tables with the code. If it reports any
     difference, do not merge: the live schema is not the baseline.
  4. SQL editor: `SELECT version_num FROM alembic_version_app;` returns `0001`, and MLflow's
     `alembic_version` is unchanged.
  5. Merge. A job that runs the new code before the stamp stops with `SchemaError` and
     writes nothing; the stamp fixes it.
  6. After a day of green jobs, delete the `backup-before-stamp` branch.

  Roll back by reverting the merge: the old code ignores `alembic_version_app`. There is
  nothing to downgrade - the baseline `0001` refuses to, because that would drop every table.
- **A later migration:** `.venv\Scripts\alembic revision --rev-id 0003 -m "<what changes>"`,
  then append the id to `SCHEMA_REVISIONS` in `src/data/db.py` (`tests/test_migrations.py`
  checks they match). Before merging: a Neon backup branch, `alembic upgrade head`, `alembic
  check`. To roll back, `alembic downgrade <previous revision>` first, then revert the code.

## Convert the leftover hypertables (0002)

Revision `0002` turns production's four hypertables into plain tables (README §16.8). On
Postgres with TimescaleDB, in one transaction, it:

1. locks the tables that are still hypertables against writes (reads still work);
2. copies each table into a plain table and aborts on any row-count difference;
3. copies the grants (the dashboard's `dashboard_ro`);
4. drops the hypertable with its chunks and renames the copy;
5. restores `<table>_pkey` and runs `ANALYZE`.

Then, on every database, it drops the four `<table>_timestamp_utc_idx` indexes. A failure
anywhere rolls everything back.

Dropping about 900 chunks takes a lock per chunk and per chunk index, all held until
commit. Every session shares one lock table, with room for about
`max_locks_per_transaction × max_connections` locks. If the conversion runs out of room,
it fails with `out of shared memory` (hint: `max_locks_per_transaction`). Nothing has
changed at that point. The fallback, `alembic -x per_table=true upgrade head`, converts
each table in its own transaction and holds about a third of the locks at once. The
rehearsal tells which one production needs.

For every command below:
- Connect as the **owner role**, the one the jobs' `DATABASE_URL` uses. Only the owner can
  drop the tables, and the migration refuses to run as any other role.
- Use the **non-pooled** connection string. In Neon's *Connect* dialog, turn connection
  pooling off; the host has no `-pooler`.
- Set `DATABASE_URL` for the PowerShell session only.

SQL to record before and after (Neon SQL editor, on the branch being converted):

```sql
SHOW max_locks_per_transaction;
SHOW max_connections;
SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables ORDER BY 1;
SELECT 'load_actual' AS t, count(*) FROM load_actual
UNION ALL SELECT 'load_forecast_official', count(*) FROM load_forecast_official
UNION ALL SELECT 'weather_hourly', count(*) FROM weather_hourly
UNION ALL SELECT 'load_forecast_model', count(*) FROM load_forecast_model;
SELECT table_name, grantee, privilege_type FROM information_schema.role_table_grants
WHERE table_name IN ('load_actual', 'load_forecast_official', 'weather_hourly', 'load_forecast_model')
ORDER BY 1, 2, 3;
SELECT tablename, indexname FROM pg_indexes
WHERE tablename IN ('load_actual', 'load_forecast_official', 'weather_hourly', 'load_forecast_model')
ORDER BY 1, 2;
```

### Rehearsal (on a copy of production)

1. Neon → *Branches* → *Create branch* `rehearse-0002` from `main` (current data). Give its
   compute the **same size as production's**: `max_connections` depends on it.
2. On `rehearse-0002`, run the SQL above. Note `max_locks_per_transaction`,
   `max_connections`, the chunk counts (297/297/297/7 when written), the row counts and the
   grants.
3. In PowerShell, from the repository root:

   ```powershell
   $env:DATABASE_URL = "<rehearse-0002: owner role, non-pooled>"
   .venv\Scripts\alembic current      # 0001
   $start = Get-Date
   .venv\Scripts\alembic upgrade head
   "upgrade took {0:n1} s" -f ((Get-Date) - $start).TotalSeconds
   .venv\Scripts\alembic current      # 0002 (head)
   .venv\Scripts\alembic check        # No new upgrade operations detected.
   ```

   The upgrade logs the rows per table and `Locks held by the conversion before commit`.
   Note both.
4. Run the SQL again. The hypertables query returns nothing. The row counts and grants
   match step 2. Each table has one index left, `<table>_pkey`.
5. Rehearse the rollback, then upgrade again. The second upgrade converts nothing and only
   drops the indexes again:

   ```powershell
   .venv\Scripts\alembic downgrade 0001   # the four DESC indexes come back
   .venv\Scripts\alembic upgrade head
   .venv\Scripts\alembic check
   Remove-Item Env:DATABASE_URL
   ```

6. **If step 3 fails with `out of shared memory`**, nothing has changed. Run
   `.venv\Scripts\alembic -x per_table=true upgrade head` and continue from the same point.
   Production then uses the same flag.
7. Record the numbers in README §16.8: the two settings, the locks held, the duration and
   the row counts. Then delete the `rehearse-0002` branch.

### Production

1. **Before the window starts:** the PR is open and CI is green, so the merge is one click.
2. **When:** right after an hourly `ingest` run is green (the scheduler starts it at :20).
   Avoid 05:30–07:00 and around 08:30 UTC, when the forecast and monitor-retrain run. The
   merge follows within minutes, before the next :20 ingest.
3. Neon → *Branches* → *Create branch* `backup-before-0002` from `main` (current data).
4. Run the SQL above on `main` and note the results.
5. In PowerShell:

   ```powershell
   $env:DATABASE_URL = "<main: owner role, non-pooled>"
   .venv\Scripts\alembic current      # 0001
   (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")   # the restore point: write it down
   .venv\Scripts\alembic upgrade head # add -x per_table=true if the rehearsal needed it
   .venv\Scripts\alembic current      # 0002 (head)
   .venv\Scripts\alembic check        # No new upgrade operations detected.
   Remove-Item Env:DATABASE_URL
   ```

6. Run the SQL again: no hypertables, and the same row counts and grants.
7. **Merge.** Until then, a job on the old code stops with `SchemaError` (revision 0002 is
   unknown to it) and writes nothing.
8. Actions → *ingest* → *Run workflow*. Check that it is green and that the dashboard
   loads; it reads through `dashboard_ro`.
9. After a day of green jobs:
   - delete the `backup-before-0002` branch;
   - in the SQL editor, confirm the hypertables query is still empty, then run
     `DROP EXTENSION timescaledb;`.

### Rollback

- **Schema (code problem):** run `.venv\Scripts\alembic downgrade 0001` against `main`
  first. The four indexes come back; the tables stay plain, and the 0001 code does not
  need hypertables. Then revert the merge and push.
- **Data problem, within 6 hours of the restore point:** use Neon's instant restore to
  restore `main` to the noted UTC time. The data, the schema version and the hypertables
  return to their state at that time. Then revert the merge and push. The next ingest
  fetches the hours stored since then again.
- **Data problem, later than that:**
  1. Revert the merge and push: `backup-before-0002` is at revision 0001.
  2. Point the `DATABASE_URL` secrets at `backup-before-0002`: the GitHub secret with the
     owner role, the Streamlit secret with `dashboard_ro`. The branch has the same roles
     and passwords, under its own host.
  3. Backfill: Actions → *ingest* → *Run workflow* → mode **backfill**.

  Whatever was written to `main` after the backup does not come back: model forecasts,
  monitoring events, and MLflow runs and model versions.

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
