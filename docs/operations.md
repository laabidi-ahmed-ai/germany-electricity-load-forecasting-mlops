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

**The forecast's deadline.** A D+1 forecast counts only if it is issued by 12:00 Berlin on
the day before its target day, when the day-ahead auction closes: 11:00 UTC in winter,
10:00 UTC in summer. The 05:40 run has about 5 hours of margin (`forecast.deadline_margin_hours`).
A later issue is stored and shown, but never scored (README §16.9).

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
  **force**. (Without it, the run exits because today's forecast exists.) The new issue is
  added next to the earlier one. If it is made before the deadline, it replaces the earlier
  one in the scores; after the deadline it is kept, but not scored.
- **Rehearse a retrain without touching the champion.** Actions → *monitor-retrain* →
  *Run workflow* → mode **force**, **dry_run** ticked.
- **Evaluate the calendar-day (D+1) forecast without touching production.** Actions →
  *bootstrap* → *Run workflow* → tick **skip_backfill**, horizon **d_plus_1**, untick
  **promote**. The log shows the cross-validation report (LightGBM vs the 48 h and 168 h
  baselines) and the one-year fair backtest against the official forecast (*Run the fair
  backtest*, below); the run is in MLflow, nothing is registered or forecast.
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
  3. Compare B with A on:
     - the CV mean MAE;
     - the paired folds (same validation windows, how many B wins);
     - the both-baselines verdict;
     - the fair backtest's **headline row** (issue-time weather: MAE, gain over the official
       forecast, CI). The measured-weather row is not the comparison.

     `best_iteration` (MLflow) should match if the data was the same; the official
     forecast's backtest MAE must be identical. If a run's backtest step is red with
     "Previous Runs weather fetch failed: giving up on …", Open-Meteo was down, and the model
     code is not at fault. Run A and B again later, back to back, so that they see the same
     data. Other fetch errors are not outages (*Run the fair backtest*, below).
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

## Run the fair backtest (D+1)

The one-year backtest behind README §15:
- the model is trained once, on data up to a year before the latest actual;
- each D+1 issue of the following year is replayed with the weather forecast that existed at
  05:40 UTC on the day before (Open-Meteo Previous Runs);
- the same model is also scored with measured weather, on the same hours.

Why and how: README §16.11. It only reads the database: nothing is registered, forecast or
written.

**The production number.** Push the branch with the code to evaluate, then Actions →
*bootstrap* → *Run workflow*:
- *Use workflow from*: that branch (or `main`);
- **skip_backfill** ticked, **skip_training** unticked, horizon **d_plus_1**, **promote**
  unticked.

Avoid roughly 05:30–07:00 and 08:30 UTC (the `promotion` queue, above). The run also logs a
training run to MLflow, so the `--promote-latest` warning above applies.

In the log of the step *Out-of-sample backtest vs the official forecast*, keep:
1. the line `Measured-weather backtest (...)`, which is printed before anything is fetched;
2. the `Fair backtest:` block, down to the `autocorrelation` line;
3. the `Per month` table.

Check:
- The official MAE is the same in all three rows: they score the same hours.
- The summary says `0 dropped for missing archived forecasts`, or a handful (above 1 % the
  log warns).
- There are 364 or 365 target days, depending on the hour of the latest actual. There are 365
  when it is stamped between 23:00 and 03:00 UTC (22:00 and 03:00 in summer) and 364
  otherwise; run #7, with the latest actual at 21:00 UTC, gave 364.
- The verdict line says `clearly beats`, `does not clearly beat` or `clearly loses to` the
  official forecast. `too few blocks for a reliable interval` means fewer than 10 weeks were
  scored, and there is no verdict.

Record the numbers in README §15 and §16.11 with the run's URL: they are only in the Actions
log, and those logs expire.

**The backtest step is red with "Previous Runs weather fetch failed".** The measured-weather
line above the error was already scored. What follows the colon says which case it is:
- **`giving up on … after 4 attempts`** (HTTP 5xx, 408/429, timeouts, connection errors,
  bodies that are not JSON): Open-Meteo is down, as with the 503s of 2 Nov 2026 (README
  §16.2). The model code is not at fault. Re-run the bootstrap later.
- **`… returned HTTP 400`** (or another final 4xx), or a bare field name such as `'hourly'`
  (a missing field): this is not an outage. Either the window starts before the archive
  (best_match is kept from January 2024), or the API changed. Re-running will not help: check
  `--backtest-days` / `--as-of`, then `WeatherClient.fetch_previous_runs`.

**Locally** (PowerShell; the connection string lives only in this session):

```powershell
$env:DATABASE_URL = "<connection string>"
.venv\Scripts\python -m src.monitoring.performance --backtest-days 365 --horizon d_plus_1 --fair-weather
Remove-Item Env:DATABASE_URL
```

Without `DATABASE_URL` it runs on the local SQLite file (`make data` brings it up to date).
It needs no ENTSO-E token (the official forecast comes from SMARD). It trains one model and
sends five requests to Open-Meteo.

**Which database role.** The command only reads `load_actual`, `load_forecast_official`,
`load_forecast_model` and `weather_hourly`.
- Verified, by the offline tests: it runs against a SQLite file opened read-only. It never
  calls `init_db`, so there is no schema-version check and no read of `alembic_version_app`.
- Not verified: the grants in production. `docs/cloud-setup.md` gives `dashboard_ro`
  `SELECT` on all tables plus default privileges, so its connection string should work. A
  permission error means a missing grant: use the owner connection string instead.

**If the forecast trigger moves.** `ISSUE_TIME_UTC` in `src/monitoring/fair_backtest.py`
(05:40 UTC) mirrors the external scheduler's `forecast` job. If that job ever moves earlier
than 05:40 UTC, move the constant with it. Otherwise the replay would use weather forecasts
that the live job did not have yet. A later trigger is safe, because the replay is then
stricter than live.

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
  `DATABASE_URL` unset, record it at the baseline and upgrade it, or delete the file and
  run the backfill again. 0002 drops the four time indexes if the file has them, and skips
  them if not. 0003 refuses if the file has forecasts of a model version without an export
  (`make register` writes the exports):

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
- **A later migration:** `.venv\Scripts\alembic revision --rev-id 0004 -m "<what changes>"`,
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

## Keep every issued forecast (0003)

Revision `0003` changes how forecasts are stored (README §16.9). In one transaction, it:

1. checks that every stored forecast's model version has its export in `model_artifacts`
   (the guard). If one has not, it stops before changing anything and names the versions;
2. adds `horizon` to `load_forecast_model`, filled in from those exports, and makes the
   primary key `(timestamp_utc, model_version, issued_at)`: a re-issue adds rows instead of
   overwriting the earlier issue. On Postgres these are in-place `ALTER`s, so the table
   keeps its rows and its grants;
3. creates `load_forecast_official_revision`, every value the official forecast has had. It
   starts as a copy of `load_forecast_official`, and any row-count difference aborts. From
   then on, each ingest adds the hours whose value is new or changed. The copy is a
   snapshot of the values held at the upgrade: its `ingested_at` is when each row was last
   written (each ingest rewrites the hours it re-reads), not when the value was first seen,
   and earlier revisions were never kept. The revision history is complete only from the
   upgrade onward;
4. creates `load_forecast_model_weather`, the per-city weather each stored forecast used, as
   it was read. It starts empty.

From the deploy on, an hour is scored with the latest issue made by its deadline (see the
daily timeline). The dashboard, the monitor and Datadog report the on-time share.

For every command below, as for 0002:
- connect as the **owner role**, the one the jobs' `DATABASE_URL` uses;
- use the **non-pooled** connection string;
- set `DATABASE_URL` for the PowerShell session only.

SQL to record before and after (Neon SQL editor, on the branch being migrated):

```sql
SELECT version_num FROM alembic_version_app;
SELECT model_name, model_version, count(*) AS rows, min(issued_at), max(issued_at)
FROM load_forecast_model GROUP BY 1, 2 ORDER BY 2;
SELECT count(*) FROM load_forecast_official;
SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint
WHERE conrelid = 'load_forecast_model'::regclass AND contype = 'p';
SELECT table_name, grantee, privilege_type FROM information_schema.role_table_grants
WHERE table_name LIKE 'load_forecast%' ORDER BY 1, 2, 3;
SELECT pg_size_pretty(pg_database_size(current_database()));
-- The guard, in advance: forecasts of a version without an export. Must return no rows.
SELECT DISTINCT f.model_name, f.model_version FROM load_forecast_model f
WHERE NOT EXISTS (SELECT 1 FROM model_artifacts a
                  WHERE a.name = f.model_name AND a.version = f.model_version);
```

After the upgrade only (the tables do not exist before):

```sql
SELECT model_version, horizon, count(*) FROM load_forecast_model GROUP BY 1, 2 ORDER BY 1;
SELECT count(*) FROM load_forecast_official_revision;  -- = load_forecast_official at first
SELECT count(*) FROM load_forecast_model_weather;      -- 0 at first
```

**If the guard query returns rows, stop and send me the output.** An export of that version
was pruned, so its horizon is unknown.

### Rehearsal (on a copy of production)

1. Neon → *Branches* → *Create branch* `rehearse-0003` from `main` (current data). Create
   it **after the day's 05:40 forecast is stored**. Tomorrow's hours then already have an
   issue, which step 6 re-issues.
2. On `rehearse-0003`, run the SQL above. Note the counts, the primary key, the grants and
   the size. The guard query returns no rows.
3. In PowerShell, from the repository root:

   ```powershell
   $env:DATABASE_URL = "<rehearse-0003: owner role, non-pooled>"
   .venv\Scripts\alembic current      # 0002
   $start = Get-Date
   .venv\Scripts\alembic upgrade head
   "upgrade took {0:n1} s" -f ((Get-Date) - $start).TotalSeconds
   .venv\Scripts\alembic current      # 0003 (head)
   .venv\Scripts\alembic check        # No new upgrade operations detected.
   ```

   The upgrade logs the horizon given to each version and the rows seeded into
   `load_forecast_official_revision`. Note both.
4. Run the SQL again, both parts. The counts match step 2. Every version has its export's
   horizon. The primary key has three columns, and the grants of `load_forecast_model` are
   unchanged. Then run the GRANT step of the production runbook (step 6 below) on the
   branch: the branch has the same roles.
5. **A real ingest** on the branch. It calls SMARD and Open-Meteo, and writes only to the
   branch:

   ```powershell
   .venv\Scripts\python -m src.data.ingest
   ```

   It must end green. The summary line of `smard_forecast` says how many hours were new or
   changed: the hours SMARD published since the branch was made, and any revision. Note it.
6. **A dry-run forecast**, then **a real one** (both mandatory):

   ```powershell
   .venv\Scripts\python -m src.serving.batch_forecast --dry-run
   .venv\Scripts\python -m src.serving.batch_forecast
   ```

   The dry run writes no forecast rows (it may store fetched weather on the branch). Note
   the target window and the `deadline margin` from its output. The rehearsal's issue is
   probably late (after 11:00 UTC); that does not matter for storage. The real run is the
   only test of the new write against Postgres, and of "a re-issue appends". Check it on the
   branch, with the target start from the output:

   ```sql
   SELECT issued_at, model_version, horizon, count(*) FROM load_forecast_model
   WHERE timestamp_utc >= '<target start>' GROUP BY 1, 2, 3 ORDER BY 1;
   SELECT issued_at, count(DISTINCT city) AS cities, count(DISTINCT timestamp_utc) AS hours,
          count(*) AS rows FROM load_forecast_model_weather GROUP BY 1 ORDER BY 1;
   ```

   The first query shows two issues for the target hours, each with one row per hour (24;
   23 or 25 on a clock-change day). One is production's at 05:40, unchanged; the other is
   the new one, with `horizon` `d_plus_1`. The second query shows the new issue with
   5 cities × the target hours (120 rows on a 24-hour day).
7. The live scores on the branch, read-only:

   ```powershell
   .venv\Scripts\python -m src.monitoring.performance
   ```

   Note the 7-day and 30-day lines, including "on time X% (n late, m missing)".
8. Rehearse the rollback, then upgrade again:

   ```powershell
   .venv\Scripts\alembic downgrade 0002
   ```

   Run the step 6 queries and the first part of the SQL. The target hours now have one
   issue, the later one (the rehearsal's). The weather-inputs table and the revision table
   are gone, and the primary key has two columns again. This is what a downgrade deletes
   for good (see Rollback). Then:

   ```powershell
   .venv\Scripts\alembic upgrade head
   .venv\Scripts\alembic check
   Remove-Item Env:DATABASE_URL
   ```

9. Record the numbers in README §16.9, then delete the `rehearse-0003` branch.

### Production

1. **Before the window starts:** the PR is open and CI is green, so the merge is one click.
2. **When:** right after an hourly `ingest` run is green (the scheduler starts it at :20).
   Avoid 05:30–07:00 and around 08:30 UTC, when the forecast and monitor-retrain run. The
   merge follows within minutes, before the next :20 ingest.
3. Neon → *Branches* → *Create branch* `backup-before-0003` from `main` (current data).
4. Run the SQL above on `main` and note the results. The guard query returns no rows.
5. In PowerShell:

   ```powershell
   $env:DATABASE_URL = "<main: owner role, non-pooled>"
   .venv\Scripts\alembic current      # 0002
   (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'")   # the restore point: write it down
   .venv\Scripts\alembic upgrade head
   .venv\Scripts\alembic current      # 0003 (head)
   .venv\Scripts\alembic check        # No new upgrade operations detected.
   Remove-Item Env:DATABASE_URL
   ```

6. **Grants for the dashboard.** In the SQL editor, as the owner role (the one the upgrade
   ran as):

   ```sql
   GRANT SELECT ON load_forecast_official_revision, load_forecast_model_weather TO dashboard_ro;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro;
   SELECT table_name, privilege_type FROM information_schema.role_table_grants
   WHERE grantee = 'dashboard_ro' ORDER BY 1, 2;
   ```

   The last query lists `SELECT` on every application table, the two new ones included.
   The default privileges make tables the owner role creates later readable too.
7. Run the SQL again, both parts: the same counts, the three-column key, the grants.
8. **Merge.** Until then, a job on the old code stops with `SchemaError` (revision 0003 is
   unknown to it) and writes nothing.
9. Actions → *ingest* → *Run workflow*. Check that it is green. Then Streamlit → *Manage
   app* → ⋮ → *Reboot app*, and check that the dashboard loads (the change touches
   `dashboard/`).
10. The next morning:
    - *forecast* and *monitor-retrain* are green;
    - the forecast log shows `deadline margin` of about +5.3 h (winter);
    - `SELECT issued_at, count(*) FROM load_forecast_model_weather GROUP BY 1;` shows the
      new issue with 120 rows (5 cities × 24 hours);
    - the dashboard shows the on-time share, and Datadog receives `forecast.on_time_pct`
      and `forecast.deadline_margin_hours`.
11. After a day of green jobs, delete the `backup-before-0003` branch.

### Rollback

- **What `alembic downgrade 0002` deletes for good:**
  - every forecast issue except the latest per hour and model version;
  - every official revision recorded since the upgrade (the table is dropped);
  - every weather input recorded since the upgrade (the table is dropped).

  No later upgrade brings them back. A downgrade is therefore for a code problem right
  after the deploy. Later on, prefer a forward fix.
- **Code problem, right after the deploy:** run `.venv\Scripts\alembic downgrade 0002`
  against `main` first, then revert the merge and push. Do the two back to back, outside
  05:30–07:00 UTC. In between, jobs on the new code stop with `SchemaError` and write
  nothing, and the dashboard errors until the revert is deployed and the app rebooted.
- **Data problem, within 6 hours of the restore point:** use Neon's instant restore to
  restore `main` to the noted UTC time. Data, schema version and tables return to their
  state at that time. Then revert the merge and push. The next ingest fetches the hours
  stored since then again.
- **Data problem, later than that:**
  1. Revert the merge and push: `backup-before-0003` is at revision 0002.
  2. Point the `DATABASE_URL` secrets at `backup-before-0003`: the GitHub secret with the
     owner role, the Streamlit secret with `dashboard_ro`.
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
