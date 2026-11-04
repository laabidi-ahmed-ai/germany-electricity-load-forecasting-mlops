# Cloud setup — running the loop with your PC off

Everything runs on **GitHub Actions** (scheduler + compute) against one **free cloud
Postgres**. Total cost: €0/month. README §14 describes the idea; this page is the
operator's runbook.

## How persistence works (the one design decision that matters)

GitHub runners are ephemeral: whatever a job writes to disk is gone when it ends.
Three things must survive between runs, and all three live in the cloud Postgres:

| What | Where | Why |
|---|---|---|
| Data (load, official forecast and its revisions, weather, every model forecast issue with its weather inputs, monitoring events) | our tables (`src/data/db.py`) | idempotent upserts; histories are appended, never overwritten |
| **The champion model** | `model_artifacts` table: zipped MLflow pyfunc bundle (~3.4 MB) + features / horizon / train window / CV metrics, one row flagged `is_champion` | `registry.promote()` exports it; `registry.load_champion(engine)` loads it. The forecast job, the API and the monitor need **only `DATABASE_URL`**. Rollback = `--promote <old version>`. Old exports are pruned (last 5 kept, champion never). |
| MLflow lineage (runs, params, metrics, model versions, aliases) | MLflow's own tables in the same Postgres (`MLFLOW_TRACKING_URI` = `DATABASE_URL`) | browse it locally with `mlflow ui --backend-store-uri "<DATABASE_URL>"`. MLflow *artifact files* stay on the runner — they are not needed, the DB export is the serving copy. |

Rejected alternative: an S3-compatible object store for MLflow artifacts (Cloudflare R2 /
Backblaze B2 / Supabase Storage). It works, but adds an account, four more secrets and
boto3 for a 3 MB file that fits in a table row.

## Workflows

| Workflow | Trigger | What it does | Install |
|---|---|---|---|
| `ingest.yml` | hourly, external trigger at `:20` (GitHub cron `:15` as backup) + manual (`incremental` / `backfill`) | SMARD load + official day-ahead forecast, Open-Meteo weather → DB | `requirements/ingest.txt` only (~1 min) |
| `forecast.yml` | daily 05:40 UTC, external trigger (GitHub cron as backup) + manual (`force` to re-issue) | incremental ingest → champion (from DB) → forecast for the champion's horizon (currently D+1, tomorrow 00:00–24:00 Berlin; scored only if issued by 12:00 Berlin today) → `load_forecast_model` + its weather inputs | full project |
| `monitor_retrain.yml` | when `forecast` completes (GitHub cron 08:30 UTC as backup) + manual (`auto` / `check-only` / `force`, `dry_run`) | performance vs actuals & official forecast, Evidently drift, triggers → champion/challenger → promote only if it wins | full project |
| `bootstrap.yml` | manual, once | backfill 2021-03-01→now → features → train + CV → register & export champion → first forecast → monitoring check | full project |
| `ci.yml` | push / PR | ruff + pytest (integration tests excluded) | dev extras |

All jobs fail fast with a clear error if `DATABASE_URL` is missing. The database jobs share
one setup step (`.github/actions/setup`), and every job runs on a pinned `ubuntu-24.04`
image rather than `ubuntu-latest`, so a runner-image upgrade happens when it is chosen and
tested, not on GitHub's schedule.

## Checklist (what you do once)

1. **Create a free Postgres.** [Neon](https://neon.tech) (1 GB per project, no card) or
   [Supabase](https://supabase.com) (500 MB). Copy the connection string; it looks like
   `postgresql://user:password@host/dbname?sslmode=require`. Either form is accepted
   (`postgresql://` is routed to the psycopg 3 driver automatically). If you want
   MLflow's tables in a separate database, create a second one and use it as
   `MLFLOW_TRACKING_URI`.
2. **Push the repository to GitHub** (public repo = unlimited Actions minutes; private =
   2,000 min/month, the hourly job uses ~30 min/day of that).
3. **Add the secrets** under *Settings → Secrets and variables → Actions*:
   - `DATABASE_URL` — required
   - `MLFLOW_TRACKING_URI` — optional (defaults to `DATABASE_URL`)
4. **Run `bootstrap`** (*Actions → bootstrap → Run workflow*, defaults). ~10 minutes:
   backfills 5½ years, trains, promotes the champion, writes the first forecast, runs the
   first monitoring check. Re-running is safe.
5. **Confirm** in the Actions logs: `germany-load-day-ahead@champion = v1 (...)` and
   `stored 24 forecast rows` (for a D+1 champion: 23 or 25 on clock-change days). From then on the three scheduled workflows keep the
   system alive. (Optionally trigger `ingest` and `forecast` manually once to see them
   green before the first cron.)
6. **Set up the external scheduler** (`docs/operations.md`): GitHub's cron alone runs the
   daily jobs hours late.
7. **Optional: alerting.** Add the `DD_API_KEY` secret and create the Datadog monitors
   (`docs/observability.md`).

Local development is unchanged: SQLite data DB + SQLite MLflow by default. To point
your laptop at the cloud, put the same `DATABASE_URL` in `.env`.

## Dashboard (Streamlit Community Cloud, free)

`dashboard/app.py` reads the same Postgres and renders forecast vs. actuals, the
head-to-head against the official forecast, drift signals and the retraining timeline.
It installs only `dashboard/requirements.txt` (Streamlit, plotly, pandas, SQLAlchemy,
psycopg) - no LightGBM / MLflow / Evidently - so it boots in well under a minute.

1. [share.streamlit.io](https://share.streamlit.io) → *New app* → this repository,
   branch `main`, main file path `dashboard/app.py`. Community Cloud picks up the
   requirements file next to the entrypoint automatically.
2. *Advanced settings → Secrets*: add `DATABASE_URL = "postgresql://..."`. Prefer a
   **read-only role** here - the dashboard only reads, and it additionally opens every
   Postgres transaction with `default_transaction_read_only=on`, but a read-only role
   keeps a public app from ever holding write credentials. On Neon, as the owner role (the
   one the jobs and migrations use):

   ```sql
   CREATE ROLE dashboard_ro LOGIN PASSWORD '...';
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_ro;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_ro;
   ```

   The last line makes tables the owner role creates later (a migration) readable too.
   Without it, each new table needs its own `GRANT SELECT` (docs/operations.md does it for
   each migration). (Supabase: same, from the SQL editor.)
3. Deploy. Queries are cached for 10 minutes; the *Refresh data* button clears the cache.

Locally: `make dashboard` (or the `dashboard` service in `docker-compose.yml`) uses
whatever `DATABASE_URL` is in `.env`, falling back to the SQLite file.

## Storage budget (Neon free tier, 1 GB per project)

Neon's pricing page (neon.com/pricing, read on 3 Nov 2026): "Free is capped at 1 GB of
Postgres storage per project and 20 GB total across all projects", and going over blocks
writes.

- Load + official forecast: ~50 k rows each.
- Weather: ~250 k rows (~40 MB).
- Model forecasts: 24 rows per issue (a re-issue adds its own).
- Their weather inputs: 5 cities × 24 h = 120 rows per issue, ~44 k rows a year.
- Official forecast revisions: ~50 k rows from the first copy, then ~24 a day (each new
  target day) plus any revision.
- Model exports: 3.4 MB × ≤5 versions.
- MLflow tables: small.

Roughly 100–150 MB after a year, comfortably inside the tier.
`SELECT pg_size_pretty(pg_database_size(current_database()));` shows the real number.

## Things to know

- GitHub disables scheduled workflows after 60 days without a push. Any commit re-arms them.
- GitHub's cron is best-effort: here it ran the daily jobs 4–8 hours late and the hourly
  ingest only ~4–5 times a day (README §16.1). That is why an external scheduler starts
  them, and why the automatic forecast / monitoring runs are guarded to once per UTC day.
  The ingest jobs are idempotent and incremental (each re-reads an overlap window: 48 h of
  load, 7 days of archived weather), so a re-run does not duplicate rows and a late or
  failed run mostly delays data: gaps inside the overlap window are repaired on the next
  run. Holes older than the overlap window are not repaired automatically. Training,
  `bootstrap` and retrain runs are not idempotent: each one creates a new model version.
- The drift thresholds and retrain margins are CLI/config knobs
  (`src/monitoring/retrain.py`); manual `monitor-retrain` runs with `check-only` or
  `force` + `dry_run` let you rehearse without touching the champion.
