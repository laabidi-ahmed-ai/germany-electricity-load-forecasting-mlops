# Cloud setup — running the loop with your PC off

Everything runs on **GitHub Actions** (scheduler + compute) against one **free cloud
Postgres**. Total cost: €0/month. README §14 describes the idea; this page is the
operator's runbook.

## How persistence works (the one design decision that matters)

GitHub runners are ephemeral: whatever a job writes to disk is gone when it ends.
Three things must survive between runs, and all three live in the cloud Postgres:

| What | Where | Why |
|---|---|---|
| Data (load, official forecast, weather, model forecasts, monitoring events) | our tables (`src/data/db.py`) | idempotent upserts, TimescaleDB hypertables when available |
| **The champion model** | `model_artifacts` table: zipped MLflow pyfunc bundle (~3.4 MB) + features / horizon / train window / CV metrics, one row flagged `is_champion` | `registry.promote()` exports it; `registry.load_champion(engine)` loads it. The forecast job, the API and the monitor need **only `DATABASE_URL`**. Rollback = `--promote <old version>`. Old exports are pruned (last 5 kept, champion never). |
| MLflow lineage (runs, params, metrics, model versions, aliases) | MLflow's own tables in the same Postgres (`MLFLOW_TRACKING_URI` = `DATABASE_URL`) | browse it locally with `mlflow ui --backend-store-uri "<DATABASE_URL>"`. MLflow *artifact files* stay on the runner — they are not needed, the DB export is the serving copy. |

Rejected alternative: an S3-compatible object store for MLflow artifacts (Cloudflare R2 /
Backblaze B2 / Supabase Storage). It works, but adds an account, four more secrets and
boto3 for a 3 MB file that fits in a table row.

## Workflows

| Workflow | Trigger | What it does | Install |
|---|---|---|---|
| `ingest.yml` | hourly `:15` UTC + manual (`incremental` / `backfill`) | SMARD load + official day-ahead forecast, Open-Meteo weather, ENTSO-E if token → DB | `requirements/ingest.txt` only (~1 min) |
| `forecast.yml` | daily 05:40 UTC + manual | incremental ingest → champion (from DB) → 24 h day-ahead forecast → `load_forecast_model` | full project |
| `monitor_retrain.yml` | daily 06:00 UTC + manual (`auto` / `check-only` / `force`, `dry_run`) | performance vs actuals & official forecast, Evidently drift, triggers → champion/challenger → promote only if it wins | full project |
| `bootstrap.yml` | manual, once | backfill 2021-03-01→now → features → train + CV → register & export champion → first forecast → monitoring check | full project |
| `ci.yml` | push / PR | ruff + pytest (integration tests excluded) | dev extras |

All jobs fail fast with a clear error if `DATABASE_URL` is missing. ENTSO-E is optional
everywhere: an empty `ENTSOE_API_TOKEN` means "skip ENTSO-E", never a failure.

## Checklist (what you do once)

1. **Create a free Postgres.** [Neon](https://neon.tech) (0.5 GB, no card) or
   [Supabase](https://supabase.com) (500 MB). Copy the connection string; it looks like
   `postgresql://user:password@host/dbname?sslmode=require`. Either form is accepted
   (`postgresql://` is routed to the psycopg 3 driver automatically). If you want
   MLflow's tables in a separate database, create a second one and use it as
   `MLFLOW_TRACKING_URI`.
2. **Push the repository to GitHub** (public repo = unlimited Actions minutes; private =
   2,000 min/month, the hourly job uses ~30 min/day of that).
3. **Add the secrets** under *Settings → Secrets and variables → Actions*:
   - `DATABASE_URL` — required
   - `ENTSOE_API_TOKEN` — optional, add it whenever the token arrives
   - `MLFLOW_TRACKING_URI` — optional (defaults to `DATABASE_URL`)
4. **Run `bootstrap`** (*Actions → bootstrap → Run workflow*, defaults). ~10 minutes:
   backfills 5½ years, trains, promotes the champion, writes the first forecast, runs the
   first monitoring check. Re-running is safe.
5. **Confirm** in the Actions logs: `germany-load-day-ahead@champion = v1 (...)` and
   `stored 24 forecast rows`. From then on the three scheduled workflows keep the
   system alive. (Optionally trigger `ingest` and `forecast` manually once to see them
   green before the first cron.)

Local development is unchanged: SQLite data DB + SQLite MLflow by default. To point
your laptop at the cloud, put the same `DATABASE_URL` in `.env`.

## Storage budget (Neon free tier, 0.5 GB)

Load + official forecast: ~50 k rows each; weather: ~250 k rows (~40 MB); model
forecasts: 24 rows/day; model exports: 3.4 MB × ≤5 versions; MLflow tables: small.
Roughly 100 MB after a year — comfortably inside the tier.

## Things to know

- GitHub disables scheduled workflows after 60 days without a push. Any commit re-arms them.
- Cron fires can be a few minutes late under load; the jobs are idempotent and the
  forecast job refreshes actuals itself, so late runs are harmless.
- The drift thresholds and retrain margins are CLI/config knobs
  (`src/monitoring/retrain.py`); manual `monitor-retrain` runs with `check-only` or
  `force` + `dry_run` let you rehearse without touching the champion.
