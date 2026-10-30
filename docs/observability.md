# Observability: job metrics and alerts in Datadog

GitHub e-mails you when a job **crashes**. It says nothing when a job **does not run**,
runs hours late, writes nothing, or keeps running while the model gets worse — and those
were the problems this system actually had (README §16). So each job sends a handful of
metrics to Datadog when it ends, and Datadog monitors alert on them.

There is no Datadog agent: GitHub runners are temporary machines, so `src/observability.py`
POSTs the metrics to Datadog's HTTP API once per run. Without `DD_API_KEY` nothing is sent.

## What is sent

Every metric starts with `germany_load.` and carries the tags `job:<ingest|forecast|monitor>`
and `runner:github` (or `runner:local` from a laptop — filter on `runner:github`).

| Metric | Sent by | Meaning |
|---|---|---|
| `job.run` (tag `status:ok|failed`) | every job | 1 per run |
| `job.failed` | every job | 1 if the run failed (exception or a failed ingest step) |
| `job.duration_seconds` | every job | run time |
| `ingest.rows` (tag `step`) | ingest | rows written per step (`smard`, `smard_forecast`, `weather_archive`, `weather_forecast`) |
| `ingest.step_failed` (tag `step`) | ingest | 1 if that step failed |
| `data.freshness_hours` | ingest | hours since the newest actual load hour |
| `forecast.age_hours` | ingest | hours since the latest model forecast was issued (a watchdog for the forecast job) |
| `forecast.issued`, `forecast.hours`, `forecast.data_age_hours` | forecast | a forecast was stored; how many hours; how old the newest actual was at issue time |
| `forecast.skipped` | forecast | a backup run found today's forecast already issued |
| `model.mape`, `official.mape`, `model.improvement_pct` (tag `window:7d|30d`) | monitor | live accuracy head-to-head |
| `drift.share` | monitor | share of monitored columns that drifted |
| `drift.unavailable` | monitor | 1 if the drift check could not run (it never fails the job, so this is the only signal) |
| `trigger.fired` (tag `trigger:error|official|drift`) | monitor | 1 if that retrain trigger fired |
| `retrain.ran`, `retrain.promoted` | monitor | a challenger was trained / promoted |
| `monitor.skipped` | monitor | a backup run found today's pass already recorded |

## Setup (once, ~15 minutes)

### 1. Datadog account
Activate the Datadog offer from the GitHub Student Developer Pack and sign up. When asked
for a **region / site**, pick **EU** (`datadoghq.eu`) — the data and the jobs' database are
in Europe. Whatever you choose, note the address you land on: `app.datadoghq.eu` means the
site is `datadoghq.eu`; `app.datadoghq.com` means `datadoghq.com`; `us5.datadoghq.com`
means `us5.datadoghq.com`. Skip the agent-installation wizard: this project does not use
an agent.

### 2. API key
Datadog → your avatar (bottom left) → **Organization Settings → API Keys → + New Key**,
name it `github-actions`, copy the key. (An *API* key, not an *Application* key.)

### 3. GitHub
Repository → **Settings → Secrets and variables → Actions**:
- **Secrets** tab → *New repository secret*: `DD_API_KEY` = the key.
- **Variables** tab → *New repository variable*: `DD_SITE` = your site (e.g. `datadoghq.eu`).
  It is not a secret; if you leave it out, `datadoghq.eu` is assumed.

### 4. Check that metrics arrive
Actions → **ingest** → *Run workflow*. In the log of the *Ingest* step, the last lines
should include `sent N metric(s) to Datadog`. In Datadog → **Metrics → Explorer**, type
`germany_load.` — the metrics appear within a minute or two. (`could not send ... 403`
in the log = wrong key or wrong `DD_SITE`.)

## Monitors (alerts)

Datadog → **Monitors → + New Monitor → Metric**. For each monitor below: switch the query
to the code editor (the `</>` button next to the query), paste the query, set the
thresholds, the *no data* option, a name, and in the message add `@your-email@example.com`
so the alert e-mails you. Create them in this order.

| # | Name | Query | Alert when | No data |
|---|---|---|---|---|
| 1 | Ingest stopped | `sum(last_3h):sum:germany_load.job.run{job:ingest,runner:github} < 1` | below 1 | **Notify if data is missing for more than 180 minutes** |
| 2 | Actual load data is stale | `min(last_2h):max:germany_load.data.freshness_hours{runner:github} > 6` | above 6 (warning 4.5) | off |
| 3 | No forecast today | `min(last_1h):max:germany_load.forecast.age_hours{runner:github} > 25` | above 25 (warning 24.5) | off |
| 4 | Model error above retrain threshold | `max(last_1d):max:germany_load.model.mape{window:7d,runner:github} > 3.5` | above 3.5 (warning 3.2) | off |
| 5 | A job failed | `max(last_1h):max:germany_load.job.failed{runner:github} by {job} > 0.5` | above 0.5 — choose **Multi Alert** (one alert per job) | off |
| 6 | Drift check unavailable | `max(last_1d):max:germany_load.drift.unavailable{runner:github} > 0.5` | above 0.5 | off |

Why these six:
1. **Ingest stopped** — the external scheduler or the token broke; nothing else notices.
   No metrics at all for 3 h is itself the alert (the "no data" option).
2. **Stale data** — ingest runs but SMARD stopped publishing (or a step keeps failing).
   Normal freshness is 1–3 h, because SMARD publishes with a delay.
3. **No forecast today** — the forecast is issued around 05:42 UTC; if the next one is
   missed, the ingest run at 06:20 UTC reports an age of ~24.6 h (warning) and the 07:20
   run ~25.6 h (alert). The hourly ingest reports this age, so the alert works even when
   the forecast job never started.
4. **Model error** — the same 3.5 % 7-day MAPE the retrain trigger uses; the retrain loop
   reacts on its own, this tells a human it is happening.
5. **A job failed** — overlaps with GitHub's e-mails, but puts every signal in one place.
6. **Drift check unavailable** — the drift check is allowed to fail without failing the
   monitor job (so accuracy is still scored), which also means a broken drift check stays
   green on GitHub. This happened once: a new pandas major broke Evidently's import
   (README §16.2).

Use **Test Notifications** (bottom of the monitor page) once to check the e-mail arrives.

## Dashboard

Datadog → **Dashboards → New Dashboard** (name: *Germany load forecasting*). Add these
widgets (*Add widgets* → type → paste the query in the code editor):

| Widget | Type | Query |
|---|---|---|
| Model vs official MAPE (7 d) | Timeseries | `max:germany_load.model.mape{window:7d}` and `max:germany_load.official.mape{window:7d}` |
| Model vs official MAPE (30 d) | Timeseries | same with `window:30d` |
| Data freshness (h) | Query Value | `max:germany_load.data.freshness_hours{runner:github}` |
| Forecast age (h) | Query Value | `max:germany_load.forecast.age_hours{runner:github}` |
| Drift share | Timeseries | `max:germany_load.drift.share{runner:github}` |
| Rows ingested per step | Timeseries (bars) | `sum:germany_load.ingest.rows{runner:github} by {step}` |
| Job duration (s) | Timeseries | `max:germany_load.job.duration_seconds{runner:github} by {job}` |
| Failed jobs | Timeseries (bars) | `sum:germany_load.job.failed{runner:github} by {job}` |
| Monitor status | Monitor Summary | the six monitors above |

## Turning it off
Delete the `DD_API_KEY` secret: the jobs keep working and simply stop sending metrics.
