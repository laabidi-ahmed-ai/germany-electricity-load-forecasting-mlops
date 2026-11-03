# Germany Electricity Load Forecasting — An End-to-End MLOps System

**Live dashboard:** https://germany-electricity-load-forecasting.streamlit.app

> A production-style machine-learning system that forecasts **Germany's national electricity demand (load)** for every hour of the next calendar day (D+1). It ingests fresh grid and weather data every hour, benchmarks itself every day against the **official grid-operator day-ahead forecast**, and **retrains itself** when the daily check finds data drift or performance decay — all on a fully free-tier, cloud-scheduled stack.

## Headline result

In a one-year retrospective backtest (2 Nov 2025 – 2 Nov 2026, 8,760 hours; trained once before the window, never retrained), the D+1 model had **25.2 % lower MAE than the official grid-operator day-ahead forecast** on the same target days: 1,515 MW vs 2,025 MW.

```mermaid
xychart-beta
    title "One-year backtest, same D+1 hours: MAE in MW (lower is better)"
    x-axis ["Official TSO day-ahead forecast", "LightGBM (D+1)"]
    y-axis "MAE (MW)" 0 --> 2500
    bar [2025, 1515]
```

**Main limitation:** the backtest used *measured* weather for the past year, whereas a real forecast — ours live, and the official one — can only use weather *forecasts*. The live gap is therefore expected to be smaller; the live record on the dashboard is the final judge. Cross-validation against baselines and the full numbers: §15.

---

## Table of Contents

- [Headline result](#headline-result)

1. [Overview](#1-overview)
2. [The business problem, in plain terms](#2-the-business-problem-in-plain-terms)
3. [What the system does (the operating loop)](#3-what-the-system-does-the-operating-loop)
4. [What makes this project different](#4-what-makes-this-project-different)
5. [System architecture](#5-system-architecture)
6. [The MLOps lifecycle](#6-the-mlops-lifecycle)
7. [Data sources & geography](#7-data-sources--geography)
8. [Modeling approach](#8-modeling-approach)
9. [Monitoring, drift & automated retraining](#9-monitoring-drift--automated-retraining)
10. [Tech stack](#10-tech-stack)
11. [Repository structure](#11-repository-structure)
12. [Configuration & secrets](#12-configuration--secrets)
13. [Running it locally](#13-running-it-locally)
14. [Running it in the cloud](#14-running-it-in-the-cloud)
15. [Results](#15-results)
16. [Operating it: problems found in production and how they were fixed](#16-operating-it-problems-found-in-production-and-how-they-were-fixed)
17. [Roadmap](#17-roadmap)
18. [Glossary](#18-glossary)
19. [License, sources & acknowledgements](#19-license-sources--acknowledgements)

---

## 1. Overview

Electricity demand forecasting is a genuine, high-value industrial problem: grid operators (TSOs), utilities, and energy traders across Europe run whole teams on it, because a better day-ahead load forecast directly lowers cost, fuel use, and emissions while keeping the grid stable.

This system tackles that problem end to end and, crucially, **operates over time** rather than running once. It ingests real grid and weather data on a schedule, forecasts Germany's national load a day ahead, measures its own accuracy against the official grid-operator forecast as fresh ground truth arrives, and retrains automatically when it detects drift or performance decay — keeping the previous model for one-step rollback. The underlying capability — "forecast how much of something will be needed so the business can plan" — transfers directly to demand and capacity forecasting in retail, logistics, telecoms, and finance.

---

## 2. The business problem, in plain terms

Electricity is unusual: **it cannot be stored at grid scale.** The grid must produce, every second, almost exactly as much power as everyone is consuming at that same second. Too little → blackouts. Too much → waste and instability.

But power plants cannot be switched on instantly — they need to be told *hours ahead*, sometimes a full day, how much to produce. So grid operators are forced to make a bet about the future: **"How much electricity will the country use tomorrow, hour by hour?"** They plan all their generation around that bet.

Demand is always moving. It rises on cold winter evenings (heating) and hot summer afternoons (cooling), during working hours, and on weekdays; it falls at night, on weekends, and on public holidays. Weather, the clock, the calendar, and economic activity all push it up and down.

**The task of this system is exactly that bet:** given the recent demand history, the weather forecast, and the calendar, predict Germany's electricity load for the coming day. A better forecast means less wasted fuel, lower cost, fewer emissions, and a more stable grid.

---

## 3. What the system does (the operating loop)

Once deployed, the system runs on its own:

- **Every hour** — a scheduled job pulls the newest electricity-load figures (SMARD), the official day-ahead forecast, and the latest weather (Open-Meteo), and stores them in a cloud database. This is the hourly data collection that runs unattended.
- **Every morning (05:40 UTC)** — a batch job produces the day-ahead forecast: expected load for each hour of **tomorrow, 00:00–24:00 Berlin time** — exactly the hours the official grid-operator forecast covers and the electricity market trades at noon. It stores the forecast alongside the official forecast for the same hours, so the two sit side by side. The monitoring job starts as soon as the forecast finishes.
- **On demand** — a serving API returns the latest forecast instantly.
- **Every day, looking back** — because yesterday's *actual* load has now been published, a job computes the model's error and the official forecast's error over the same period and updates the accuracy history.
- **Every day, after the forecast** — a monitor checks for data drift (incoming data moving away from the training distribution) and rising error. If either crosses a calibrated threshold, it triggers a retraining job: a fresh *challenger* model is trained on the latest data and promoted to production **only if it beats the current *champion*** on a held-out out-of-time window — with the previous model retained for rollback.
- **Always visible** — a live dashboard shows forecast vs. actuals, model accuracy vs. the official forecast, drift signals, and a timeline of model versions and retraining events.

---

## 4. What makes this project different

Load forecasting is a well-studied problem; the distinguishing factor here is the engineering around it.

1. **It benchmarks against the official professional forecast.** The grid operators publish not only the *actual* load but also their own *day-ahead load forecast*. This system measures itself head-to-head against that professional benchmark on live data — a credible, honest yardstick that most forecasting projects never use because they don't incorporate it.
2. **It operates over time.** Hourly ingestion, daily monitoring, a growing history, and real seasonal drift make the monitoring and retraining loop authentic rather than simulated.
3. **It self-heals with guardrails.** Drift/error-triggered retraining with champion–challenger promotion is fully wired and enforces an out-of-time holdout and a minimum-improvement margin, which reduces the risk of promoting a worse model or of thrashing between models.

---

## 5. System architecture

```mermaid
flowchart TD
    subgraph Sources["External data sources (free)"]
        A1["SMARD (Bundesnetzagentur)<br/>(actual load + official day-ahead forecast)"]
        A2["Open-Meteo<br/>(weather: temp, wind, solar, cloud)"]
    end

    subgraph Ingestion["Ingestion (scheduled, cloud)"]
        B1["Hourly ingestion job<br/>(GitHub Actions)"]
    end

    subgraph Storage["Storage layer (cloud)"]
        C1[("Time-series DB<br/>Postgres")]
        C2[("Model registry + artifacts<br/>(in Postgres)")]
    end

    subgraph Features["Feature engineering"]
        D1["Lag, rolling, calendar & weather features<br/>(leakage-safe, horizon-aware)"]
    end

    subgraph Training["Training & experimentation"]
        E1["Baselines → LightGBM"]
        E2["Experiment tracking + registry<br/>(MLflow)"]
    end

    subgraph Serving["Serving"]
        F1["Batch: day-ahead forecast job"]
        F2["Real-time: FastAPI endpoint"]
    end

    subgraph Ops["Monitoring & retraining"]
        G1["Accuracy vs. actuals & vs. official forecast"]
        G2["Drift detection (Evidently)"]
        G3["Trigger → champion/challenger → promote/rollback"]
    end

    subgraph UI["Presentation"]
        H1["Live dashboard (Streamlit)"]
    end

    A1 --> B1
    A2 --> B1
    B1 --> C1
    C1 --> D1
    D1 --> E1
    E1 --> E2
    E2 --> C2
    C2 --> F1
    C2 --> F2
    F1 --> G1
    F2 --> G1
    C1 --> G2
    G1 --> G3
    G2 --> G3
    G3 --> E1
    G1 --> H1
    G2 --> H1
    F1 --> H1
```

**Design principle:** every arrow is a scheduled job or an API call, and every component runs on a free tier or on GitHub's runners — so the system is independent of any always-on personal machine.

---

## 6. The MLOps lifecycle

The project is organized around the canonical MLOps loop; each stage maps to a folder under `src/`.

| Stage | What happens | Key tools |
|---|---|---|
| **Problem & metrics** | Target = day-ahead hourly load; ML metrics = MAE / RMSE / MAPE; the signature metric is error relative to the official forecast. | — |
| **Data ingestion** | Scheduled pulls of load + official forecast (SMARD) and weather (Open-Meteo) into the time-series DB, with multi-year historical backfill. | `requests`, GitHub Actions |
| **Storage** | Postgres as the single source of truth for data, the model registry, and model artifacts; idempotent upserts. | Postgres |
| **Feature engineering** | Lag, rolling, calendar/holiday, and weather features, computed leakage-safe on a complete hourly index; horizon-aware selection for day-ahead validity. | `pandas`, `holidays` |
| **Training & experimentation** | Naive baselines → LightGBM; strict time-series cross-validation; every run logged. | LightGBM, MLflow |
| **Evaluation** | MAE/RMSE/MAPE, error sliced by hour of day, and out-of-sample comparison against the official forecast. | `numpy`, `pandas` |
| **Registry & versioning** | Champion/challenger via MLflow aliases; promotion rules; one-step rollback. | MLflow Model Registry |
| **Serving** | Batch day-ahead job + real-time FastAPI endpoint; Dockerized. | FastAPI, Docker |
| **Monitoring** | Accuracy vs. fresh actuals and vs. the official forecast; data drift; logging. | Evidently |
| **Retraining & feedback** | Drift/error triggers → challenger → out-of-time validation → promote or roll back → log the decision. | GitHub Actions, MLflow |
| **CI/CD** | Tests and linting on every push; scheduled pipelines as workflows. | GitHub Actions, pytest, ruff |
| **Presentation** | Public live dashboard. | Streamlit |

---

## 7. Data sources & geography

### Sources (all free)

- **SMARD (Bundesnetzagentur)** — the German Federal Network Agency's official data platform. Provides actual national load and the official day-ahead load forecast, hourly, back to 2015, with no API key required. This is the primary source for German load.
- **Open-Meteo** — free weather API (no key) for historical and forecast temperature, wind, solar radiation, and cloud cover — the physical drivers of demand.

### Geography

Electricity data is reported by **grid zones, not cities**, so there is no per-city ("Berlin", "Dortmund") series.

- **Primary target:** Germany's national load as published by SMARD for **region `DE`** (Germany only). The official day-ahead forecast used as the benchmark comes from the same SMARD region, so the comparison is like-for-like in geography. (The electricity *market* is the DE-LU bidding zone, Germany + Luxembourg; Luxembourg's load is not in this target.)
- **Future extension:** the **four German control areas**, each run by a different TSO — **50Hertz** (east), **Amprion** (west), **TenneT DE** (north–south), **TransnetBW** (south-west) — forecast separately, mirroring real TSO operations.

Weather is sampled at **city level** (several major German cities) as *input features*; the prediction target is always the zone's total load.

---

## 8. Modeling approach

Day-ahead load forecasting is a **supervised** problem. The model progresses from simple to sophisticated so each step is justified by measured improvement:

1. **Naive baselines** — "same hour 2 days ago" and "same hour last week" (seasonal naive) for the D+1 champion. These set the bar every other model must beat. ("Same hour yesterday", used for the first champion, leaks for D+1 — §16.5.)
2. **Gradient boosting** — **LightGBM** on lag, calendar, and weather features. This is a genuinely strong baseline for load forecasting.

**Horizon-aware features.** For an honest day-ahead forecast, a feature may only use information available at forecast-issue time. The model therefore uses load lags at or beyond its horizon's cutoff — 48 h and longer for the D+1 champion (`load_lag_24` is dropped), 24 h and longer for the first champion (v1) — plus calendar features, and weather (available as a forecast), and deliberately excludes features that depend on the most recent, still-unobserved hours — which keeps the *load information* on par with the official day-ahead forecast. One caveat: the batch job forecasts the 24 hours after the latest published actual (lead times of 1–24 h, with a same-morning weather forecast), whereas the official forecast is issued the previous day for the whole calendar day (roughly 12–36 h ahead); the comparison is therefore favourable to the model on lead time and weather freshness. **That caveat applied to the first champion; the task mismatch is now removed:** since November 2026 the champion uses the calendar-day horizon `d_plus_1` — every morning it forecasts all of tomorrow, 00:00–24:00 Berlin time, the same task (same target day) as the official forecast. The weather information is still not equal: the one-year backtest uses measured weather rather than the forecasts available at issue time (§15), and live, the two forecasts are not built from the same weather forecast. Tomorrow's hours are 19–43 h after the latest actual, so its information cutoff is 48 h: it keeps `load_lag_48` and `load_lag_168` but drops `load_lag_24`, and it is compared with honest baselines ("same hour 2 days ago", "same hour last week"). See §16.5 for the evaluation and the decision.

**Validation.** Strict **time-series cross-validation** (expanding window) — never random splits, which would leak the future into the past. Early stopping is done on a chronological tail of the training data (the last 10 %), never on the evaluation fold. The tail only picks the number of trees: the model is then refit on every training row with that number, so the most recent months are in the model. CV folds, the backtest, the challenger and the final model all use this same procedure, so the reported numbers describe how the deployed model is trained.

**Metrics.** MAE, RMSE, MAPE, error sliced by hour of day, and — the signature metric — **error relative to the official day-ahead forecast.**

---

## 9. Monitoring, drift & automated retraining

Automated retraining here means a disciplined, guarded pipeline — not online learning:

- **Performance monitoring** — each day, once the real load is published, compute the champion's error and the official forecast's error over the same hours; track both. Every issued forecast is kept. An hour is scored with the latest issue made by its deadline, 12:00 Berlin on the day before for D+1, when the day-ahead auction closes; a later issue is never scored. The share of hours with an on-time forecast is tracked too (§16.9).
- **Data drift detection** — compare recent data (and features) against the training distribution with **Evidently**. Thresholds are calibrated against historical regime shifts rather than left at library defaults, so ordinary year-to-year variation does not trigger false alarms.
- **Trigger** — retraining fires when rolling error crosses a threshold, when the official forecast beats the model by too much, or when drift is detected.
- **Champion / challenger** — a challenger is trained on the latest data and evaluated against the champion on data that arrived *after* the champion's training window (a true out-of-time holdout).
- **Promotion or rollback** — the challenger is promoted **only if it beats the champion by a minimum margin**; the previous model is retained for one-step rollback. Every decision is logged to MLflow and a monitoring-events table.

This loop is why the system is designed to keep running (hourly ingestion, a daily check), because a retraining trigger is only meaningful once fresh data has had time to accumulate.

---

## 10. Tech stack

> Every component has a free tier sufficient to run the whole system at no cost.

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Data ingestion | `requests` (SMARD, Open-Meteo) |
| Storage | Postgres (Neon / Supabase free tier); SQLite for local dev |
| Schema migrations | Alembic (jobs check the schema version, never migrate) |
| Feature engineering | pandas, `holidays` |
| Modeling | LightGBM |
| Experiment tracking + registry | MLflow (Postgres-backed) |
| Serving | FastAPI (real-time) + scheduled batch job |
| Containerization | Docker / docker-compose |
| Orchestration / scheduling | GitHub Actions, started on time by an external scheduler (cron-job.org); GitHub cron as backup |
| Monitoring / drift | Evidently |
| Observability / alerting | Datadog (job metrics over its HTTP API, monitors, dashboard) |
| CI/CD | GitHub Actions + pytest + ruff |
| Dependencies | ranges in `pyproject.toml`, exact versions locked with `uv` (`requirements/constraints.txt`) |
| Dashboard | Streamlit |
| Config / secrets | pydantic-settings, `.env`, GitHub Secrets |

---

## 11. Repository structure

```
germany-electricity-load-forecasting-mlops/
├── README.md
├── LICENSE
├── pyproject.toml                 # dependencies & tooling config
├── Makefile                       # common commands (setup, test, data, train, serve, …)
├── alembic.ini                    # schema migrations (scripts in src/data/migrations/)
├── Dockerfile                     # serving image (API + batch job)
├── docker-compose.yml             # local Postgres + API + dashboard
├── .env.example                   # template for secrets (no real values)
├── requirements/                  # lock file (constraints.txt) + lean per-job sets (ingest)
├── docs/                          # cloud setup, operations runbook, observability
├── notebooks/                     # exploration only, never the source of truth
├── .github/workflows/
│   ├── ci.yml                     # tests + lint on push
│   ├── ingest.yml                 # hourly ingestion
│   ├── forecast.yml               # daily day-ahead forecast
│   ├── monitor_retrain.yml        # monitoring + drift-triggered retrain, after each forecast
│   └── bootstrap.yml              # one-time cloud seed (backfill → train → register)
├── config/
│   └── settings.py                # typed configuration
├── src/
│   ├── data/                      # ingestion & DB access (smard, weather, db, ingest, migrations/)
│   ├── features/                  # build_features, horizons (leakage-safe, horizon-aware)
│   ├── models/                    # baselines, train, evaluate, registry, tracking
│   ├── serving/                   # api (FastAPI), forecast, batch_forecast
│   ├── monitoring/                # drift, performance, retrain
│   └── observability.py           # optional job metrics -> Datadog
├── dashboard/
│   ├── app.py                     # Streamlit dashboard
│   ├── queries.py                 # read-only queries behind it
│   └── requirements.txt           # lean deps for Streamlit Community Cloud
└── tests/                         # pytest unit + integration tests
```

---

## 12. Configuration & secrets

Secrets are **never** committed. For local use, copy `.env.example` to `.env` (gitignored); for the cloud jobs, add the same keys as **GitHub Secrets**.

```dotenv
# .env.example (abridged)
# DATABASE_URL=postgresql://user:password@host:5432/dbname   # unset = local SQLite
OPEN_METEO_BASE_URL=https://api.open-meteo.com/v1
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db     # the cloud jobs point this at DATABASE_URL
SMARD_REGION=DE                                    # SMARD region code
DATA_START_DATE=2021-03-01                         # post-COVID cutoff
LOG_LEVEL=INFO
```

Every data source is keyless. The secrets are:

- `DATABASE_URL` — connection string from a free Postgres provider (Neon / Supabase). Required.
- `MLFLOW_TRACKING_URI` — optional; the cloud jobs fall back to `DATABASE_URL`.
- `DD_API_KEY` — optional; job metrics are sent to Datadog only when it is set (`docs/observability.md`).
- The external scheduler's token — a fine-grained GitHub token stored in the scheduler (cron-job.org), not in the repository; it can start, re-run and cancel this repository's workflows, but cannot read secrets or change code. A leaked token can still dispatch any workflow (including `bootstrap`, which retrains and promotes a new champion) and disable workflows, which also stops the GitHub cron backups — revoke it immediately (`docs/operations.md`).

---

## 13. Running it locally

```bash
# 1. Clone and enter
git clone https://github.com/laabidi-ahmed-ai/germany-electricity-load-forecasting-mlops.git
cd germany-electricity-load-forecasting-mlops

# 2. Install (Python 3.11+)
make setup        # creates .venv and installs the locked versions into it
# (every other make target runs .venv's Python, so run `make setup` first; without make:
#  pip install -e ".[dev]" -c requirements/constraints.txt, then call the python -m commands)

# 3. Configure (SQLite is used by default while DATABASE_URL stays commented out)
cp .env.example .env

# 4. Backfill historical data, build features, train, promote the first champion
make data
make features
make train
make register     # export the trained model as the serving champion

# 5. Forecast and serve
make forecast     # batch day-ahead forecast -> database
make serve        # FastAPI at http://localhost:8000
make dashboard    # Streamlit at http://localhost:8501
```

---

## 14. Running it in the cloud

The system is designed so **no personal machine needs to stay on**:

- **Scheduling** lives in **GitHub Actions**: `ingest.yml` hourly, `forecast.yml` daily at 05:40 UTC, and `monitor_retrain.yml` right after each forecast (a `workflow_run` chain). GitHub spins up a temporary runner, executes the job, and shuts it down — free (unlimited for public repos). An external scheduler starts the hourly and daily jobs on time through GitHub's `workflow_dispatch` API, with GitHub's own cron kept as a backup — see §16 for why, and `docs/operations.md` for the setup.
- **The database is cloud-hosted** (Neon / Supabase free tier), and the model registry and artifacts persist there too, so every job works on a fresh runner with only `DATABASE_URL`.
- **The dashboard** is hosted on Streamlit Community Cloud with a public URL.

A one-time `bootstrap` workflow seeds the cloud database (backfill → features → train → register the first champion → first forecast); from then on the hourly and daily schedules keep it running.

---

## 15. Results

### Current champion: calendar day D+1 (the official forecast's task)

The champion is **v3**: the D+1 LightGBM refit on its full training window after early stopping (§16.7), trained on data up to 2 Nov 2026 08:00 UTC and promoted on 2 Nov 2026 (MLflow run `6371e4785d82483bb0690a134e2c892b`).

12-fold expanding-window cross-validation from v3's own training run (one-month folds, 7 Nov 2025 – 2 Nov 2026), 19 features:

| Model | MAE (MW) | MAPE (%) |
|---|---:|---:|
| Seasonal naive (same hour, 48 h ago) | 6,238 | 11.95 |
| Seasonal naive (same hour, 168 h ago) | 2,827 | 5.33 |
| **LightGBM (D+1)** | **1,496** | **2.84** |

One-year out-of-sample backtest against the official forecast, with the champion's training code (trained on data up to 2 Nov 2025, never retrained, then scored on every hour from 2 Nov 2025 to 2 Nov 2026 — 8,760 hours; recorded from evaluation run B, §16.7):

| | MAE (MW) | MAPE (%) |
|---|---:|---:|
| **LightGBM (D+1)** | **1,515** | **2.87** |
| Official TSO day-ahead forecast | 2,025 | 3.82 |

- LightGBM beats the best baseline by 47.1 % (MAE) and wins on every one of the 12 folds.
- On the official forecast's task (same target days) it had **25.2 % lower MAE in a one-year retrospective backtest using measured weather**, and a lower daily MAE on **255 of 366 days**. These are UTC days: the window ends at the latest published hour, not at midnight, so its first and last days are partial.
- Caveat: the backtest uses *measured* weather for the past year, whereas live forecasts (ours and the official one) must use weather *forecasts* — so the live gap is expected to be smaller. The live record (dashboard) is the final judge.

### First champion (v1): the 24 hours after the latest actual

Measured with expanding-window time-series cross-validation and out-of-sample day-ahead backtests over 2021–2026 (German national load, SMARD region DE):

| Model | MAE (MW) | MAPE (%) |
|---|---:|---:|
| Seasonal naive (same hour, 24h ago) | 3,929 | 7.43 |
| Seasonal naive (same hour, 168h ago) | 2,937 | 5.53 |
| **LightGBM (day-ahead)** | **1,286** | **2.43** |
| Official TSO day-ahead forecast | — | 3.85 |

- LightGBM reduces MAE by ~56% vs. the best naive baseline and wins on every cross-validation fold.
- In an out-of-sample day-ahead backtest, the model beats the **official grid-operator forecast in 12 of 12 monthly windows** (2.43% vs. 3.85% MAPE). Read with the caveat in §8: the model's window starts right after the latest published actual and uses fresher weather than a true previous-day issue would have.

**Live, on fresh actuals, v1** (as of 15 October 2026, four weeks after go-live; the dashboard updates these daily):

| Window | Hours scored | Model MAPE | Official MAPE | Model MAE vs official |
|---|---:|---:|---:|---:|
| Last 7 days | 163 | 3.16 % | 3.17 % | 2.5 % lower |
| Last 30 days | 636 | 2.81 % | 3.24 % | 13.5 % lower |

Live accuracy is worse than the backtest (2.81 % vs 2.43 % MAPE over 30 days) and the last week is essentially a tie: the first half of October was hard for both forecasts (see §16.3). The live record is still short; it is the number to watch, not the backtest.

---

## 16. Operating it: problems found in production and how they were fixed

The system went live on 17 September 2026. Some problems only show up when a system runs unattended for weeks, so the approach was: **observe first, measure, then fix** — as long as a problem could not corrupt data. The ingest jobs are incremental and idempotent (each re-reads an overlap window before the last stored hour — 48 h of load, 7 days of archived weather — and upserts), so a re-run does not duplicate rows and a late or failed run mostly delays data: gaps inside the overlap window are repaired on the next run. Holes older than the overlap window are not repaired automatically. Training, `bootstrap` and retrain runs are not idempotent: each one creates a new model version. That made it safe to leave the system alone and collect real evidence before changing anything.

### 16.1 GitHub's scheduler ran the daily jobs 4–8 hours late

**What we observed (18 Sep – 14 Oct, measured from the Actions run history).** GitHub's `schedule` trigger is best-effort, and on the free runners it was consistently late:

| Job | Scheduled | What actually happened |
|---|---|---|
| `forecast` | daily 05:40 UTC | ran on all 27 days, but **4.0–7.8 h late** (median 5.6 h; typically 11:00–13:30 UTC) |
| `monitor-retrain` | daily 06:00 UTC | ran on all 27 days, 4.1–8.1 h late |
| `ingest` | 24 runs/day | **4.5 runs/day on average** (2–7); median gap 5.1 h, longest 10.5 h |

**Why it was left running for four weeks.** Nothing was lost: the forecast job ingests fresh data itself before forecasting, and ingestion catches up on every run. Waiting turned "cron is sometimes late" into numbers — it was *always* late, by a stable 4–8 hours, rather than occasionally skipped — which decided the fix (a platform limitation, not a bug in our code, so no amount of retrying inside GitHub helps).

**How it affected the work.**

1. **The forecast arrived too late to be useful.** Germany's day-ahead electricity auction closes at 12:00 Berlin time. A forecast meant to inform that decision was published after the auction closed on 26 of 27 days.
2. **Coverage and scoring.** Each forecast covers the 24 hours after the latest published actual. Because the start time wandered by up to ~3 hours from one day to the next, consecutive forecast windows sometimes overlapped and sometimes left hours with no forecast at all (those hours are simply never scored). Where windows overlapped, the later forecast replaced the earlier one for the same hour — a shorter lead time that slightly flatters the live score.
3. **Freshness.** With ~4–5 ingests a day, the actuals on the dashboard were often 5–10 hours old.

**The fix.**

- An **external scheduler** (cron-job.org, free) calls GitHub's `workflow_dispatch` API on time: `ingest` every hour, `forecast` at 05:40 UTC. It uses a fine-grained token that can only run this repository's workflows. GitHub's cron stays as a backup.
- **`monitor-retrain` is chained** to the forecast (`workflow_run`), so it starts the moment the forecast finishes instead of guessing a time slot. A later cron remains as a backup.
- **Once-per-day guards** (`batch_forecast --once-per-day`, `retrain --once-per-day`): a late backup run sees that today's forecast / monitoring pass already exists and exits without doing anything. Without them, the backup would re-issue the forecast at midday and bring back problem 2. A manual run with `force` still re-forecasts on purpose.

**Resolved later (§16.5):** forecasting the 24 hours *after the latest actual* is not the same as forecasting *tomorrow's calendar day*, which is what the official forecast and the auction use. The calendar-day (D+1) champion removes the remaining overlap/gap effect and makes the official comparison the same task (same target day); the weather information is still not equal (§8).

**Resolved later (§16.9):** problem 2's replacement is gone too. Every issue is now kept, and an hour is scored with the latest issue made by its deadline: a late re-issue can no longer flatter the live score, and a forecast that arrives after the auction no longer counts at all.

### 16.2 Incident log

| Date (2026) | What happened | Root cause | Fix |
|---|---|---|---|
| 18 Sep | `forecast` failed with `NoDataError` (`load_lag_24` missing) | SMARD published one hour of load late, leaving a hole in recent history; the job correctly refused to forecast from incomplete data | Now **bridges gaps of up to 3 missing hours** by interpolation for serving only (never written back as actuals) and logs which hours it filled; longer gaps still stop the forecast, because they mean a data problem, not a delay |
| 18 Sep – 14 Oct | Scheduled jobs 4–8 h late, ingest ~4–5×/day | GitHub's best-effort cron (§16.1) | External trigger + chained monitor + once-per-day guards |
| 1 Oct | `ingest` #74 failed: "Could not parse SQLAlchemy URL" | The `DATABASE_URL` secret was pasted in TOML form (with quotes) and with the dashboard's read-only user | Secret replaced with the writer URL, no quotes; jobs and the public dashboard now use separate database roles (write vs read-only) |
| 1 Oct | Dashboard error after a deploy: `module 'dashboard.queries' has no attribute 'WINDOWS_DAYS'` | Streamlit Cloud reloaded the new `app.py` but kept the previous version of an imported module in memory | Rebooting the app; after any deploy that changes `dashboard/`, reboot once and check it loads |
| 5 Oct | `ingest` #95 failed on the Open-Meteo forecast request | Open-Meteo returned a body that was not JSON four times in ~7 s; the old retry waits (1.5 s, 2.3 s, 3.4 s) were too short to ride out the hiccup. The next run succeeded and backfilled, so no data was lost | Retries now wait 3 s, 9 s and 27 s, also retry HTTP 429 (rate limited), and the error names the HTTP status, content type and start of the body instead of only "Expecting value" |
| 30 Oct (detected) | Drift check silently unavailable; CI #17 failed on `test_compute_drift_with_real_evidently` | Every job installs its dependencies fresh, and `pandas>=2.1` had no upper bound: pip picked up pandas 3, whose removed internals Evidently still imports. The monitor job is designed to survive a drift error (accuracy is still scored), so it stayed green; only CI's fresh install exposed it. A laptop with an older environment passed | `pandas>=2.1,<3` in every dependency file until Evidently supports pandas 3, and a `drift.unavailable` metric with its own Datadog alert, so a broken drift check can no longer look green. The next CI run then failed on Python 3.12 only: NumPy 2.5 (which needs Python 3.12+) deprecates something pandas 2.x still does, so `numpy<2.5` joined the cap - production runs Python 3.11 and could not install NumPy 2.5. The caps fixed these two packages; the lock file (§16.6) fixes the cause |
| 2 Nov | `ingest` failed from 13:20 UTC for hours: Open-Meteo returned 503 "The service is overloaded" | A provider outage. In `forecast`, the same ingest runs as the first step, so a failure there would have skipped the forecast even though SMARD data was fresh and the target day's weather was already stored (each ingest stores 3 days of weather forecast) | `forecast` now runs the forecast step even when only that refresh failed, and the job still ends red. The forecast's own checks remain the safety net: it refuses a too-old latest actual or gaps over 3 h, and fetches missing target-day weather itself, failing if Open-Meteo cannot supply it |

### 16.3 The retraining loop, exercised on live data

The guarded retraining path ran for real in October:

- **5–8 October:** the 7-day MAPE rose above the 3.5 % trigger (up to 3.94 %). **11 October:** half of the monitored columns drifted, which fired the drift trigger (this autumn it is mostly weather columns — wind speed, cloud cover, humidity — that move away from the reference period).
- Each time a **challenger was trained** and compared with the champion on hours neither had trained on. It did not beat the champion by the required 3 % margin, so the **champion (v1) was kept** and the decision was logged to MLflow and the events table.
- Both forecasts were worse in that period: the official forecast's 7-day MAPE reached 4.75 %, against 3.94 % for the model. A week that is hard for everyone, rather than a model that has gone stale, is exactly the case the anti-churn margin exists for: retraining on it would not have helped.

### 16.4 Alerting: from "GitHub e-mails me on a crash" to monitors on what the jobs do

The scheduler problem in §16.1 ran for four weeks and was found by reading the Actions history by hand: no job ever *failed*, so nothing alerted. The same blind spot covers a job that never starts, an ingest that runs but writes nothing, data that stops arriving, or a model that slowly degrades. Crash e-mails cannot see any of those.

So each job now reports a few metrics to **Datadog** when it ends — rows written per step, data freshness, forecast age, job duration and outcome, live model and official MAPE, drift share, trigger states and retrain decisions (`src/observability.py`). There is no agent (runners are temporary); the job POSTs to Datadog's HTTP API, and a Datadog outage is logged but can never fail a job. Six monitors alert by e-mail:

| Monitor | Catches |
|---|---|
| Ingest stopped (no run for 3 h) | the external scheduler or its token broke |
| Actual load data stale (> 6 h) | SMARD stopped publishing, or an ingest step keeps failing |
| No forecast today (forecast older than 25 h) | the forecast job never ran — reported by the *hourly ingest*, so it works even when the forecast job cannot report anything |
| 7-day MAPE above the champion's limit (4.0 % for D+1) | the model is degrading (the retrain loop reacts; this tells a human) |
| A job failed | any failure, per job, in the same place as everything else |
| Drift check unavailable | a drift check that errors but, by design, does not fail the monitor job |

Setup and the exact monitor queries: `docs/observability.md`. Without the `DD_API_KEY` secret nothing is sent, so local runs, tests and forks are unaffected.

### 16.5 From "the next 24 hours" to the official task: the calendar-day (D+1) champion

**The problem.** The first champion forecast the 24 hours after the latest published actual. The official forecast covers *tomorrow's calendar day*. Close, but not the same task — every "the model beats the official forecast" claim needed a caveat (§8).

**What was built (Batch C).** A second horizon, `d_plus_1`: every morning, all of tomorrow 00:00–24:00 Berlin time (23 or 25 hours on clock-change days). Tomorrow's hours are 19–43 h after the latest actual, so `load_lag_24` would be leakage and is dropped; the information cutoff is 48 h, leaving 4–5 h of slack for a late SMARD publication, and the forecast refuses to run if the latest actual is too old to cover the whole day. Serving, retraining, drift monitoring and the backtest all follow the **champion's** horizon. Building it surfaced two details worth knowing: "same hour yesterday" is itself a leaking *baseline* for D+1 (replaced by "same hour 2 days ago"), and computing "tomorrow" as "midnight + 24 h" picks the wrong day on the autumn clock change (fixed with calendar dates, and tested).

**How it was decided.** Changing the task is a product decision, not something the automatic retraining should do, so the model was first trained in an *evaluate-only* run (`bootstrap` with `horizon=d_plus_1`, `promote` off) and judged on evidence: does it beat honest baselines (cross-validation) and, above all, the official forecast on the same task (one-year backtest)? Its error is naturally higher than v1's (2.88 % vs 2.44 % CV MAPE) because it forecasts further ahead with less information — comparing those two numbers would be comparing a 100 m and a 200 m time. Against the official forecast on the same hours it had 24.6 % lower MAE in a one-year retrospective backtest using measured weather, so it was promoted; rollback was `python -m src.models.registry --promote 1`. Its successor, the current champion, is the same D+1 model refit on its full training window (§16.7): 25.2 % lower MAE in a one-year retrospective backtest using measured weather (§15).

**Consequences.** The retraining error trigger is now per horizon — about 1.4× the cross-validated MAPE: 3.5 % for the rolling model, 4.0 % for D+1 — so the harder task does not cause pointless retrains or alerts (the Datadog monitor follows the same limit).

**What the backtest taught.** The model's worst day of the year was 4 June 2026 (10 % vs the official 2 %): Corpus Christi, a public holiday in only some German states, which the model deliberately ignores. The official forecast's worst stretch was 7–12 January 2026 (9–10 %), where the model stayed at 2–6 %. Regional holidays are now on the roadmap.

### 16.6 Locked dependencies: a new release can no longer change production

**The problem.** Both October CI failures (§16.2) had the same cause: `pyproject.toml` only gave *ranges* (`pandas>=2.1`), and every job installed fresh, so each run got whatever was newest on PyPI that day. Production on Monday and on Tuesday could run different library versions with no commit in between. Capping pandas and NumPy fixed those two packages, but the project installs about 150 packages in total (MLflow and Evidently alone bring dozens), and any of them could do the same.

**How it was observed.** Not by production: by CI, which also installs fresh and failed first. The drift check had already been silently broken in production (§16.2).

**The fix (Batch D).** Two layers, each with one job:
- `pyproject.toml` keeps the **ranges** — what the code is known to work with.
- `requirements/constraints.txt` is the **lock**: the exact version of every package, direct and indirect, generated from those ranges by `uv` (`make lock`). It is *universal* — one file for Python 3.11+ on Linux, macOS and Windows, with a separate pin where a package differs by Python version.

Every install uses it: the cloud jobs (lean ingest and full), CI, and `make setup` all run `pip install … -c requirements/constraints.txt`; the Docker image runs `pip install -r requirements/constraints.txt`, then `pip install --no-deps . && pip check`. CI therefore tests exactly the versions production runs. Offline tests (`tests/test_lock.py`) fail if a dependency is added to `pyproject.toml` without re-locking, or if a bound changes so that a pinned version no longer satisfies it.

**The trade-off.** Upgrades no longer arrive by themselves, so they are now a deliberate step: `make lock-upgrade`, run the tests, push, and let CI confirm — one reviewed commit, easy to revert (`docs/operations.md`). The Streamlit dashboard keeps its own short range file (`dashboard/requirements.txt`): it only reinstalls when it is redeployed, and a broken deploy shows immediately on the page.

### 16.7 Refitting after early stopping (measured)

**The problem.** Early stopping held out the most recent 10 % of the training window — about 6–7 months of data — only to pick the number of trees. The model that was kept had been trained without it, so the months closest to what it forecasts never reached the model.

**The change.** The tail still picks the number of trees; the model is then refit on every row with that number. CV folds, the backtest, the challenger and the final model all use the same procedure, so every reported number describes how the deployed model is trained.

**How it was evaluated.** Changing how the model is trained changes every number above, so it was measured before it was merged: two *evaluate-only* bootstraps on the same data, one from `main` (A, no refit) and one from the branch (B, refit). Both selected the same number of trees (`best_iteration` 587), confirming they saw the same data.

| | A: no refit | B: refit |
|---|---:|---:|
| CV mean MAE (12 folds) | 1,522 MW | 1,499 MW (−1.5 %) |
| CV mean MAPE | 2.89 % | 2.84 % |
| Backtest MAE (official: 2,025 MW in both runs) | 1,538 MW | 1,515 MW |
| Backtest: lower MAE than the official forecast | 24.0 % | 25.2 % |
| Backtest: days with lower daily MAE (of 366 UTC days) | 252 | 255 |

B was better on 9 of the 12 paired folds, and the gains were largest in the most recent folds (fold 11: −90 MW) — where the refit adds the data closest to the validation month.

The 1,499 MW is run B's; v3's own training run (§15: 1,496 MW, `best_iteration` 595) came a few hours later with slightly more data.

**The decision.** Merged and promoted. The gain is modest, and on its own not conclusive: a one-sided sign test on the 12 folds gives p ≈ 0.07. But cross-validation and the full-year backtest agree on its direction and size.

**The record.** MLflow runs: A `17229c7a7dbb4edda01cd328457b7c64`, B `c3bf57f0f56148a9a188e2a66b88d9c8`. The backtest numbers are recorded only in this section: the backtest is printed to the GitHub Actions log, and those logs expire. The procedure for the next change of this kind is in `docs/operations.md` (*Evaluate a model-code change against production*).

### 16.8 Leftover TimescaleDB hypertables, converted to plain tables

**What was found.** On 2 Nov 2026, the first `alembic check` against production reported four indexes no code declared: `<table>_timestamp_utc_idx` on `timestamp_utc DESC`, one per time-series table. They came from TimescaleDB. `init_db` made the four tables hypertables when the extension was available, until commit c7e7058 (1 Oct 2026) removed that setup without converting the tables back. Production kept 297/297/297/7 chunks (weekly, since March 2021), plus indexes that duplicate the primary keys. The baseline migration (`0001`) recorded them as they were.

**The fix (revision `0002`).** Each hypertable is copied into a plain table, which then replaces it. One transaction does it all:
- the tables are locked against writes, while reads continue;
- any row-count difference aborts the conversion;
- the grants are copied, so the dashboard's read-only role keeps working;
- each primary key is restored under its usual name.

The duplicate indexes are dropped. Dropping ~900 chunks in one transaction can exceed the Postgres lock table, so a one-transaction-per-table fallback is built in (`-x per_table=true`). The conversion was rehearsed on a Neon branch with a compute the size of production's before it ran on `main`. Runbook: `docs/operations.md` (*Convert the leftover hypertables*).

| Rehearsal (`rehearse-0002`) | |
|---|---|
| `max_locks_per_transaction` / `max_connections` | 64 / 901 |
| Locks held by the conversion before commit | 13,651 |
| Upgrade duration | 27.5 s |
| Rows: `load_actual` / `load_forecast_official` / `weather_hourly` / `load_forecast_model` | 49,758 / 49,775 / 249,120 / 1,163 |
| Mode used | one transaction |

### 16.9 Every issued forecast kept; only on-time forecasts scored

**The problem.** The live record could not be audited "as of issue time". Three histories were overwritten:
- `load_forecast_model` was keyed by hour and model version, so a re-issue replaced the earlier issue of the same hours;
- `load_forecast_official` keeps one value per hour, so revisions of the official forecast were invisible;
- the weather a forecast used lived only in `weather_hourly`, where the archive later overwrites forecast values.

Scoring also took the latest issue of each hour, even one made after the hour had started (§16.1, problem 2).

**The fix (revision `0003`).**
- `load_forecast_model` has `issued_at` in its primary key, so a re-issue adds rows. Each row also stores the model's `horizon`, filled in for past forecasts from the model exports. A check refuses the migration, before any change, if a version has no export.
- `load_forecast_official_revision` keeps every value the official forecast has had. It started as a copy of the official forecast table; each ingest adds only the hours whose value is new or changed. That copy is a snapshot of the values held at the upgrade, stamped with when each row was last written rather than when the value was first seen, and earlier revisions were never kept: the revision history is complete only from the upgrade onward.
- `load_forecast_model_weather` keeps the per-city weather each stored forecast used, written in the same transaction as the forecast.

**The scoring rule.** Each horizon has an issue deadline. For D+1 it is 12:00 Berlin on the day before the target day, when the day-ahead auction closes: 11:00 UTC in winter, 10:00 UTC in summer, the same for every hour of a 23- or 25-hour clock-change day. For the rolling horizons it is the target hour itself. Either way, an hour that has already started when the forecast is issued counts as late. An hour is scored with the latest issue made by its deadline; a late issue is stored and shown, never scored. Each window also reports its hours as on time, late (only late issues) or missing (no issue). The on-time share is in the performance report, the monitoring events, the dashboard and Datadog (`forecast.on_time_pct`), and the forecast job reports its margin to the deadline (`forecast.deadline_margin_hours`).

One consequence for the record before the fix: where a later issue had overwritten an earlier one, only the later one survives. For the first champion (v1), whose deadline is the target hour, some of those hours were issued after the hour started. They now count as late and leave the live score.

The migration was rehearsed on a Neon branch before it ran on `main`, including a real ingest and a real forecast on the branch. Runbook: `docs/operations.md` (*Keep every issued forecast*).

| Rehearsal (`rehearse-0003`) | |
|---|---|
| Upgrade duration | _TBD_ s |
| `load_forecast_model` rows / issues per version, with backfilled horizon | _TBD_ |
| `load_forecast_official_revision` seeded (= `load_forecast_official`) | _TBD_ rows |
| Real ingest on the branch: official hours new or changed | _TBD_ |
| `batch_forecast --dry-run`: target window, deadline margin | _TBD_ → _TBD_, _TBD_ h |
| Real forecast on the branch: new issue's rows / weather inputs; earlier issue kept | _TBD_ / _TBD_ (5 cities × _TBD_ h); _TBD_ |
| Downgrade → upgrade: `load_forecast_model` rows | _TBD_ → _TBD_ → `alembic check` clean |
| On-time share, 7 d / 30 d; hours now late | _TBD_ % / _TBD_ %; _TBD_ |
| Database size before / after | _TBD_ / _TBD_ |


---

## 17. Roadmap

Planned technical extensions, in rough order:

- **Regional holidays** — a "share of the population on holiday" feature, so days like Corpus Christi (a holiday in some states only) stop being the model's worst days (§16.5).

- **Deep learning** — LSTM/GRU and a Temporal Fusion Transformer, compared rigorously against the LightGBM baseline.
- **Multi-region forecasting** — the four German control areas (50Hertz, Amprion, TenneT, TransnetBW) forecast separately.
- **Probabilistic forecasting** — quantile / prediction intervals with pinball-loss evaluation, since operators care about uncertainty, not just a point estimate.
- **Net-load forecasting** — demand minus wind and solar generation, the harder and increasingly important modern problem.
- **Orchestration upgrade** — a dedicated orchestrator (Prefect/Airflow) and an optional feature store.
- **Explainability & governance** — SHAP attributions and model cards.

---

## 18. Glossary

- **Load** — electricity demand / consumption, in megawatts (MW).
- **Bidding zone** — a market area with a single electricity price (Germany + Luxembourg = `DE_LU`); the forecast target here is Germany alone (SMARD region `DE`).
- **Control area / TSO** — the region operated by one Transmission System Operator (Germany has four).
- **Day-ahead forecast** — a forecast made today for each hour of tomorrow.
- **Drift** — when incoming data's statistical pattern moves away from the training data.
- **Champion / challenger** — the model currently in production vs. a newly trained candidate.
- **Backfill** — loading historical data in bulk to train the first model.
- **Net load** — demand minus variable renewable generation (wind + solar).

---

## 19. License, sources & acknowledgements

**License:** MIT (see `LICENSE`).

**Data sources:**
- SMARD (Bundesnetzagentur) — https://www.smard.de
- Open-Meteo — https://open-meteo.com

**Acknowledgements:** Data provided by SMARD and Open-Meteo. This is an independent project and is not affiliated with any grid operator.
