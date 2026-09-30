# Germany Electricity Load Forecasting — An End-to-End MLOps System

> A production-style machine-learning system that forecasts **Germany's national electricity demand (load)** for the next 24–48 hours. It continuously ingests fresh grid and weather data, benchmarks itself against the **official grid-operator day-ahead forecast**, and **retrains itself** when data drift or performance decay is detected — all on a fully free-tier, cloud-scheduled stack.

---

## Table of Contents

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
16. [Roadmap](#16-roadmap)
17. [Glossary](#17-glossary)
18. [License, sources & acknowledgements](#18-license-sources--acknowledgements)

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

- **Every hour** — a scheduled job pulls the newest electricity-load figures (SMARD / ENTSO-E), the official day-ahead forecast, and the latest weather (Open-Meteo), and stores them in a cloud database. This is the continuous data collection that runs unattended.
- **Every morning** — a batch job produces the day-ahead forecast: expected load for each hour of the next 24 hours. It stores the forecast alongside the official grid-operator forecast for the same hours, so the two sit side by side.
- **On demand** — a serving API returns the latest forecast instantly.
- **Every day, looking back** — because yesterday's *actual* load has now been published, a job computes the model's error and the official forecast's error over the same period and updates the accuracy history.
- **Continuously** — a monitor watches for data drift (incoming data moving away from the training distribution) and rising error. If either crosses a calibrated threshold, it triggers a retraining job: a fresh *challenger* model is trained on the latest data and promoted to production **only if it beats the current *champion*** on a held-out out-of-time window — with the previous model retained for rollback.
- **Always visible** — a live dashboard shows forecast vs. actuals, model accuracy vs. the official forecast, drift signals, and a timeline of model versions and retraining events.

---

## 4. What makes this project different

Load forecasting is a well-studied problem; the distinguishing factor here is the engineering around it.

1. **It benchmarks against the official professional forecast.** The grid operators publish not only the *actual* load but also their own *day-ahead load forecast*. This system measures itself head-to-head against that professional benchmark on live data — a credible, honest yardstick that most forecasting projects never use because they don't incorporate it.
2. **It operates over time.** Continuous ingestion, a growing history, and real seasonal drift make the monitoring and retraining loop authentic rather than simulated.
3. **It self-heals with guardrails.** Drift/error-triggered retraining with champion–challenger promotion is fully wired and enforces an out-of-time holdout and a minimum-improvement margin, so it cannot promote a worse model or thrash.

---

## 5. System architecture

```mermaid
flowchart TD
    subgraph Sources["External data sources (free)"]
        A1["SMARD (Bundesnetzagentur) + ENTSO-E<br/>(actual load + official day-ahead forecast)"]
        A2["Open-Meteo<br/>(weather: temp, wind, solar, cloud)"]
    end

    subgraph Ingestion["Ingestion (scheduled, cloud)"]
        B1["Hourly ingestion job<br/>(GitHub Actions cron)"]
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
| **Data ingestion** | Scheduled pulls of load + official forecast (SMARD / ENTSO-E) and weather (Open-Meteo) into the time-series DB, with multi-year historical backfill. | `entsoe-py`, `requests`, GitHub Actions |
| **Storage** | Postgres as the single source of truth for data, the model registry, and model artifacts; idempotent upserts. | Postgres |
| **Feature engineering** | Lag, rolling, calendar/holiday, and weather features, computed leakage-safe on a complete hourly index; horizon-aware selection for day-ahead validity. | `pandas`, `holidays` |
| **Training & experimentation** | Naive baselines → LightGBM; strict time-series cross-validation; every run logged. | LightGBM, MLflow |
| **Evaluation** | MAE/RMSE/MAPE, error sliced by hour/weekday/season, and out-of-sample comparison against the official forecast. | `numpy`, `pandas` |
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
- **ENTSO-E Transparency Platform** — the European TSO data platform (REST API with a free security token). Used as a second source for actual load and the official day-ahead forecast, and for broader European coverage.
- **Open-Meteo** — free weather API (no key) for historical and forecast temperature, wind, solar radiation, and cloud cover — the physical drivers of demand.

### Geography

Electricity data is reported by **grid zones, not cities**, so there is no per-city ("Berlin", "Dortmund") series.

- **Primary target:** the **DE-LU bidding zone** (Germany + Luxembourg, merged since 2018) — Germany's total national load.
- **Future extension:** the **four German control areas**, each run by a different TSO — **50Hertz** (east), **Amprion** (west), **TenneT DE** (north–south), **TransnetBW** (south-west) — forecast separately, mirroring real TSO operations.

Weather is sampled at **city level** (several major German cities) as *input features*; the prediction target is always the zone's total load.

---

## 8. Modeling approach

Day-ahead load forecasting is a **supervised** problem. The model progresses from simple to sophisticated so each step is justified by measured improvement:

1. **Naive baselines** — "same hour yesterday" and "same hour last week" (seasonal naive). These set the bar every other model must beat.
2. **Gradient boosting** — **LightGBM** on lag, calendar, and weather features. This is a genuinely strong baseline for load forecasting.

**Horizon-aware features.** For an honest day-ahead forecast, a feature may only use information available at forecast-issue time. The model therefore uses lags of 24h and longer, calendar features, and weather (available as a forecast), and deliberately excludes features that depend on the most recent, still-unobserved hours — which keeps the *load information* on par with the official day-ahead forecast. One caveat: the batch job forecasts the 24 hours after the latest published actual (lead times of 1–24 h, with a same-morning weather forecast), whereas the official forecast is issued the previous day for the whole calendar day (roughly 12–36 h ahead); the comparison is therefore favourable to the model on lead time and weather freshness.

**Validation.** Strict **time-series cross-validation** (expanding window) — never random splits, which would leak the future into the past. Early stopping is done on a chronological tail of the training data, never on the evaluation fold.

**Metrics.** MAE, RMSE, MAPE, error sliced by hour/weekday/season, and — the signature metric — **error relative to the official day-ahead forecast.**

---

## 9. Monitoring, drift & automated retraining

Automated retraining here means a disciplined, guarded pipeline — not online learning:

- **Performance monitoring** — each day, once the real load is published, compute the champion's error and the official forecast's error over the same hours; track both.
- **Data drift detection** — compare recent data (and features) against the training distribution with **Evidently**. Thresholds are calibrated against historical regime shifts rather than left at library defaults, so ordinary year-to-year variation does not trigger false alarms.
- **Trigger** — retraining fires when rolling error crosses a threshold, when the official forecast beats the model by too much, or when drift is detected.
- **Champion / challenger** — a challenger is trained on the latest data and evaluated against the champion on data that arrived *after* the champion's training window (a true out-of-time holdout).
- **Promotion or rollback** — the challenger is promoted **only if it beats the champion by a minimum margin**; the previous model is retained for one-step rollback. Every decision is logged to MLflow and a monitoring-events table.

This loop is why the system is designed to run continuously: a retraining trigger is only meaningful once fresh data has had time to accumulate.

---

## 10. Tech stack

> Every component has a free tier sufficient to run the whole system at no cost.

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Data ingestion | `entsoe-py`, `requests` (SMARD, ENTSO-E, Open-Meteo) |
| Storage | Postgres (Neon / Supabase free tier), TimescaleDB hypertables when the extension is available; SQLite for local dev |
| Feature engineering | pandas, `holidays` |
| Modeling | LightGBM |
| Experiment tracking + registry | MLflow (Postgres-backed) |
| Serving | FastAPI (real-time) + scheduled batch job |
| Containerization | Docker / docker-compose |
| Orchestration / scheduling | GitHub Actions (cron) |
| Monitoring / drift | Evidently |
| CI/CD | GitHub Actions + pytest + ruff |
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
├── Dockerfile                     # serving image (API + batch job)
├── docker-compose.yml             # local Postgres + API + dashboard
├── .env.example                   # template for secrets (no real values)
├── requirements/                  # lean per-job dependency sets (e.g. ingest)
├── docs/                          # design notes (e.g. cloud setup)
├── notebooks/                     # exploration only, never the source of truth
├── .github/workflows/
│   ├── ci.yml                     # tests + lint on push
│   ├── ingest.yml                 # hourly ingestion (cron)
│   ├── forecast.yml               # daily day-ahead forecast (cron)
│   ├── monitor_retrain.yml        # daily monitoring + drift-triggered retrain
│   └── bootstrap.yml              # one-time cloud seed (backfill → train → register)
├── config/
│   └── settings.py                # typed configuration
├── src/
│   ├── data/                      # ingestion & DB access (smard, entsoe, weather, db, ingest)
│   ├── features/                  # build_features, horizons (leakage-safe, horizon-aware)
│   ├── models/                    # baselines, train, evaluate, registry, tracking
│   ├── serving/                   # api (FastAPI), forecast, batch_forecast
│   └── monitoring/                # drift, performance, retrain
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
ENTSOE_API_TOKEN=your_entsoe_security_token_here   # optional
# DATABASE_URL=postgresql://user:password@host:5432/dbname   # unset = local SQLite
OPEN_METEO_BASE_URL=https://api.open-meteo.com/v1
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db     # the cloud jobs point this at DATABASE_URL
BIDDING_ZONE=DE_LU                                 # ENTSO-E bidding zone
SMARD_REGION=DE                                    # SMARD region code
DATA_START_DATE=2021-03-01                         # post-COVID cutoff
LOG_LEVEL=INFO
```

- `DATABASE_URL` — connection string from a free Postgres provider (Neon / Supabase).
- `ENTSOE_API_TOKEN` — optional. The system runs fully on SMARD without it; the ENTSO-E second source activates automatically once it is set.

---

## 13. Running it locally

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/germany-electricity-load-forecasting-mlops.git
cd germany-electricity-load-forecasting-mlops

# 2. Install (Python 3.11+)
pip install -e ".[dev]"          # or: make setup

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

- **Scheduling** lives in **GitHub Actions** (cron): `ingest.yml` hourly, `forecast.yml` and `monitor_retrain.yml` daily. GitHub spins up a temporary runner, executes the job, and shuts it down — free (unlimited for public repos).
- **The database is cloud-hosted** (Neon / Supabase free tier), and the model registry and artifacts persist there too, so every job works on a fresh runner with only `DATABASE_URL`.
- **The dashboard** is hosted on Streamlit Community Cloud with a public URL.

A one-time `bootstrap` workflow seeds the cloud database (backfill → features → train → register the first champion → first forecast); the hourly and daily schedules then run continuously.

---

## 15. Results

Measured with expanding-window time-series cross-validation and out-of-sample day-ahead backtests over 2021–2026 (German national load, DE-LU):

| Model | MAE (MW) | MAPE (%) |
|---|---:|---:|
| Seasonal naive (same hour, 24h ago) | 3,929 | 7.43 |
| Seasonal naive (same hour, 168h ago) | 2,937 | 5.53 |
| **LightGBM (day-ahead)** | **1,286** | **2.43** |
| Official TSO day-ahead forecast | — | 3.85 |

- LightGBM reduces MAE by ~56% vs. the best naive baseline and wins on every cross-validation fold.
- In an out-of-sample day-ahead backtest, the model beats the **official grid-operator forecast in 12 of 12 monthly windows** (2.43% vs. 3.85% MAPE). Read with the caveat in §8: the model's window starts right after the latest published actual and uses fresher weather than a true previous-day issue would have.

*Live accuracy against fresh actuals, uptime, and retraining events accumulate as the deployed system runs, and are surfaced on the dashboard.*

---

## 16. Roadmap

Planned technical extensions, in rough order:

- **Deep learning** — LSTM/GRU and a Temporal Fusion Transformer, compared rigorously against the LightGBM baseline.
- **Multi-region forecasting** — the four German control areas (50Hertz, Amprion, TenneT, TransnetBW) forecast separately.
- **Probabilistic forecasting** — quantile / prediction intervals with pinball-loss evaluation, since operators care about uncertainty, not just a point estimate.
- **Net-load forecasting** — demand minus wind and solar generation, the harder and increasingly important modern problem.
- **Orchestration upgrade** — a dedicated orchestrator (Prefect/Airflow) and an optional feature store.
- **Explainability & governance** — SHAP attributions and model cards.

---

## 17. Glossary

- **Load** — electricity demand / consumption, in megawatts (MW).
- **Bidding zone** — a market area with a single electricity price (Germany + Luxembourg = `DE_LU`).
- **Control area / TSO** — the region operated by one Transmission System Operator (Germany has four).
- **Day-ahead forecast** — a forecast made today for each hour of tomorrow.
- **Drift** — when incoming data's statistical pattern moves away from the training data.
- **Champion / challenger** — the model currently in production vs. a newly trained candidate.
- **Backfill** — loading historical data in bulk to train the first model.
- **Net load** — demand minus variable renewable generation (wind + solar).

---

## 18. License, sources & acknowledgements

**License:** MIT (see `LICENSE`).

**Data sources:**
- SMARD (Bundesnetzagentur) — https://www.smard.de
- ENTSO-E Transparency Platform — https://transparency.entsoe.eu
- Open-Meteo — https://open-meteo.com

**Acknowledgements:** Data provided by SMARD, ENTSO-E, and Open-Meteo. This is an independent project and is not affiliated with any grid operator.
