# Germany Electricity Load Forecasting — An End-to-End MLOps System

> A production-style machine-learning system that forecasts **Germany's national electricity demand (load)** for the next 24–48 hours, continuously collects fresh grid and weather data, monitors its own accuracy against the **official grid-operator forecast**, and **retrains itself** when the world drifts.
>
> Built as a solo portfolio project to demonstrate the full MLOps lifecycle — from live data ingestion to automated retraining — on a **€0/month free-tier stack**.

---

## Table of Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [The business problem, in plain terms](#2-the-business-problem-in-plain-terms)
3. [What the system actually does (the living loop)](#3-what-the-system-actually-does-the-living-loop)
4. [What makes this project different](#4-what-makes-this-project-different)
5. [System architecture](#5-system-architecture)
6. [The MLOps lifecycle, phase by phase](#6-the-mlops-lifecycle-phase-by-phase)
7. [Data sources & geography](#7-data-sources--geography)
8. [Modeling approach](#8-modeling-approach)
9. [Monitoring, drift & automated retraining](#9-monitoring-drift--automated-retraining)
10. [Tech stack (and why each piece)](#10-tech-stack-and-why-each-piece)
11. [Repository structure](#11-repository-structure)
12. [Configuration & secrets](#12-configuration--secrets)
13. [How to run it locally](#13-how-to-run-it-locally)
14. [How it runs in the cloud (free, PC-off)](#14-how-it-runs-in-the-cloud-free-pc-off)
15. [Phased build plan](#15-phased-build-plan)
16. [9-month roadmap](#16-9-month-roadmap)
17. [Results](#17-results)
18. [Skills demonstrated](#18-skills-demonstrated)
19. [Glossary](#19-glossary)
20. [License, sources & acknowledgements](#20-license-sources--acknowledgements)

---

## 1. Why this project exists

Most machine-learning portfolio projects are a notebook trained once on a static Kaggle CSV. They never run again, never see new data, and never demonstrate the part of the job that actually matters in industry: **operating a model over time.**

This project is deliberately the opposite. It is a *living* system. It has been ingesting real European grid data and making daily forecasts since it went live, so it can show the things that only become real when data keeps arriving and the world keeps changing — genuine data drift, accuracy tracked against fresh ground truth, and retraining that fires for real reasons.

The domain — **electricity demand forecasting** — was chosen because it is a genuine, high-value industrial problem that large European enterprises (grid operators / TSOs, utilities, energy traders, consultancies) employ whole teams to solve. It is also a *transferable* skill: "forecast how much of something will be needed so the business can plan" is the same core capability retailers, logistics firms, and banks hire for.

**Target roles this project is built to signal for:** Machine Learning Engineer, MLOps Engineer, Data Scientist (forecasting / time-series), ML Engineering internships (PFE) across the EU.

---

## 2. The business problem, in plain terms

Electricity is unusual: **it cannot be stored at grid scale.** The grid must produce, every second, almost exactly as much power as everyone is consuming at that same second. Too little → blackouts. Too much → waste and instability.

But power plants cannot be switched on instantly — they need to be told *hours ahead*, sometimes a full day, how much to produce. So grid operators are forced to make a bet about the future: **"How much electricity will the country use tomorrow, hour by hour?"** They plan all their generation around that bet.

Demand is always moving. It rises on cold winter evenings (heating) and hot summer afternoons (cooling), during working hours, and on weekdays; it falls at night, on weekends, and on public holidays. Weather, the clock, the calendar, and economic activity all push it up and down.

**The task of this system is exactly that bet:** given the recent demand history, the weather forecast, and the calendar, predict Germany's electricity load for the coming hours and the next day. A better forecast means less wasted fuel, lower cost, fewer emissions, and a more stable grid.

---

## 3. What the system actually does (the living loop)

Once deployed, the system runs on its own:

- **Every hour** — a scheduled job pulls the newest electricity-load figures (from ENTSO-E) and the latest weather (from Open-Meteo) and stores them in a cloud database. This is the *continuous data collection* that runs while nobody is watching.
- **Every morning** — a batch job produces the day-ahead forecast: expected load for each hour of the next 24–48 hours. It saves the forecast **and** fetches the official ENTSO-E day-ahead forecast for the same hours, so the two sit side by side.
- **On demand, all day** — a serving API returns the latest forecast instantly to anyone (a dashboard, a grader, a recruiter) who asks.
- **Every day, looking back** — because yesterday's *actual* load has now been published, a job computes the model's error **and** the official forecast's error on the same period, and updates the accuracy history.
- **Continuously** — a monitor watches for (a) data drift (the incoming data drifting away from what the model was trained on) and (b) rising error. If either crosses a threshold, it automatically triggers a **retraining** job. The new "challenger" model is trained on the latest data, validated against the current "champion," and promoted **only if it wins** — with the old model kept for rollback.
- **Always visible** — a live dashboard shows the forecast vs. actuals, model accuracy vs. the official forecast, drift signals, and a timeline of model versions and retraining events.

That dashboard, with a public URL, is the portfolio centerpiece.

---

## 4. What makes this project different

The base idea (energy load forecasting) is common. The execution is not. Three things set it apart:

1. **It benchmarks against the official professional forecast.** ENTSO-E publishes not only the *actual* load but also the grid operators' own *day-ahead load forecast*. This project measures itself head-to-head against that professional benchmark on live data — a credible, honest yardstick almost no student project uses.
2. **It genuinely operates over time.** Continuous ingestion + a growing history + real seasonal drift means the monitoring and retraining loop is authentic, not simulated. "This has been running in production for N months" is a claim very few portfolios can make.
3. **It self-heals.** Drift-triggered retraining with champion–challenger promotion and rollback is fully wired and actually fires — not just described in a README.

**Later-stage differentiator (roadmap):** *net-load forecasting* — forecasting demand **minus** wind and solar generation — which is the genuinely hard, modern problem grid operators struggle with as renewables grow.

---

## 5. System architecture

```mermaid
flowchart TD
    subgraph Sources["External data sources (free)"]
        A1["ENTSO-E Transparency Platform<br/>(actual load + official forecast)"]
        A2["Open-Meteo<br/>(weather: temp, wind, solar, cloud)"]
    end

    subgraph Ingestion["Ingestion (scheduled, cloud)"]
        B1["Hourly ingestion job<br/>(GitHub Actions cron)"]
    end

    subgraph Storage["Storage layer (cloud, free tier)"]
        C1[("Time-series DB<br/>Postgres / TimescaleDB")]
        C2[("Raw data / model artifacts<br/>object store")]
    end

    subgraph Features["Feature engineering"]
        D1["Lag, rolling, calendar & weather features<br/>(leakage-safe)"]
        D2["Feature definitions / store"]
    end

    subgraph Training["Training & experimentation"]
        E1["Baselines → LightGBM → LSTM/TFT"]
        E2["Experiment tracking + model registry<br/>(MLflow)"]
        E3["Hyperparameter tuning (Optuna)"]
    end

    subgraph Serving["Serving"]
        F1["Batch: day-ahead forecast job"]
        F2["Real-time: FastAPI endpoint"]
    end

    subgraph Ops["Monitoring & retraining"]
        G1["Accuracy vs. actuals & vs. official forecast"]
        G2["Drift detection (Evidently)"]
        G3["Retraining trigger → champion/challenger → promote/rollback"]
    end

    subgraph UI["Presentation"]
        H1["Live dashboard (Streamlit)"]
    end

    A1 --> B1
    A2 --> B1
    B1 --> C1
    B1 --> C2
    C1 --> D1
    D1 --> D2
    D2 --> E1
    E1 --> E2
    E1 --> E3
    E2 --> F1
    E2 --> F2
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

**Design principle:** every arrow is either a scheduled job or an API call, and every component runs on a free tier or on GitHub's runners — so the whole system is independent of any always-on personal machine.

---

## 6. The MLOps lifecycle, phase by phase

This project is organized around the canonical MLOps loop. Each stage maps to a folder in `src/` (see [Repository structure](#11-repository-structure)).

| Stage | What happens | Key tools |
|---|---|---|
| **1. Problem & metrics** | Define the target (day-ahead hourly load), the business metric (forecast error → balancing cost), the offline ML metric (MAE / RMSE / MAPE / pinball loss), and the guardrail (p99 API latency, no accuracy regression on promotion). | — |
| **2. Data ingestion** | Scheduled pulls of load + official forecast (ENTSO-E) and weather (Open-Meteo); written to the time-series DB. Historical backfill of several years for training. | `entsoe-py`, `requests`, GitHub Actions |
| **3. Storage** | Time-series database as the single source of truth; object store for raw snapshots and model artifacts; **data versioning** so each model ties to the exact dataset it trained on. | Postgres/TimescaleDB, DVC |
| **4. Feature engineering** | Lag features (t−1, t−24, t−168), rolling statistics, calendar/holiday flags, weather features. Scalers fit **only on training data** to prevent leakage. Feature definitions kept in one place. | `pandas`, `holidays`, (optional) Feast |
| **5. Training & experimentation** | Baselines → gradient boosting → deep learning. Time-series cross-validation. Every run logged with params and metrics. Hyperparameter search. | LightGBM, PyTorch, MLflow, Optuna |
| **6. Evaluation** | Point metrics + probabilistic metrics (pinball/quantile), significance vs. baseline, error sliced by hour/season/weekday, online-vs-offline comparison. | `scikit-learn`, custom |
| **7. Model registry & versioning** | Champion/challenger stages, promotion rules, rollback. Models are versioned artifacts, not files in a folder. | MLflow Model Registry |
| **8. Serving** | Batch day-ahead forecast job + real-time FastAPI endpoint. Both Dockerized. | FastAPI, Docker |
| **9. Monitoring** | Accuracy vs. fresh actuals, accuracy vs. the official forecast, data drift, latency/health, logging. | Evidently, custom checks |
| **10. Retraining & feedback** | Drift/error triggers → retrain challenger → validate → promote or roll back → log the decision. | GitHub Actions, MLflow |
| **11. CI/CD** | Tests, linting, and build on every push; scheduled jobs as workflows. | GitHub Actions, pytest, ruff |
| **12. Presentation** | Public live dashboard tying it all together. | Streamlit |

---

## 7. Data sources & geography

### Sources (both free)

- **ENTSO-E Transparency Platform** — free public REST API (registration + a one-time security token required). Provides, per European bidding zone / control area: **actual total load**, **day-ahead load forecast** (the official benchmark), generation by source, and prices. Accessed in Python via the `entsoe-py` client.
- **Open-Meteo** — free weather API, **no key required** for non-commercial use. Provides historical and forecast temperature, wind, solar radiation, and cloud cover — the physical drivers of electricity demand.

### Geography — important

Electricity data is reported by **grid zones, not cities.** There is no "Berlin" or "Dortmund" series.

- **Primary target:** the **DE-LU bidding zone** (Germany + Luxembourg, merged since 2018) — Germany's total national load. In `entsoe-py` the country code is `DE_LU`. This is the single, clean forecasting target for the core build.
- **Later (roadmap):** the **four German control areas**, each run by a different TSO — **50Hertz** (east, incl. Berlin), **Amprion** (west, incl. the Ruhr/Dortmund), **TenneT DE** (north–south), **TransnetBW** (south-west). Forecasting these four zones separately ("multi-region forecasting") mirrors how the real TSOs operate and is a strong differentiator.

Weather is still used at **city level** (a few representative German cities) as *input features* — temperature drives demand — but the thing being predicted is always the **zone's total load**.

---

## 8. Modeling approach

Forecasting is a **supervised** learning problem (predict a number, graded against the true number). The project deliberately progresses from simple to sophisticated so each step is justified by measured improvement:

1. **Naive baselines** — "same hour yesterday," "same hour last week," seasonal naive. These set the bar every other model must beat.
2. **Classical / gradient boosting** — **LightGBM** on lag + rolling + calendar + weather features. This is genuinely the strong baseline for load forecasting and is often hard to beat.
3. **Deep learning** — an **LSTM/GRU** sequence model, then a **Temporal Fusion Transformer (TFT)** for multi-horizon forecasting with interpretable attention.
4. **Probabilistic forecasting (roadmap)** — quantile/prediction intervals (not just a point estimate), because real grid operators need uncertainty, and most student projects skip it.

**Validation:** strict **time-series cross-validation** (expanding/rolling window) — never random splits, which would leak the future into the past.

**Metrics:** MAE, RMSE, MAPE for point accuracy; **pinball loss** for probabilistic forecasts; error sliced by hour-of-day, weekday/weekend, and season; and — the signature metric — **error relative to the official ENTSO-E forecast.**

---

## 9. Monitoring, drift & automated retraining

"The machine retrains itself" means a disciplined, automated pipeline — **not** magical online learning. Concretely:

- **Performance monitoring** — each day, once the real load is published, compute the champion model's error on the just-passed period, and the official forecast's error on the same period. Track both over time.
- **Data drift detection** — compare the statistical distribution of recent incoming data (and features) against the training distribution using **Evidently**. A cold snap or seasonal shift shows up here.
- **Trigger** — if error exceeds a threshold, or drift is detected, the retraining pipeline is launched automatically.
- **Champion / challenger** — the pipeline trains a fresh *challenger* on the latest data, then evaluates it fairly against the current *champion* on a held-out recent window.
- **Promotion or rollback** — the challenger is promoted to production **only if it beats the champion** by a meaningful margin. The previous model is retained so a bad promotion can be rolled back in one step.
- **Everything is logged and versioned** — every retraining decision (what triggered it, which model won, which dataset it used) is recorded in MLflow and the run history.

This loop is *why the project needs to run over time*: you cannot demonstrate a retraining trigger that has never had a reason to fire.

---

## 10. Tech stack (and why each piece)

> Everything below has a free tier sufficient for this project. Nothing here requires a paid subscription.

| Concern | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | Standard for ML/MLOps. |
| Data pull | **`entsoe-py`, `requests`** | Clean access to ENTSO-E; keyless HTTP for Open-Meteo. |
| Storage (time series) | **Postgres / TimescaleDB** (Neon or Supabase free tier) | Real database, not CSVs; time-series optimized; cloud-hosted so it survives PC-off. |
| Object storage | **Local `/data` + object store** (e.g. MinIO locally, or provider free tier) | Raw snapshots and model artifacts. |
| Data versioning | **DVC** | Reproducible link between a model and the exact data it trained on. |
| Feature layer | **pandas** (+ optional **Feast**) | Feature engineering; Feast is an optional stretch for a true feature store. |
| Modeling | **LightGBM, PyTorch** (LSTM/TFT), **scikit-learn** | Strong classical baseline + deep learning. |
| Experiment tracking + registry | **MLflow** | Logs runs, params, metrics; model registry with stages + rollback. |
| Hyperparameter tuning | **Optuna** | Efficient search with time-series-aware CV. |
| Serving (real-time) | **FastAPI** | Fast, typed, production-standard API. |
| Serving (batch) | **Scheduled job** | Day-ahead forecast written to DB. |
| Containerization | **Docker / docker-compose** | Reproducible environments; one command to run the stack. |
| Orchestration / scheduling | **GitHub Actions cron** (core) → **Prefect/Airflow** (roadmap) | Free scheduled jobs with no server; upgradeable to a real orchestrator. |
| Monitoring / drift | **Evidently** | Data drift + performance reports. |
| CI/CD | **GitHub Actions** + **pytest** + **ruff** | Test, lint, build on every push. |
| Dashboard | **Streamlit** (Community Cloud) | Free public live dashboard = the portfolio face. |
| Config / secrets | **pydantic-settings**, `.env`, GitHub Secrets | No hardcoded tokens. |

---

## 11. Repository structure

```
germany-electricity-load-forecasting-mlops/
├── README.md                      # this document
├── CLAUDE.md                      # build conventions for the coding assistant (optional)
├── pyproject.toml                 # dependencies & tooling config
├── Makefile                       # common commands (setup, test, run, train, serve)
├── docker-compose.yml             # local stack (db + api + dashboard)
├── .env.example                   # template for secrets (no real values)
├── .github/
│   └── workflows/
│       ├── ci.yml                 # tests + lint on push
│       ├── ingest.yml             # hourly ingestion (cron)
│       ├── forecast.yml           # daily day-ahead forecast (cron)
│       └── monitor_retrain.yml    # daily monitoring + drift-triggered retrain
├── config/
│   └── settings.py                # typed configuration
├── data/                          # local data (gitignored; tracked via DVC)
├── notebooks/                     # exploration only, not the source of truth
├── src/
│   ├── data/                      # ingestion & DB access
│   │   ├── entsoe_client.py
│   │   ├── weather_client.py
│   │   ├── ingest.py
│   │   └── db.py
│   ├── features/                  # feature engineering (leakage-safe)
│   │   └── build_features.py
│   ├── models/                    # training, evaluation, registry
│   │   ├── baselines.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── registry.py
│   ├── serving/                   # batch + real-time serving
│   │   ├── api.py                 # FastAPI app
│   │   └── batch_forecast.py
│   └── monitoring/                # drift, performance, retraining
│       ├── drift.py
│       ├── performance.py
│       └── retrain.py
├── dashboard/
│   └── app.py                     # Streamlit dashboard
└── tests/                         # pytest unit + integration tests
```

---

## 12. Configuration & secrets

Secrets are **never** committed. Copy `.env.example` to `.env` for local use, and add the same keys as **GitHub Secrets** for the cloud jobs.

```dotenv
# .env.example
ENTSOE_API_TOKEN=your_entsoe_security_token_here
DATABASE_URL=postgresql://user:password@host:5432/dbname
OPEN_METEO_BASE_URL=https://api.open-meteo.com/v1
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
BIDDING_ZONE=DE_LU
```

- `ENTSOE_API_TOKEN` — generated in your ENTSO-E account after the `RESTful API access` request is approved (up to 3 working days). **The system runs on historical data without it; live ingestion switches on the moment you add it.**
- `DATABASE_URL` — connection string from your free Postgres provider (Neon/Supabase).

---

## 13. How to run it locally

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/germany-electricity-load-forecasting-mlops.git
cd germany-electricity-load-forecasting-mlops

# 2. Install (Python 3.11+)
pip install -e ".[dev]"          # or: make setup

# 3. Configure
cp .env.example .env             # then fill in values

# 4. Backfill historical data (years of load + weather)
python -m src.data.ingest --backfill --years 5

# 5. Build features and train
python -m src.features.build_features
python -m src.models.train

# 6. Evaluate (incl. vs. official forecast)
python -m src.models.evaluate

# 7. Serve
uvicorn src.serving.api:app --reload      # real-time API at http://localhost:8000
streamlit run dashboard/app.py            # dashboard at http://localhost:8501

# Or run the whole stack:
docker-compose up
```

---

## 14. How it runs in the cloud (free, PC-off)

The system is designed so **your personal computer does not need to be on.**

- **Scheduling** lives in **GitHub Actions** (cron). `ingest.yml` runs hourly; `forecast.yml` runs each morning; `monitor_retrain.yml` runs daily. GitHub spins up a temporary runner, executes the job, and shuts it down — free (unlimited for public repos; ~2,000 min/month for private, far more than these short jobs use).
- **The database is cloud-hosted** (Neon/Supabase free tier), so the runners have somewhere to write when your laptop is closed.
- **The dashboard** is hosted on Streamlit Community Cloud with a public URL.

Result: **cloud scheduler + cloud database + cloud dashboard = a system that keeps living while you sleep, for €0/month.**

> Caveats to know: GitHub pauses scheduled workflows after 60 days of zero repo activity (this project commits regularly, so it won't trigger), and cron timing can be a few minutes late under load (irrelevant for hourly data).

---

## 15. Phased build plan

> This is the blueprint the coding assistant follows. Each phase is independently testable; do **not** move on until the current phase runs and its tests pass.

**Phase 0 — Scaffolding.** Create the repo structure, `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`, `CLAUDE.md`, and CI (`ci.yml` with pytest + ruff). Add a trivial passing test to prove CI works.

**Phase 1 — Data ingestion (historical first).** Implement `entsoe_client.py` (load + official forecast for `DE_LU`) and `weather_client.py` (Open-Meteo). Implement `db.py` (schema + upserts) and `ingest.py` with a `--backfill` mode. Backfill several years of history. *Runs without a token using manually exported historical CSVs / public historical data; live API path is wired for when the token arrives.*

**Phase 2 — Feature engineering.** `build_features.py`: lag (t−1, t−24, t−168), rolling stats, calendar/holiday flags, weather joins. Enforce **no leakage** (scalers fit on train only). Unit-test the feature logic.

**Phase 3 — Baselines + first real model.** `baselines.py` (seasonal naive), then LightGBM in `train.py`. Time-series CV in `evaluate.py`. Log everything to MLflow. Produce the first "beats naive, and here's how it compares to the official forecast" result.

**Phase 4 — Serving.** `api.py` (FastAPI: `/health`, `/forecast`), `batch_forecast.py` (writes day-ahead forecast to DB). Dockerize. Integration-test the endpoints.

**Phase 5 — Monitoring & retraining.** `drift.py` (Evidently), `performance.py` (error vs. actuals and vs. official forecast), `retrain.py` (champion/challenger + promote/rollback). Wire the trigger logic.

**Phase 6 — Orchestration in the cloud.** GitHub Actions workflows: `ingest.yml`, `forecast.yml`, `monitor_retrain.yml`. Move DB to the cloud free tier. Confirm jobs run with the repo secrets.

**Phase 7 — Dashboard.** `dashboard/app.py` (Streamlit): forecast vs. actuals, accuracy vs. official forecast, drift signals, model-version timeline. Deploy publicly.

**Phase 8 — Polish.** Fill in results, architecture image, and badges in this README. Add the deep-learning model (LSTM → TFT). Tidy tests and docs.

---

## 16. 9-month roadmap

- **Month 1** — Phases 0–4 solid: ingestion, features, LightGBM, serving, CI. Live on the free cloud stack, collecting data daily.
- **Month 2** — Phases 5–7: monitoring, drift-triggered retraining, dashboard public. First real retraining events start appearing as data accumulates.
- **Month 3** — Deep learning: LSTM/GRU, then Temporal Fusion Transformer; compare rigorously against LightGBM.
- **Month 4** — Multi-region: forecast the four German control areas (50Hertz, Amprion, TenneT, TransnetBW) separately.
- **Month 5** — Probabilistic forecasting (quantiles / prediction intervals) + pinball metrics.
- **Month 6** — Net-load forecasting (demand minus wind & solar) — the modern hard problem.
- **Month 7** — Orchestration upgrade (Prefect/Airflow), optional Feast feature store, optional cloud deployment using GitHub Student Pack credits.
- **Month 8** — Explainability (SHAP), model cards, data governance / EU-regulation awareness notes.
- **Month 9** — Full write-up, demo video, and CV/portfolio packaging for internship applications.

---

## 17. Results

> _To be filled in as the system runs. Suggested content:_

- **Live dashboard:** _link_
- **Headline accuracy:** MAE / MAPE of the champion model over the last 30 days.
- **Vs. official forecast:** how the model compares to the ENTSO-E day-ahead forecast.
- **Uptime / data continuity:** collecting since _date_.
- **Retraining events:** number of automated retrains and what triggered them.

| Model | MAE (MW) | MAPE (%) | Beats seasonal naive? | Beats official forecast? |
|---|---|---|---|---|
| Seasonal naive | _tbd_ | _tbd_ | — | — |
| LightGBM | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| LSTM | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TFT | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

---

## 18. Skills demonstrated

- **Software engineering:** Python, SQL, git, Docker, testing (pytest), linting, typed config, OOP, clean project structure.
- **Data & feature engineering:** API ETL, time-series database, leakage-safe feature engineering, data versioning (DVC), missing-data handling.
- **MLOps & deployment:** experiment tracking & model registry (MLflow), batch + real-time serving (FastAPI), CI/CD (GitHub Actions), orchestration, champion/challenger, rollback.
- **Monitoring:** data drift (Evidently), performance vs. fresh ground truth, automated retraining triggers, feedback loop, logging.
- **Modeling:** baselines, gradient boosting (LightGBM), deep learning (LSTM/GRU, Temporal Fusion Transformer), time-series cross-validation, hyperparameter optimization (Optuna), probabilistic forecasting.
- **Specialization:** time-series forecasting in the energy domain.

---

## 19. Glossary

- **Load** — electricity demand / consumption, measured in megawatts (MW).
- **Bidding zone** — a market area with a single electricity price (Germany + Luxembourg = `DE_LU`).
- **Control area / TSO** — the region operated by one Transmission System Operator (Germany has four).
- **Day-ahead forecast** — a forecast made today for each hour of tomorrow.
- **Drift** — when incoming data's statistical pattern moves away from the training data.
- **Champion / challenger** — the model currently in production vs. a newly trained candidate.
- **Backfill** — loading historical data in bulk to train the first model.
- **Net load** — demand minus variable renewable generation (wind + solar).

---

## 20. License, sources & acknowledgements

**License:** MIT (see `LICENSE`).

**Data sources:**
- ENTSO-E Transparency Platform — https://transparency.entsoe.eu
- Open-Meteo — https://open-meteo.com

**Acknowledgements:** Data provided by ENTSO-E and Open-Meteo. This is an independent educational project and is not affiliated with any grid operator.

---

_Built as an end-to-end MLOps learning project. Feedback and issues welcome._
