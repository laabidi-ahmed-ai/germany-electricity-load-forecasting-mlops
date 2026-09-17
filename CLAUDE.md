# CLAUDE.md — Build Conventions & House Rules

> This file is read automatically by Claude Code at the start of every session.
> It sets the rules for how this project is built. **`README.md` is the
> authoritative specification** — always follow it. This file governs *how* you work.

---

## Project in one line

An end-to-end MLOps system that forecasts **Germany's national electricity load**
(day-ahead, hourly), continuously ingests fresh grid + weather data, monitors its
own accuracy against the official grid-operator forecast, and retrains itself on drift.

Full details, architecture, and the phased build plan are in **`README.md` (section 15)**.

---

## Golden rules (do not break these)

1. **Follow the README.** `README.md` is the spec. Build to it. If something is
   ambiguous, ask before improvising.
2. **Work strictly phase by phase** (README §15). Do **not** start a phase until the
   previous one runs and its tests pass. After each phase, **stop and summarize**
   what you built and how to run/test it, then wait for my go-ahead.
3. **Never hardcode secrets.** All tokens/URLs come from `.env` (local) and GitHub
   Secrets (cloud), loaded via `config/settings.py`. Never commit `.env` or a real token.
4. **Free-tier only.** Every tool/service must have a free tier (see README §10).
   No choice that requires a paid subscription.
5. **Target = Germany national load.** Bidding zone `DE_LU` (ENTSO-E) / region `DE`
   (SMARD). Data starts **2021-03-01** (post-COVID cutoff — the pandemic period is
   deliberately excluded because its demand patterns are unrepresentative).
6. **No data leakage, ever.** In time series this is the #1 killer. Scalers and any
   fitted transforms are fit on **training data only**. Use **time-based splits**
   (expanding/rolling window) — never random splits.
7. **Test what you build.** Every phase adds tests. `pytest` must stay green.
8. **Small, reviewable commits.** One logical change per commit; clear messages.

---

## Data sources (both official, free, keyless to start)

- **SMARD** (Bundesnetzagentur) — actual German load, no token. Used to start
  building/training **immediately**. Starter fetch script: `fetch_starter_data.py`
  (run it, verify German load averages ~52,000-57,000 MW - the real SMARD series,
  lower since the 2022 energy crisis - then refactor into
  `src/data/`).
- **ENTSO-E** — official day-ahead load forecast (the benchmark) + European coverage.
  Requires a security token (`ENTSOE_API_TOKEN`); arrives by email within a few days.
  Build the ENTSO-E path now, but keep everything runnable **without** the token.
- **Open-Meteo** — weather features, no key.

---

## Tech stack (see README §10 for the full table)

Python 3.11+ · Postgres/TimescaleDB (Neon/Supabase free tier) · pandas · LightGBM ·
PyTorch (LSTM/TFT) · MLflow · Optuna · FastAPI · Docker · Evidently · Streamlit ·
GitHub Actions (scheduling + CI) · DVC · pytest · ruff.

---

## Standard commands (define these in the Makefile)

```bash
make setup      # install deps into a venv
make test       # pytest
make lint       # ruff
make data       # run the ingestion / backfill
make train      # train + evaluate, log to MLflow
make serve      # run the FastAPI app locally
make dashboard  # run the Streamlit dashboard
```

---

## Repository layout

Follow the structure in **README.md §11** exactly:
`src/{data,features,models,serving,monitoring}`, `config/`, `dashboard/`, `tests/`,
`.github/workflows/`. Notebooks are for exploration only — never the source of truth.

---

## Definition of done for a phase

- Code runs end-to-end for that phase.
- New tests exist and `pytest` is green.
- `ruff` is clean.
- A short note explaining what was built and how to verify it.
- Committed with a clear message. Then **stop and wait for review.**
