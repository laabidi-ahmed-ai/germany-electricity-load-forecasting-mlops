"""Time-series cross-validation + metrics (README §15, Phase 3).

Evaluation protocol
-------------------
* **Expanding-window CV** (``src.features.build_features.expanding_window_splits``):
  the last ``n_splits`` blocks of ``val_hours`` are validation folds, oldest
  first; each fold trains on *everything* before it. Never a random split.
* Every model in a run sees exactly the same folds and the same horizon-valid
  feature columns (``src.features.horizons``).
* Metrics per fold and averaged across folds: **MAE**, **RMSE**, **MAPE** (%).
* The report states explicitly whether LightGBM beats *both* seasonal-naive
  baselines on mean MAE.

CLI (``make evaluate``)::

    python -m src.models.evaluate [--features PATH] [--n-splits 12] [--val-hours 720]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.features.build_features import DEFAULT_OUTPUT as FEATURES_PARQUET
from src.features.build_features import TARGET, expanding_window_splits
from src.features.horizons import DAY_AHEAD, HORIZONS, Horizon, select_features
from src.models.tracking import DEFAULT_EXPERIMENT, setup_mlflow

log = logging.getLogger(__name__)

DEFAULT_N_SPLITS = 12
DEFAULT_VAL_HOURS = 24 * 30  # one-month folds -> 12 folds = the last year
DEFAULT_GAP_HOURS = 0
DEFAULT_MIN_TRAIN_HOURS = 24 * 365

METRICS = ("mae", "rmse", "mape")
BASELINE_NAMES = ("seasonal_naive_24", "seasonal_naive_168")
CHALLENGER_NAME = "lightgbm"


class Model(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def get_params(self) -> dict[str, Any]: ...


ModelFactory = Callable[[], Model]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> dict[str, float]:
    """MAE, RMSE (MW) and MAPE (%) of a point forecast."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if np.isnan(yp).any():
        raise ValueError("predictions contain NaN")
    err = yp - yt
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.abs(yt)) * 100.0),
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    horizon: str
    features: list[str]
    config: dict[str, Any]
    folds: pd.DataFrame  # fold, model, val_start, val_end, n_train, n_val, mae, rmse, mape
    summary: pd.DataFrame = field(init=False)  # index=model; <metric>_mean / <metric>_std

    def __post_init__(self) -> None:
        agg = self.folds.groupby("model")[list(METRICS)].agg(["mean", "std"])
        agg.columns = [f"{m}_{s}" for m, s in agg.columns]
        # Keep first-seen model order (baselines first, challenger last).
        order = list(dict.fromkeys(self.folds["model"]))
        self.summary = agg.loc[order]

    def mean(self, model: str, metric: str = "mae") -> float:
        return float(self.summary.loc[model, f"{metric}_mean"])

    def beats_baselines(
        self,
        challenger: str = CHALLENGER_NAME,
        baselines: tuple[str, ...] = BASELINE_NAMES,
        metric: str = "mae",
    ) -> dict[str, dict[str, Any]]:
        """Per baseline: does the challenger have a lower mean ``metric``, and by how much?"""
        out: dict[str, dict[str, Any]] = {}
        if challenger not in self.summary.index:
            return out
        c = self.mean(challenger, metric)
        for b in baselines:
            if b not in self.summary.index:
                continue
            bv = self.mean(b, metric)
            out[b] = {"beats": bool(c < bv), "improvement_pct": float((bv - c) / bv * 100.0)}
        return out

    def beats_all_baselines(self, metric: str = "mae") -> bool:
        comp = self.beats_baselines(metric=metric)
        return bool(comp) and all(v["beats"] for v in comp.values())


def cross_validate(
    df: pd.DataFrame,
    factories: dict[str, ModelFactory],
    horizon: Horizon = DAY_AHEAD,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    val_hours: int = DEFAULT_VAL_HOURS,
    gap_hours: int = DEFAULT_GAP_HOURS,
    min_train_hours: int = DEFAULT_MIN_TRAIN_HOURS,
) -> CVResult:
    """Fit/score every model on every expanding-window fold with horizon-valid features."""
    features = select_features(df.columns, horizon)
    if not features:
        raise ValueError(f"no usable features for horizon {horizon.name}")
    df = df.sort_index()
    config = {
        "n_splits": n_splits,
        "val_hours": val_hours,
        "gap_hours": gap_hours,
        "min_train_hours": min_train_hours,
    }
    log.info(
        "CV %s: %s | %d features | models=%s", horizon.name, config, len(features), list(factories)
    )

    rows: list[dict[str, Any]] = []
    folds = expanding_window_splits(
        df,
        n_splits=n_splits,
        val_hours=val_hours,
        min_train_hours=min_train_hours,
        gap_hours=gap_hours,
    )
    for k, (train, val) in enumerate(folds):
        X_tr, y_tr = train[features], train[TARGET]
        X_va, y_va = val[features], val[TARGET]
        for name, factory in factories.items():
            model = factory()
            model.fit(X_tr, y_tr)
            m = compute_metrics(y_va, model.predict(X_va))
            rows.append(
                {
                    "fold": k,
                    "model": name,
                    "val_start": val.index.min(),
                    "val_end": val.index.max(),
                    "n_train": len(train),
                    "n_val": len(val),
                    **m,
                }
            )
            log.info(
                "fold %2d %-18s %s -> %s  MAE=%8.1f RMSE=%8.1f MAPE=%5.2f%%",
                k,
                name,
                val.index.min().date(),
                val.index.max().date(),
                m["mae"],
                m["rmse"],
                m["mape"],
            )
    return CVResult(horizon.name, features, config, pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(cv: CVResult) -> str:
    """Markdown: per-fold table, mean ± std per model, and the baseline verdict."""
    lines = [
        f"## Expanding-window CV - horizon: {cv.horizon}",
        f"config: {cv.config} | {len(cv.features)} features",
        "",
        "### Per fold",
        "",
        "| fold | validation window | model | MAE (MW) | RMSE (MW) | MAPE (%) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for r in cv.folds.itertuples(index=False):
        window = f"{r.val_start.date()} -> {r.val_end.date()}"
        lines.append(
            f"| {r.fold} | {window} | {r.model} | {r.mae:,.0f} | {r.rmse:,.0f} | {r.mape:.2f} |"
        )
    lines += [
        "",
        "### Mean over folds (± std)",
        "",
        "| model | MAE (MW) | RMSE (MW) | MAPE (%) |",
        "|---|---:|---:|---:|",
    ]
    for model, r in cv.summary.iterrows():
        lines.append(
            f"| {model} | {r.mae_mean:,.0f} ± {r.mae_std:,.0f} | "
            f"{r.rmse_mean:,.0f} ± {r.rmse_std:,.0f} | {r.mape_mean:.2f} ± {r.mape_std:.2f} |"
        )
    comp = cv.beats_baselines()
    if comp:
        lines += ["", "### Verdict (mean MAE)", ""]
        for b, v in comp.items():
            word = "beats" if v["beats"] else "does NOT beat"
            lines.append(
                f"- {CHALLENGER_NAME} **{word}** {b}: "
                f"{cv.mean(CHALLENGER_NAME):,.0f} vs {cv.mean(b):,.0f} MW "
                f"({v['improvement_pct']:+.1f}% MAE reduction)"
            )
        verdict = "YES" if cv.beats_all_baselines() else "NO"
        lines.append(f"- **{CHALLENGER_NAME} beats both seasonal-naive baselines: {verdict}**")
    return "\n".join(lines)


def log_cv_metrics(cv: CVResult) -> None:
    """Log per-fold (step = fold) and mean/std metrics of every model to the active MLflow run."""
    import mlflow

    for r in cv.folds.itertuples(index=False):
        mlflow.log_metrics({f"{r.model}_{m}": getattr(r, m) for m in METRICS}, step=int(r.fold))
    for model, r in cv.summary.iterrows():
        mlflow.log_metrics({f"{model}_{col}": float(r[col]) for col in cv.summary.columns})
    for b, v in cv.beats_baselines().items():
        mlflow.log_metric(f"{CHALLENGER_NAME}_vs_{b}_improvement_pct", v["improvement_pct"])
    if CHALLENGER_NAME in cv.summary.index:
        mlflow.log_metric("beats_all_baselines", int(cv.beats_all_baselines()))


def log_cv_run(
    cv: CVResult, *, experiment: str = DEFAULT_EXPERIMENT, run_name: str | None = None
) -> str:
    """Log a standalone CV run (no model artifact) to MLflow and return its run id."""
    import mlflow

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name or f"cv_{cv.horizon}") as run:
        mlflow.set_tags({"stage": "evaluation", "horizon": cv.horizon, "phase": "3"})
        mlflow.log_params({"horizon": cv.horizon, "n_features": len(cv.features), **cv.config})
        mlflow.log_dict({"features": cv.features}, "features.json")
        log_cv_metrics(cv)
        mlflow.log_text(format_report(cv), "cv_report.md")
        mlflow.log_text(cv.folds.to_csv(index=False), "cv_folds.csv")
        return run.info.run_id


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    from src.models.baselines import BASELINE_FACTORIES
    from src.models.train import lightgbm_factory, load_feature_frame

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.models.evaluate")
    p.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    p.add_argument("--horizon", choices=sorted(HORIZONS), default=DAY_AHEAD.name)
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--val-hours", type=int, default=DEFAULT_VAL_HOURS)
    p.add_argument("--gap-hours", type=int, default=DEFAULT_GAP_HOURS)
    p.add_argument("--min-train-hours", type=int, default=DEFAULT_MIN_TRAIN_HOURS)
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)

    df = load_feature_frame(args.features)
    cv = cross_validate(
        df,
        {**BASELINE_FACTORIES, CHALLENGER_NAME: lightgbm_factory()},
        HORIZONS[args.horizon],
        n_splits=args.n_splits,
        val_hours=args.val_hours,
        gap_hours=args.gap_hours,
        min_train_hours=args.min_train_hours,
    )
    print(format_report(cv))
    if not args.no_mlflow:
        print(f"\nMLflow run: {log_cv_run(cv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
