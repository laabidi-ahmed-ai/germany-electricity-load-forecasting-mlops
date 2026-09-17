"""LightGBM day-ahead load model: training + MLflow logging (README §15, Phase 3).

``LightGBMForecaster`` wraps ``lightgbm.LGBMRegressor`` behind the same
``fit / predict / get_params / name`` protocol as the baselines, so the CV in
``evaluate.py`` treats every model identically.

Early stopping is done on the *chronological tail* of the training data (last
``early_stopping_frac``), never on the evaluation fold - otherwise the number of
trees would be tuned on the data we report metrics on.

No scaler: tree models are invariant to monotone feature transforms. If a linear
model is ever added, fit its scaler on the training split only (CLAUDE.md #6).

CLI (``make train``)::

    python -m src.models.train [--features PATH] [--n-splits 12] [--val-hours 720]

runs the expanding-window CV (baselines + LightGBM), fits the final model on all
data, and logs params / per-fold metrics / feature importances / the model to
MLflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.features.build_features import DEFAULT_OUTPUT as FEATURES_PARQUET
from src.features.build_features import TARGET
from src.features.horizons import DAY_AHEAD, HORIZONS, Horizon, excluded_features, select_features
from src.models import evaluate
from src.models.baselines import BASELINE_FACTORIES
from src.models.tracking import DEFAULT_EXPERIMENT, setup_mlflow

log = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",  # L2; MAE is reported but L2 trains more stably
    "n_estimators": 3000,  # upper bound - early stopping picks the real number
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 100
EARLY_STOPPING_FRAC = 0.10


class LightGBMForecaster:
    """LGBMRegressor with chronological-tail early stopping; fit/predict protocol."""

    name = MODEL_NAME

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        early_stopping_frac: float = EARLY_STOPPING_FRAC,
        early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    ) -> None:
        self.params = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        self.early_stopping_frac = early_stopping_frac
        self.early_stopping_rounds = early_stopping_rounds
        self.model_ = None
        self.feature_names_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMForecaster:
        import lightgbm as lgb

        if not X.index.is_monotonic_increasing:
            raise ValueError("X must be sorted in time for chronological early stopping")
        self.feature_names_ = list(X.columns)
        model = lgb.LGBMRegressor(**self.params)

        n_stop = int(len(X) * self.early_stopping_frac)
        if self.early_stopping_rounds > 0 and n_stop >= 24:
            X_fit, y_fit = X.iloc[:-n_stop], y.iloc[:-n_stop]
            X_es, y_es = X.iloc[-n_stop:], y.iloc[-n_stop:]
            model.fit(
                X_fit,
                y_fit,
                eval_X=X_es,
                eval_y=y_es,
                eval_metric="l1",
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            self.best_iteration_ = int(model.best_iteration_ or model.n_estimators)
        else:
            model.fit(X, y)
            self.best_iteration_ = int(model.n_estimators)
        self.model_ = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        return self.model_.predict(X[self.feature_names_])

    def get_params(self) -> dict[str, Any]:
        return {
            **self.params,
            "early_stopping_frac": self.early_stopping_frac,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

    def feature_importances(self) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("model is not fitted")
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        split = self.model_.booster_.feature_importance(importance_type="split")
        return (
            pd.DataFrame({"feature": self.feature_names_, "gain": gain, "split": split})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


def lightgbm_factory(params: dict[str, Any] | None = None):
    def _make() -> LightGBMForecaster:
        return LightGBMForecaster(params)

    return _make


def train_final_model(
    df: pd.DataFrame, horizon: Horizon = DAY_AHEAD, params: dict[str, Any] | None = None
) -> tuple[LightGBMForecaster, list[str]]:
    """Fit LightGBM on the whole frame using only horizon-valid features."""
    cols = select_features(df.columns, horizon)
    model = LightGBMForecaster(params).fit(df[cols], df[TARGET])
    log.info(
        "final %s model: %d rows, %d features, best_iteration=%d",
        horizon.name,
        len(df),
        len(cols),
        model.best_iteration_,
    )
    return model, cols


def load_feature_frame(path: Path = FEATURES_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make features` first")
    df = pd.read_parquet(path)
    if str(df.index.tz) != "UTC":
        raise ValueError("feature frame index must be UTC")
    return df.sort_index()


# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
def log_training_run(
    model: LightGBMForecaster,
    feature_cols: list[str],
    cv: evaluate.CVResult,
    horizon: Horizon,
    df: pd.DataFrame,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
) -> str:
    """One MLflow run: params, per-fold + mean CV metrics, baseline comparison, model."""
    import mlflow
    import mlflow.lightgbm

    setup_mlflow(experiment)
    with mlflow.start_run(run_name=run_name or f"{MODEL_NAME}_{horizon.name}") as run:
        mlflow.set_tags(
            {"model": MODEL_NAME, "horizon": horizon.name, "stage": "training", "phase": "3"}
        )
        mlflow.log_params({f"lgbm_{k}": v for k, v in model.get_params().items()})
        mlflow.log_params(
            {
                "horizon": horizon.name,
                "min_lag_hours": horizon.min_lag_hours,
                "n_features": len(feature_cols),
                "n_train_rows": len(df),
                "train_start": df.index.min().isoformat(),
                "train_end": df.index.max().isoformat(),
                "best_iteration": model.best_iteration_,
                **{f"cv_{k}": v for k, v in cv.config.items()},
            }
        )
        mlflow.log_dict(
            {
                "features": feature_cols,
                "excluded_for_horizon": excluded_features(df.columns, horizon),
            },
            "features.json",
        )
        evaluate.log_cv_metrics(cv)
        mlflow.log_text(evaluate.format_report(cv), "cv_report.md")
        mlflow.log_text(cv.folds.to_csv(index=False), "cv_folds.csv")
        mlflow.log_text(model.feature_importances().to_csv(index=False), "feature_importance.csv")
        mlflow.lightgbm.log_model(
            model.model_,
            name="model",
            input_example=df[feature_cols].head(5),
        )
        log.info("MLflow run %s logged (experiment %r)", run.info.run_id, experiment)
        return run.info.run_id


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.models.train")
    p.add_argument("--features", type=Path, default=FEATURES_PARQUET)
    p.add_argument("--horizon", choices=sorted(HORIZONS), default=DAY_AHEAD.name)
    p.add_argument("--n-splits", type=int, default=evaluate.DEFAULT_N_SPLITS)
    p.add_argument("--val-hours", type=int, default=evaluate.DEFAULT_VAL_HOURS)
    p.add_argument("--gap-hours", type=int, default=evaluate.DEFAULT_GAP_HOURS)
    p.add_argument("--min-train-hours", type=int, default=evaluate.DEFAULT_MIN_TRAIN_HOURS)
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    horizon = HORIZONS[args.horizon]
    df = load_feature_frame(args.features)

    factories = {**BASELINE_FACTORIES, MODEL_NAME: lightgbm_factory()}
    cv = evaluate.cross_validate(
        df,
        factories,
        horizon,
        n_splits=args.n_splits,
        val_hours=args.val_hours,
        gap_hours=args.gap_hours,
        min_train_hours=args.min_train_hours,
    )
    print(evaluate.format_report(cv))

    model, cols = train_final_model(df, horizon)
    print("\nTop features by gain:")
    print(model.feature_importances().head(10).to_string(index=False))

    if not args.no_mlflow:
        run_id = log_training_run(model, cols, cv, horizon, df)
        print(f"\nMLflow run: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
