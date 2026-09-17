"""Retraining trigger + champion / challenger promotion (README §9, Phase 5).

Triggers (any one fires -> retrain)
-----------------------------------
1. **error**   : rolling 7-day MAPE of the champion's live forecasts > ``mape_threshold_pct``
2. **official**: over 30 days the official forecast beats the model by more than
                 ``official_gap_pct`` (we exist to beat the benchmark)
3. **drift**   : Evidently dataset or prediction drift (``src.monitoring.drift``)

Champion / challenger (fair by construction)
--------------------------------------------
* **holdout**    = data that arrived *after* the champion's ``train_end`` (never seen
                   by the champion), capped at the last ``holdout_days``. If fewer
                   than ``min_holdout_hours`` exist the comparison is skipped - a
                   champion cannot be dethroned on data it was trained on.
* **candidate**  = LightGBM trained on everything strictly before the holdout, with
                   the same day-ahead features; scored on the holdout vs the champion.
* **promotion**  only if the candidate's holdout MAE is at least ``min_improvement_pct``
                   lower than the champion's (the anti-churn margin). The winner is
                   then refitted on *all* data, logged with the standard CV report,
                   registered, and the ``champion`` alias moved. Otherwise: keep.
* **rollback**   = ``python -m src.models.registry --promote <old version>``.

Every decision (trigger evaluation and outcome) is logged to MLflow (tags
``stage=retrain``, ``decision``, ``reason``) and to the ``monitoring_events`` table.

CLI::

    python -m src.monitoring.retrain --check-only      # make monitor
    python -m src.monitoring.retrain                   # make retrain (acts only if triggered)
    python -m src.monitoring.retrain --force [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from config.settings import get_settings
from src.data import db
from src.features.build_features import TARGET, build_features
from src.features.horizons import DAY_AHEAD, select_features
from src.models import evaluate, registry, train
from src.models.baselines import BASELINE_FACTORIES
from src.models.evaluate import compute_metrics
from src.models.registry import LoadedModel
from src.models.tracking import DEFAULT_EXPERIMENT, setup_mlflow
from src.monitoring import drift as drift_mod
from src.monitoring import performance as perf_mod

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainConfig:
    # error trigger
    error_window_days: int = 7
    mape_threshold_pct: float = 3.5
    # official-forecast trigger
    official_window_days: int = 30
    official_gap_pct: float = 15.0
    # drift trigger
    drift_window_days: int = drift_mod.DEFAULT_WINDOW_DAYS
    drift_share_threshold: float = drift_mod.DEFAULT_DRIFT_SHARE
    drift_column_threshold: float = drift_mod.DEFAULT_COLUMN_THRESHOLD
    # champion / challenger
    holdout_days: int = 14
    min_holdout_hours: int = 24 * 7
    min_improvement_pct: float = 3.0
    lgbm_params: dict[str, Any] | None = None
    cv_n_splits: int = evaluate.DEFAULT_N_SPLITS
    cv_val_hours: int = evaluate.DEFAULT_VAL_HOURS
    cv_min_train_hours: int = evaluate.DEFAULT_MIN_TRAIN_HOURS


DEFAULT_CONFIG = RetrainConfig()


@dataclass
class Check:
    name: str
    fired: bool
    value: float | None
    threshold: float
    detail: str


@dataclass
class TriggerDecision:
    as_of: pd.Timestamp
    champion_version: str
    checks: list[Check]
    performance: perf_mod.PerformanceReport | None
    drift: drift_mod.DriftReport | None
    drift_error: str | None = None

    @property
    def triggered(self) -> bool:
        return any(c.fired for c in self.checks)

    @property
    def reasons(self) -> list[str]:
        return [c.detail for c in self.checks if c.fired]

    def summary(self) -> str:
        lines = [
            f"Retrain triggers as of {self.as_of:%Y-%m-%d %H:%M}Z (champion v{self.champion_version}):"
        ]
        for c in self.checks:
            state = "FIRED" if c.fired else "ok   "
            val = "n/a" if c.value is None else f"{c.value:.2f}"
            lines.append(
                f"  [{state}] {c.name:<9} value={val:<8} threshold={c.threshold:<6} {c.detail}"
            )
        lines.append(f"  => RETRAIN {'TRIGGERED' if self.triggered else 'not triggered'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "champion_version": self.champion_version,
            "triggered": self.triggered,
            "reasons": self.reasons,
            "checks": [asdict(c) for c in self.checks],
            "performance": self.performance.to_dict() if self.performance else None,
            "drift": self.drift.to_dict() if self.drift else None,
            "drift_error": self.drift_error,
        }


# --------------------------------------------------------------------------- #
# Trigger evaluation
# --------------------------------------------------------------------------- #
def evaluate_triggers(
    engine: Engine,
    champion: LoadedModel,
    cfg: RetrainConfig = DEFAULT_CONFIG,
    *,
    as_of: pd.Timestamp | None = None,
    frame: pd.DataFrame | None = None,
    drift_runner: drift_mod.Runner = drift_mod.run_evidently,
) -> TriggerDecision:
    as_of = _utc(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    checks: list[Check] = []

    # 1 + 2: live performance of the champion's stored forecasts
    perf = perf_mod.compute_report(
        engine,
        as_of=as_of,
        windows=(cfg.error_window_days, cfg.official_window_days),
        model_version=champion.version,
    )
    w_err = perf.window(cfg.error_window_days)
    if w_err is not None and w_err.judged:
        checks.append(
            Check(
                "error",
                fired=w_err.model_mape > cfg.mape_threshold_pct,
                value=w_err.model_mape,
                threshold=cfg.mape_threshold_pct,
                detail=f"{cfg.error_window_days}d MAPE {w_err.model_mape:.2f}% over {w_err.n_hours}h",
            )
        )
    else:
        n = 0 if w_err is None else w_err.n_hours
        checks.append(
            Check(
                "error",
                False,
                None,
                cfg.mape_threshold_pct,
                f"insufficient data ({n} scored hours)",
            )
        )

    w_off = perf.window(cfg.official_window_days)
    if w_off is not None and w_off.judged:
        gap = -w_off.improvement_pct  # +x% = official is x% better than the model
        checks.append(
            Check(
                "official",
                fired=gap > cfg.official_gap_pct,
                value=gap,
                threshold=cfg.official_gap_pct,
                detail=(
                    f"{cfg.official_window_days}d: model MAE {w_off.model_mae:,.0f} vs official "
                    f"{w_off.official_mae:,.0f} MW (official better by {gap:+.1f}%)"
                ),
            )
        )
    else:
        n = 0 if w_off is None else w_off.n_hours
        checks.append(
            Check(
                "official",
                False,
                None,
                cfg.official_gap_pct,
                f"insufficient data ({n} scored hours)",
            )
        )

    # 3: drift
    drift_report: drift_mod.DriftReport | None = None
    drift_error: str | None = None
    try:
        drift_report = drift_mod.compute_drift(
            engine,
            champion,
            as_of=as_of,
            window_days=cfg.drift_window_days,
            drift_share_threshold=cfg.drift_share_threshold,
            column_threshold=cfg.drift_column_threshold,
            frame=frame,
            runner=drift_runner,
        )
        checks.append(
            Check(
                "drift",
                fired=drift_report.flag,
                value=drift_report.share_drifted,
                threshold=cfg.drift_share_threshold,
                detail=(
                    f"{drift_report.n_drifted}/{drift_report.n_monitored} features drifted "
                    f"{drift_report.drifted_columns}; prediction drift "
                    f"{'YES' if drift_report.prediction_drift else 'no'}"
                ),
            )
        )
    except Exception as err:  # drift must never take the monitor down
        drift_error = str(err)
        log.exception("drift evaluation failed")
        checks.append(Check("drift", False, None, cfg.drift_share_threshold, f"error: {err}"))

    decision = TriggerDecision(as_of, champion.version, checks, perf, drift_report, drift_error)
    log.info("%s", decision.summary())
    return decision


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
@dataclass
class ChallengerOutcome:
    decision: str  # promoted | kept | skipped
    reason: str
    champion_version: str
    challenger_version: str | None = None
    holdout_start: pd.Timestamp | None = None
    holdout_end: pd.Timestamp | None = None
    holdout_hours: int = 0
    champion_mae: float | None = None
    challenger_mae: float | None = None
    champion_mape: float | None = None
    challenger_mape: float | None = None
    improvement_pct: float | None = None
    training_run_id: str | None = None
    retrain_run_id: str | None = None
    triggers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = f"Champion/challenger: {self.decision.upper()} - {self.reason}"
        if self.champion_mae is None:
            return head
        return (
            f"{head}\n  holdout {self.holdout_start:%Y-%m-%d} -> {self.holdout_end:%Y-%m-%d} "
            f"({self.holdout_hours}h): champion v{self.champion_version} MAE {self.champion_mae:,.0f} "
            f"({self.champion_mape:.2f}%) vs challenger MAE {self.challenger_mae:,.0f} "
            f"({self.challenger_mape:.2f}%) -> improvement {self.improvement_pct:+.1f}%"
            + (
                f"\n  promoted challenger as v{self.challenger_version}"
                if self.challenger_version
                else ""
            )
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("holdout_start", "holdout_end"):
            if d[k] is not None:
                d[k] = d[k].isoformat()
        return d


def run_champion_challenger(
    engine: Engine,
    champion: LoadedModel,
    cfg: RetrainConfig = DEFAULT_CONFIG,
    *,
    as_of: pd.Timestamp | None = None,
    frame: pd.DataFrame | None = None,
    triggers: list[str] | None = None,
    dry_run: bool = False,
) -> ChallengerOutcome:
    frame = frame if frame is not None else build_features(engine, output=None)
    as_of = _utc(as_of) if as_of is not None else frame.index.max()
    frame = frame[frame.index <= as_of]
    cols = select_features(frame.columns, DAY_AHEAD)
    champion.check_features(list(frame.columns))

    _, train_end = registry.training_window(champion)
    if train_end is None:
        return ChallengerOutcome("skipped", "champion has no recorded train_end", champion.version)

    holdout = frame[frame.index > train_end]
    holdout = (
        holdout[holdout.index > holdout.index.max() - pd.Timedelta(days=cfg.holdout_days)]
        if not holdout.empty
        else holdout
    )
    if len(holdout) < cfg.min_holdout_hours:
        return ChallengerOutcome(
            "skipped",
            f"only {len(holdout)}h of data since the champion's train_end ({train_end:%Y-%m-%d %H:%M}Z); "
            f"need {cfg.min_holdout_hours}h for a fair out-of-time comparison",
            champion.version,
            holdout_hours=len(holdout),
            triggers=triggers or [],
        )

    fit_frame = frame[frame.index < holdout.index.min()]
    candidate = train.LightGBMForecaster(cfg.lgbm_params).fit(fit_frame[cols], fit_frame[TARGET])
    champ_m = compute_metrics(holdout[TARGET], champion.predict(holdout))
    cand_m = compute_metrics(holdout[TARGET], candidate.predict(holdout[cols]))
    improvement = (champ_m["mae"] - cand_m["mae"]) / champ_m["mae"] * 100.0

    outcome = ChallengerOutcome(
        decision="kept",
        reason="",
        champion_version=champion.version,
        holdout_start=holdout.index.min(),
        holdout_end=holdout.index.max(),
        holdout_hours=len(holdout),
        champion_mae=champ_m["mae"],
        challenger_mae=cand_m["mae"],
        champion_mape=champ_m["mape"],
        challenger_mape=cand_m["mape"],
        improvement_pct=improvement,
        triggers=triggers or [],
    )
    if improvement < cfg.min_improvement_pct:
        outcome.reason = f"challenger improves MAE by {improvement:+.1f}% < required {cfg.min_improvement_pct}% margin"
        return outcome

    outcome.reason = (
        f"challenger beats champion by {improvement:+.1f}% (>= {cfg.min_improvement_pct}%)"
    )
    if dry_run:
        outcome.decision = "promoted (dry-run)"
        return outcome

    # Refit the winner on everything, log it exactly like `make train`, register, promote.
    cv = evaluate.cross_validate(
        frame,
        {**BASELINE_FACTORIES, train.MODEL_NAME: train.lightgbm_factory(cfg.lgbm_params)},
        DAY_AHEAD,
        n_splits=cfg.cv_n_splits,
        val_hours=cfg.cv_val_hours,
        min_train_hours=cfg.cv_min_train_hours,
    )
    final, final_cols = train.train_final_model(frame, DAY_AHEAD, cfg.lgbm_params)
    run_id = train.log_training_run(
        final, final_cols, cv, DAY_AHEAD, frame, run_name="retrain_challenger"
    )
    mv = registry.register_run_model(run_id)
    registry.promote(mv.version, engine=engine)
    outcome.decision = "promoted"
    outcome.challenger_version = str(mv.version)
    outcome.training_run_id = run_id
    log.info("promoted challenger v%s (rollback: --promote %s)", mv.version, champion.version)
    return outcome


# --------------------------------------------------------------------------- #
# Logging of decisions
# --------------------------------------------------------------------------- #
def log_decision(
    decision: TriggerDecision,
    outcome: ChallengerOutcome | None,
    cfg: RetrainConfig,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    """One MLflow run per monitoring pass: triggers, metrics, decision and reason."""
    import mlflow

    setup_mlflow(experiment)
    kind = "retrain" if outcome is not None else "check"
    with mlflow.start_run(run_name=f"{kind}_{decision.as_of:%Y%m%dT%H}") as run:
        mlflow.set_tags(
            {
                "stage": "retrain",
                "kind": kind,
                "phase": "5",
                "champion_version": decision.champion_version,
                "triggered": str(decision.triggered),
                "decision": outcome.decision if outcome else "check-only",
                "reason": (
                    outcome.reason if outcome else "; ".join(decision.reasons) or "no trigger"
                )[:250],
            }
        )
        mlflow.log_params({f"cfg_{k}": v for k, v in asdict(cfg).items() if k != "lgbm_params"})
        mlflow.log_params({"as_of": decision.as_of.isoformat()})
        for c in decision.checks:
            mlflow.log_metric(f"check_{c.name}_fired", int(c.fired))
            if c.value is not None:
                mlflow.log_metric(f"check_{c.name}_value", c.value)
        if decision.drift is not None:
            mlflow.log_metrics(
                {
                    "drift_share": decision.drift.share_drifted,
                    "drift_n_drifted": decision.drift.n_drifted,
                    "drift_flag": int(decision.drift.flag),
                }
            )
        if outcome is not None and outcome.champion_mae is not None:
            mlflow.log_metrics(
                {
                    "holdout_hours": outcome.holdout_hours,
                    "champion_holdout_mae": outcome.champion_mae,
                    "challenger_holdout_mae": outcome.challenger_mae,
                    "champion_holdout_mape": outcome.champion_mape,
                    "challenger_holdout_mape": outcome.challenger_mape,
                    "improvement_pct": outcome.improvement_pct,
                    "promoted": int(outcome.decision.startswith("promoted")),
                }
            )
        mlflow.log_dict(decision.to_dict(), "trigger_decision.json")
        if outcome is not None:
            mlflow.log_dict(outcome.to_dict(), "challenger_outcome.json")
        return run.info.run_id


def record_event(
    engine: Engine, decision: TriggerDecision, outcome: ChallengerOutcome | None
) -> int:
    w7 = decision.performance.window(7) if decision.performance else None
    return db.insert_monitoring_event(
        engine,
        as_of=decision.as_of.to_pydatetime(),
        kind="retrain" if outcome is not None else "check",
        triggered=decision.triggered,
        decision=outcome.decision if outcome else None,
        reason=(outcome.reason if outcome else "; ".join(decision.reasons) or None),
        model_version=decision.champion_version,
        model_mape_7d=None if w7 is None else w7.model_mape,
        official_mape_7d=None if w7 is None else w7.official_mape,
        drift_share=None if decision.drift is None else decision.drift.share_drifted,
        details=json.dumps(
            {"decision": decision.to_dict(), "outcome": outcome.to_dict() if outcome else None},
            default=str,
        ),
    )


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.monitoring.retrain")
    p.add_argument("--check-only", action="store_true", help="evaluate triggers, never retrain")
    p.add_argument(
        "--force", action="store_true", help="run champion/challenger even if not triggered"
    )
    p.add_argument("--dry-run", action="store_true", help="never register / promote")
    p.add_argument("--as-of", default=None)
    p.add_argument("--mape-threshold", type=float, default=RetrainConfig.mape_threshold_pct)
    p.add_argument("--official-gap", type=float, default=RetrainConfig.official_gap_pct)
    p.add_argument("--drift-share", type=float, default=RetrainConfig.drift_share_threshold)
    p.add_argument("--drift-threshold", type=float, default=RetrainConfig.drift_column_threshold)
    p.add_argument("--holdout-days", type=int, default=RetrainConfig.holdout_days)
    p.add_argument("--min-improvement", type=float, default=RetrainConfig.min_improvement_pct)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    cfg = RetrainConfig(
        mape_threshold_pct=args.mape_threshold,
        official_gap_pct=args.official_gap,
        drift_share_threshold=args.drift_share,
        drift_column_threshold=args.drift_threshold,
        holdout_days=args.holdout_days,
        min_improvement_pct=args.min_improvement,
    )
    engine = db.get_engine(args.database_url)
    db.init_db(engine)
    champion = registry.load_champion(engine)
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    frame = build_features(engine, output=None)

    decision = evaluate_triggers(engine, champion, cfg, as_of=as_of, frame=frame)
    print(decision.performance.summary() if decision.performance else "no performance data")
    print()
    print(
        decision.drift.summary() if decision.drift else f"drift unavailable: {decision.drift_error}"
    )
    print()
    print(decision.summary())

    outcome: ChallengerOutcome | None = None
    if not args.check_only and (decision.triggered or args.force):
        outcome = run_champion_challenger(
            engine,
            champion,
            cfg,
            as_of=as_of,
            frame=frame,
            triggers=decision.reasons or (["forced"] if args.force else []),
            dry_run=args.dry_run,
        )
        print()
        print(outcome.summary())
    elif not args.check_only:
        print("\nNo trigger fired - champion kept, nothing retrained.")

    record_event(engine, decision, outcome)
    if not args.no_mlflow:
        print(f"\nMLflow run: {log_decision(decision, outcome, cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
