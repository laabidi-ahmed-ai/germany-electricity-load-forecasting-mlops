"""Data / prediction / target drift with Evidently (>= 0.7 API).

What is compared
----------------
* **current**   : the last ``window_days`` of the feature frame (built from the DB),
                  plus the champion's predictions on it and the actual target.
* **reference** : the *same season* of the training data - rows whose day-of-year
                  falls inside the current window (± ``pad_days``) in every training
                  year. Comparing a September window against a five-year mixture
                  would flag "drift" every day purely because of seasonality.
* **columns**   : load lags + weather (the model's non-calendar inputs), the
                  ``prediction`` column and the target. Calendar features are
                  deterministic functions of the date and are excluded - their
                  "drift" between any two windows is by construction.

Evidently 0.7
-------------
``Report([DataDriftPreset(columns=..., num_method="wasserstein", num_threshold=...)])
.run(current, reference)`` returns a snapshot; ``snapshot.dict()["metrics"]`` is a
list of ``{"config": {"type", "column", "threshold", ...}, "value": ...}``.
Per-column metrics carry a drift *score* (normed Wasserstein distance = shift in
units of the reference standard deviation) and the threshold used; a column
drifts when ``score > threshold``. The ``runner`` argument lets tests inject a
fake instead of Evidently.

Thresholds (calibrated on this data)
------------------------------------
Evidently's default numeric threshold (0.1) flags 5-8 of the 8 monitored
features in *every* 30-day window of every year - it detects ordinary
year-to-year weather / demand variation. Replaying the seasonal comparison over
2022-2026 shows normal windows scoring 0.05-0.25 per feature, while the genuine
regime shifts (Sep 2022 energy crisis, its 2023 aftermath) score 0.25-0.5 on the
load lags. Defaults are therefore ``column_threshold = 0.25`` and
``drift_share = 0.5`` (half of the monitored features must move). Both are
CLI / config knobs.

CLI::

    python -m src.monitoring.drift [--window-days 30] [--as-of ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from config.settings import get_settings
from src.data import db
from src.features.build_features import DEFAULT_OUTPUT as FEATURES_PARQUET
from src.features.build_features import TARGET, build_features
from src.features.horizons import DAY_AHEAD, most_recent_load_lag, select_features
from src.models.registry import LoadedModel

log = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 30
DEFAULT_PAD_DAYS = 7
DEFAULT_DRIFT_SHARE = 0.5  # >= this share of monitored features drifted -> dataset drift
DEFAULT_COLUMN_THRESHOLD = 0.25  # normed Wasserstein distance per column (see module doc)
PREDICTION_COL = "prediction"
MIN_CURRENT_ROWS = 24 * 3

Runner = Callable[[pd.DataFrame, pd.DataFrame, list[str], float], list[dict[str, Any]]]


# --------------------------------------------------------------------------- #
# Column selection + seasonal reference
# --------------------------------------------------------------------------- #
def drift_columns(feature_columns: list[str]) -> list[str]:
    """Load-derived + weather features of the day-ahead set (calendar excluded)."""
    return [
        c
        for c in select_features(feature_columns, DAY_AHEAD)
        if most_recent_load_lag(c) is not None or c.endswith("_de_avg")
    ]


def seasonal_reference(
    training: pd.DataFrame, current_index: pd.DatetimeIndex, *, pad_days: int = DEFAULT_PAD_DAYS
) -> pd.DataFrame:
    """Training rows whose day-of-year lies within the current window (± pad), any year."""
    lo = int(current_index.min().dayofyear) - pad_days
    hi = int(current_index.max().dayofyear) + pad_days
    doy = training.index.dayofyear
    if lo < 1 or hi > 366:  # window wraps the year boundary
        mask = (doy >= (lo % 366 or 366)) | (doy <= (hi % 366 or 366))
    else:
        mask = (doy >= lo) & (doy <= hi)
    mask &= training.index < current_index.min()  # reference is strictly in the past
    return training[mask]


# --------------------------------------------------------------------------- #
# Evidently runner
# --------------------------------------------------------------------------- #
def run_evidently(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    threshold: float = DEFAULT_COLUMN_THRESHOLD,
) -> list[dict[str, Any]]:
    """Run Evidently's drift preset on ``columns``; return its raw ``metrics`` list."""
    import warnings

    from evidently import Report
    from evidently.presets import DataDriftPreset

    preset = DataDriftPreset(columns=columns, num_method="wasserstein", num_threshold=threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Evidently's pandas Period/tz chatter
        snapshot = Report([preset]).run(
            current_data=current[columns], reference_data=reference[columns]
        )
    return snapshot.dict()["metrics"]


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
@dataclass
class ColumnDrift:
    column: str
    score: float
    threshold: float
    drifted: bool
    method: str = ""


@dataclass
class DriftReport:
    as_of: pd.Timestamp
    current_start: pd.Timestamp
    current_end: pd.Timestamp
    current_rows: int
    reference_rows: int
    columns: list[ColumnDrift]
    drift_share_threshold: float
    prediction: ColumnDrift | None = None
    target: ColumnDrift | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_monitored(self) -> int:
        return len(self.columns)

    @property
    def n_drifted(self) -> int:
        return sum(c.drifted for c in self.columns)

    @property
    def share_drifted(self) -> float:
        return self.n_drifted / self.n_monitored if self.n_monitored else 0.0

    @property
    def dataset_drift(self) -> bool:
        return self.n_monitored > 0 and self.share_drifted >= self.drift_share_threshold

    @property
    def prediction_drift(self) -> bool:
        return bool(self.prediction and self.prediction.drifted)

    @property
    def target_drift(self) -> bool:
        return bool(self.target and self.target.drifted)

    @property
    def flag(self) -> bool:
        """The single boolean the retraining trigger consumes."""
        return self.dataset_drift or self.prediction_drift

    @property
    def drifted_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.drifted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "current_window": [self.current_start.isoformat(), self.current_end.isoformat()],
            "current_rows": self.current_rows,
            "reference_rows": self.reference_rows,
            "n_monitored": self.n_monitored,
            "n_drifted": self.n_drifted,
            "share_drifted": self.share_drifted,
            "drift_share_threshold": self.drift_share_threshold,
            "dataset_drift": self.dataset_drift,
            "prediction_drift": self.prediction_drift,
            "target_drift": self.target_drift,
            "flag": self.flag,
            "columns": [asdict(c) for c in self.columns],
            "prediction": asdict(self.prediction) if self.prediction else None,
            "target": asdict(self.target) if self.target else None,
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            f"Drift as of {self.as_of:%Y-%m-%d %H:%M}Z: current {self.current_start:%Y-%m-%d} -> "
            f"{self.current_end:%Y-%m-%d} ({self.current_rows} rows) vs seasonal reference "
            f"({self.reference_rows} rows)",
            f"  features: {self.n_drifted}/{self.n_monitored} drifted "
            f"(share {self.share_drifted:.0%}, threshold {self.drift_share_threshold:.0%}) "
            f"-> dataset drift: {'YES' if self.dataset_drift else 'no'}",
        ]
        for c in self.columns:
            mark = "DRIFT" if c.drifted else "  ok "
            lines.append(
                f"    [{mark}] {c.column:<28} score {c.score:.3f} (threshold {c.threshold})"
            )
        for label, cd in (("prediction", self.prediction), ("target", self.target)):
            if cd is not None:
                lines.append(
                    f"  {label} drift: {'YES' if cd.drifted else 'no'} "
                    f"(score {cd.score:.3f}, threshold {cd.threshold})"
                )
        lines.append(f"  => DRIFT FLAG: {'YES' if self.flag else 'no'}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, ColumnDrift]:
    """Turn Evidently's raw metrics list into ``{column: ColumnDrift}``."""
    out: dict[str, ColumnDrift] = {}
    for m in metrics:
        cfg = m.get("config") or {}
        if not str(cfg.get("type", "")).endswith("ValueDrift"):
            continue
        col = cfg.get("column")
        value = m.get("value")
        if col is None or value is None:
            continue
        score = float(value["drift_score"] if isinstance(value, dict) else value)
        threshold = float(cfg.get("threshold", 0.1))
        out[col] = ColumnDrift(
            column=col,
            score=score,
            threshold=threshold,
            drifted=bool(score > threshold),
            method=str(cfg.get("method", "")),
        )
    return out


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def compute_drift(
    engine: Engine,
    model: LoadedModel | None = None,
    *,
    as_of: pd.Timestamp | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    pad_days: int = DEFAULT_PAD_DAYS,
    drift_share_threshold: float = DEFAULT_DRIFT_SHARE,
    column_threshold: float = DEFAULT_COLUMN_THRESHOLD,
    training: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
    runner: Runner = run_evidently,
) -> DriftReport:
    """Data + prediction + target drift of the last ``window_days`` vs the training season."""
    frame = frame if frame is not None else build_features(engine, output=None)
    as_of = _utc(as_of) if as_of is not None else frame.index.max()
    current = frame[(frame.index > as_of - pd.Timedelta(days=window_days)) & (frame.index <= as_of)]
    if len(current) < MIN_CURRENT_ROWS:
        raise ValueError(f"only {len(current)} current rows (< {MIN_CURRENT_ROWS})")

    if training is None:
        training = pd.read_parquet(FEATURES_PARQUET) if FEATURES_PARQUET.exists() else frame
    reference = seasonal_reference(training, current.index, pad_days=pad_days)
    notes: list[str] = []
    if len(reference) < MIN_CURRENT_ROWS:
        notes.append("seasonal reference too small - falling back to all past training rows")
        reference = training[training.index < current.index.min()]

    columns = drift_columns(list(frame.columns))
    extra: list[str] = []
    if model is not None:
        current = current.assign(**{PREDICTION_COL: model.predict(current)})
        reference = reference.assign(**{PREDICTION_COL: model.predict(reference)})
        extra.append(PREDICTION_COL)
    if TARGET in current.columns and current[TARGET].notna().all():
        extra.append(TARGET)

    per_column = summarize_metrics(runner(reference, current, [*columns, *extra], column_threshold))
    missing = [c for c in columns if c not in per_column]
    if missing:
        notes.append(f"no drift result for: {missing}")

    report = DriftReport(
        as_of=as_of,
        current_start=current.index.min(),
        current_end=current.index.max(),
        current_rows=len(current),
        reference_rows=len(reference),
        columns=[per_column[c] for c in columns if c in per_column],
        drift_share_threshold=drift_share_threshold,
        prediction=per_column.get(PREDICTION_COL),
        target=per_column.get(TARGET),
        notes=notes,
    )
    log.info("drift report:\n%s", report.summary())
    return report


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    from src.models.registry import load_champion

    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(prog="python -m src.monitoring.drift")
    p.add_argument("--as-of", default=None)
    p.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    p.add_argument("--drift-share", type=float, default=DEFAULT_DRIFT_SHARE)
    p.add_argument("--column-threshold", type=float, default=DEFAULT_COLUMN_THRESHOLD)
    p.add_argument("--no-model", action="store_true", help="skip prediction drift")
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    engine = db.get_engine(args.database_url)
    model = None if args.no_model else load_champion(engine)
    report = compute_drift(
        engine,
        model,
        as_of=pd.Timestamp(args.as_of) if args.as_of else None,
        window_days=args.window_days,
        drift_share_threshold=args.drift_share,
        column_threshold=args.column_threshold,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
