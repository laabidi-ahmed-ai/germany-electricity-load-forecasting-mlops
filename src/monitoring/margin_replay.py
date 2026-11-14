"""Margin replay: how much does a newer model differ from an older one over one week?

The post-promotion check (``src.monitoring.promotion``) rolls a new champion back when its MAE
over 7 live target days is more than ``rollback_margin_pct`` above the previous champion's on
the same hours. Seven days are one block: there is no interval and no test to give. This
replay sizes the margin instead, from the past: at each cut point T it trains

* model A on everything up to T - ``gap_days`` (the "previous champion"),
* model B on everything up to T (the "new champion"),

with the champion's training code, and scores both on the ``score_days`` after T - hours
neither has seen. ``worse_pct`` = how much higher B's MAE is than A's (+ = the newer model is
worse). The share of cut points with ``worse_pct`` above the margin is how often the check
would have rolled back a newer model that was simply trained the same way on more data.

What it is not:
* not a significance test - it describes the spread of a one-week paired difference;
* scored with the stored weather (mostly measured), not the forecast weather a live issue
  reads. Both models read the same inputs, so this matters little for their difference;
* B is not selected: a real promotion has also passed the holdout margin and the CV gate, so
  the share overstates the false-rollback rate;
* cut points ``step_days`` apart share training data, so they are not independent draws.

CLI (reads the database, writes nothing)::

    python -m src.monitoring.margin_replay [--cuts 26] [--step-days 14] [--gap-days 28]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

from config.log import configure_logging
from src.data import db
from src.features.build_features import TARGET, build_features
from src.features.horizons import D_PLUS_1, HORIZONS, Horizon
from src.monitoring.metrics import compute_metrics

log = logging.getLogger(__name__)

DEFAULT_CUTS = 26
DEFAULT_STEP_DAYS = 14
DEFAULT_GAP_DAYS = 28
DEFAULT_SCORE_DAYS = 7
MARGINS_PCT: tuple[float, ...] = (3.0, 5.0, 8.0)
COLUMNS = ["cut", "n_hours", "mae_previous", "mae_new", "worse_pct"]


def replay(
    frame: pd.DataFrame,
    horizon: Horizon = D_PLUS_1,
    *,
    cuts: int = DEFAULT_CUTS,
    step_days: int = DEFAULT_STEP_DAYS,
    gap_days: int = DEFAULT_GAP_DAYS,
    score_days: int = DEFAULT_SCORE_DAYS,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """One row per cut point, oldest first -> ``COLUMNS``.

    The last cut leaves exactly ``score_days`` of data after it; earlier ones are
    ``step_days`` apart. A model is fitted once per training cutoff and reused (with
    ``gap_days`` a multiple of ``step_days``, B of one cut is A of a later one).
    """
    from src.models.train import fit_model

    if min(cuts, step_days, gap_days, score_days) < 1:
        raise ValueError("cuts, step_days, gap_days and score_days must be at least 1")
    frame = frame.sort_index()
    last_cut = frame.index.max() - pd.Timedelta(days=score_days)
    cut_points = [last_cut - pd.Timedelta(days=step_days * i) for i in reversed(range(cuts))]

    fitted: dict[pd.Timestamp, tuple[Any, list[str]]] = {}

    def model_trained_to(cutoff: pd.Timestamp) -> tuple[Any, list[str]]:
        if cutoff not in fitted:
            train = frame[frame.index <= cutoff]
            if train.empty:
                raise ValueError(f"no training rows up to {cutoff} - use fewer cuts")
            fitted[cutoff] = fit_model(train, horizon, params)
        return fitted[cutoff]

    rows = []
    for cut in cut_points:
        week = frame[(frame.index > cut) & (frame.index <= cut + pd.Timedelta(days=score_days))]
        previous, cols = model_trained_to(cut - pd.Timedelta(days=gap_days))
        new, _ = model_trained_to(cut)
        mae_previous = compute_metrics(week[TARGET], previous.predict(week[cols]))["mae"]
        mae_new = compute_metrics(week[TARGET], new.predict(week[cols]))["mae"]
        rows.append(
            {
                "cut": cut,
                "n_hours": len(week),
                "mae_previous": mae_previous,
                "mae_new": mae_new,
                "worse_pct": (mae_new - mae_previous) / mae_previous * 100.0,
            }
        )
        log.info("cut %s: newer model %+.1f %% MAE vs older", cut, rows[-1]["worse_pct"])
    return pd.DataFrame(rows, columns=COLUMNS)


def summarize(result: pd.DataFrame, margins: Sequence[float] = MARGINS_PCT) -> dict[str, Any]:
    """Spread of ``worse_pct`` and, per margin, how many cut points exceed it."""
    worse = result["worse_pct"]
    return {
        "n_cuts": len(result),
        "median": float(worse.median()),
        "min": float(worse.min()),
        "max": float(worse.max()),
        "std": float(worse.std(ddof=1)) if len(result) > 1 else 0.0,
        "over": {float(m): int((worse > m).sum()) for m in margins},
    }


def format_summary(result: pd.DataFrame, margins: Sequence[float] = MARGINS_PCT) -> str:
    s = summarize(result, margins)
    lines = [
        f"Margin replay: {s['n_cuts']} cut points, newer vs older model on the week after each",
        f"  newer model's MAE vs the older one's: median {s['median']:+.1f} %, "
        f"range {s['min']:+.1f} to {s['max']:+.1f} %, std {s['std']:.1f} points",
    ]
    for margin, n in s["over"].items():
        lines.append(
            f"  worse by more than {margin:g} %: {n} of {s['n_cuts']} "
            f"({100 * n / s['n_cuts']:.0f} %)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(prog="python -m src.monitoring.margin_replay")
    p.add_argument("--cuts", type=int, default=DEFAULT_CUTS)
    p.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    p.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    p.add_argument("--score-days", type=int, default=DEFAULT_SCORE_DAYS)
    p.add_argument("--horizon", choices=sorted(HORIZONS), default=D_PLUS_1.name)
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    frame = build_features(db.get_engine(args.database_url), output=None)
    result = replay(
        frame,
        HORIZONS[args.horizon],
        cuts=args.cuts,
        step_days=args.step_days,
        gap_days=args.gap_days,
        score_days=args.score_days,
    )
    print(result.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print()
    print(format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
