"""Performance monitoring: model vs actuals vs the official day-ahead forecast.

``compute_report`` scores the model's stored day-ahead forecast and the official
forecast against the actuals on exactly the same hours (inner join of the three
tables) over rolling 7-day and 30-day windows ending at ``as_of``, plus a per-day
series for the dashboard. The metric code is shared with the dashboard in
``src.monitoring.metrics``.

``backtest_vs_official`` covers the time before live history exists: it trains a
LightGBM on all data strictly before the last ``days`` days, predicts that window with
day-ahead features and scores it against the official forecast on the same hours. It
never uses the champion itself, which has seen those hours.

CLI::

    python -m src.monitoring.performance [--as-of ...] [--backtest-days 30]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine

from config.log import configure_logging
from src.data import db
from src.data.db import OFFICIAL_SOURCE
from src.features.build_features import TARGET, build_features
from src.features.horizons import DAY_AHEAD, select_features
from src.monitoring.metrics import (
    MIN_WINDOW_HOURS,
    WindowMetrics,
    aligned_frame,
    daily_metrics,
    official_accuracy,
    score_pair,
    window_metrics,
)

log = logging.getLogger(__name__)

WINDOWS_DAYS: tuple[int, ...] = (7, 30)


@dataclass
class PerformanceReport:
    as_of: pd.Timestamp
    official_source: str
    windows: list[WindowMetrics]
    daily: pd.DataFrame
    n_aligned_hours: int
    model_versions: list[str] = field(default_factory=list)
    official_only: dict[int, dict[str, Any]] = field(default_factory=dict)

    def window(self, days: int) -> WindowMetrics | None:
        return next((w for w in self.windows if w.window_days == days), None)

    @property
    def has_data(self) -> bool:
        return any(w.judged for w in self.windows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "official_source": self.official_source,
            "n_aligned_hours": self.n_aligned_hours,
            "model_versions": self.model_versions,
            "windows": [
                {**asdict(w), "start": w.start.isoformat(), "end": w.end.isoformat()}
                for w in self.windows
            ],
            "official_only": self.official_only,
        }

    def summary(self) -> str:
        lines = [
            f"Performance as of {self.as_of:%Y-%m-%d %H:%M}Z "
            f"(official benchmark: {self.official_source}, "
            f"model versions scored: {self.model_versions or '-'})"
        ]
        if not self.has_data:
            lines.append(
                f"  no scorable model forecasts yet ({self.n_aligned_hours} aligned hours) - "
                "the live head-to-head starts once forecast hours receive actuals."
            )
        for w in self.windows:
            if not w.judged:
                lines.append(
                    f"  {w.window_days:>2}d: {w.n_hours} aligned hours (< {MIN_WINDOW_HOURS}) - not judged"
                )
                continue
            verdict = "BEATS" if w.model_beats_official else "LOSES TO"
            lines.append(
                f"  {w.window_days:>2}d ({w.n_hours}h): model MAE {w.model_mae:,.0f} MW / "
                f"{w.model_mape:.2f}%  vs official {w.official_mae:,.0f} MW / "
                f"{w.official_mape:.2f}%  -> model {verdict} official ({w.improvement_pct:+.1f}%)"
            )
        for days, m in self.official_only.items():
            lines.append(
                f"  official forecast alone, last {days}d ({m['n_hours']}h): "
                f"MAE {m['mae']:,.0f} MW / {m['mape']:.2f}%"
            )
        return "\n".join(lines)


def compute_report(
    engine: Engine,
    *,
    as_of: pd.Timestamp | None = None,
    windows: tuple[int, ...] = WINDOWS_DAYS,
    model_version: str | None = None,
    official_source: str = OFFICIAL_SOURCE,
) -> PerformanceReport:
    as_of = db.to_utc(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").floor("h")
    start = as_of - pd.Timedelta(days=max(windows))
    aligned = aligned_frame(
        engine,
        start=start,
        end=as_of,
        model_version=model_version,
        official_source=official_source,
    )
    aligned = aligned[aligned["timestamp_utc"] > start].reset_index(drop=True)
    report = PerformanceReport(
        as_of=as_of,
        official_source=official_source,
        windows=[window_metrics(aligned, as_of, d) for d in windows],
        daily=daily_metrics(aligned),
        n_aligned_hours=len(aligned),
        model_versions=[] if aligned.empty else sorted(aligned["model_version"].unique()),
    )
    for d in windows:
        oa = official_accuracy(engine, as_of, d, official_source=official_source)
        if oa["mae"] is not None:
            report.official_only[d] = oa
    log.info("performance report:\n%s", report.summary())
    return report


# --- Out-of-sample backtest vs the official forecast ---
@dataclass
class BacktestReport:
    start: pd.Timestamp
    end: pd.Timestamp
    n_hours: int
    train_rows: int
    train_end: pd.Timestamp
    model_mae: float
    model_mape: float
    official_mae: float
    official_mape: float
    model_beats_official: bool
    improvement_pct: float
    daily: pd.DataFrame
    by_hour: pd.DataFrame  # hour-of-day (local) breakdown

    def summary(self) -> str:
        verdict = "BEATS" if self.model_beats_official else "LOSES TO"
        days_won = int((self.daily["model_mae"] < self.daily["official_mae"]).sum())
        return (
            f"Out-of-sample backtest {self.start:%Y-%m-%d} -> {self.end:%Y-%m-%d} "
            f"({self.n_hours}h; model trained on {self.train_rows:,} rows up to {self.train_end:%Y-%m-%d}):\n"
            f"  model    MAE {self.model_mae:,.0f} MW / MAPE {self.model_mape:.2f}%\n"
            f"  official MAE {self.official_mae:,.0f} MW / MAPE {self.official_mape:.2f}%\n"
            f"  -> model {verdict} the official forecast ({self.improvement_pct:+.1f}% MAE); "
            f"model wins {days_won}/{len(self.daily)} days"
        )


def backtest_vs_official(
    engine: Engine,
    *,
    days: int = 30,
    as_of: pd.Timestamp | None = None,
    official_source: str = OFFICIAL_SOURCE,
    params: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
) -> BacktestReport:
    from src.models.train import LightGBMForecaster

    frame = frame if frame is not None else build_features(engine, output=None)
    as_of = db.to_utc(as_of) if as_of is not None else frame.index.max()
    holdout_start = as_of - pd.Timedelta(days=days)
    train = frame[frame.index <= holdout_start]
    holdout = frame[(frame.index > holdout_start) & (frame.index <= as_of)]
    if len(train) < 24 * 365 or holdout.empty:
        raise ValueError("not enough data for a backtest (need >= 1 year of training rows)")

    cols = select_features(frame.columns, DAY_AHEAD)
    model = LightGBMForecaster(params).fit(train[cols], train[TARGET])
    pred = pd.Series(model.predict(holdout[cols]), index=holdout.index, name="model_mw")

    official = db.read_table(
        engine,
        db.LoadForecastOfficial,
        start=holdout.index.min(),
        end=holdout.index.max(),
        source=official_source,
    ).set_index("timestamp_utc")["forecast_mw"]
    aligned = pd.DataFrame({"actual_mw": holdout[TARGET], "model_mw": pred}).join(
        official.rename("official_mw"), how="inner"
    )
    if len(aligned) < MIN_WINDOW_HOURS:
        raise ValueError(f"only {len(aligned)} hours have an official forecast in the window")
    aligned = aligned.rename_axis("timestamp_utc").reset_index()
    s = score_pair(aligned)

    local_hour = aligned["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.hour
    by_hour = (
        aligned.assign(
            hour=local_hour,
            model_ae=(aligned["model_mw"] - aligned["actual_mw"]).abs(),
            official_ae=(aligned["official_mw"] - aligned["actual_mw"]).abs(),
        )
        .groupby("hour")[["model_ae", "official_ae"]]
        .mean()
        .rename(columns={"model_ae": "model_mae", "official_ae": "official_mae"})
        .reset_index()
    )
    report = BacktestReport(
        start=aligned["timestamp_utc"].min(),
        end=aligned["timestamp_utc"].max(),
        n_hours=len(aligned),
        train_rows=len(train),
        train_end=train.index.max(),
        daily=daily_metrics(aligned),
        by_hour=by_hour,
        **s,
    )
    log.info("backtest:\n%s", report.summary())
    return report


# --- CLI ---
def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(prog="python -m src.monitoring.performance")
    p.add_argument("--as-of", default=None)
    p.add_argument("--model-version", default=None)
    p.add_argument("--official-source", default=OFFICIAL_SOURCE)
    p.add_argument("--backtest-days", type=int, default=0, help="also run an OOS backtest")
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    engine = db.get_engine(args.database_url)
    report = compute_report(
        engine,
        as_of=pd.Timestamp(args.as_of) if args.as_of else None,
        model_version=args.model_version,
        official_source=args.official_source,
    )
    print(report.summary())
    if args.backtest_days:
        bt = backtest_vs_official(
            engine,
            days=args.backtest_days,
            as_of=pd.Timestamp(args.as_of) if args.as_of else None,
            official_source=args.official_source,
        )
        print()
        print(bt.summary())
        print("\nPer day (MAE, MW):")
        print(bt.daily.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
