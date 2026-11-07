"""Fair backtest: replay every D+1 issue with the weather forecast that existed at the time.

Why. ``performance.backtest_vs_official`` predicts its holdout with ``weather_hourly``, where
the reanalysis archive replaces forecast values - measured weather. The official forecast,
and our live one, only ever had weather *forecasts*. This module scores one model, on one set
of hours, three ways:

* ``newest`` (the headline) - per hour, ``_previous_day1`` where that value was public by
  the issue, else ``_previous_day2``. Despite the key, this is *not* the newest run public at
  05:40 (the D-1 00 UTC run the live job reads): Previous Runs keeps each hour only at leads
  just over 24 h and 48 h, and for most hours of D that run's lead lies in between, so the
  archive cannot replay it;
* ``day2`` (sensitivity) - the forecast made two days ahead for every hour, which needs no
  assumption about publication delays;
* ``measured`` - the stored weather (``weather_hourly``), as ``backtest_vs_official`` uses,
  scored on the same hours as the other two. That is reanalysis except for about the last
  week before the run: the archive lags (ingest re-reads 7 days of it), so those hours still
  hold short-lead forecasts.

The issue. The live D+1 job is triggered at 05:40 UTC on the day before the Berlin target day
(``ISSUE_TIME_UTC``; real issues land a minute or two later, so this is the stricter bound).
All 23, 24 or 25 hours of a target day share that issue.

The archived forecasts. Open-Meteo's Previous Runs API (``WeatherClient.fetch_previous_runs``)
returns ``<var>_previous_day<N>``: at hour t, the value of the newest model run initialised
at or before t - N x 24 h (checked against single ICON runs on 6 Nov 2026). Such a run is
public at most ``PUBLICATION_LATENCY`` after its initialisation - Open-Meteo gives 4-6 h for
global models and 1-3 h for regional ones, and the default model mix (best_match, the one the
live forecast reads) blends both, so the slowest is assumed. A value is usable for an issue at
time i when ``t - N x 24 h + latency <= i`` (``choose_offsets`` refuses anything else). For a
05:40 issue that holds for N = 1 only for the first hour or two of the target day; N = 2 holds
for every hour as long as the latency is under 7 h 40 min.

What that means for the result. The live job reads the run of 00 or 03 UTC on the issue day
(lead 20-43 h); the replay reads runs made at least 48 h before most hours, about a day older.
The fair result is therefore a *lower* bound for the model, the measured-weather result an
upper bound, and the live record lies in between. Open-Meteo's Single Runs API would replay the
exact live run, but it is archived only from 2 April 2026, so it cannot cover a year.

No leakage. The model is trained once, exactly as in ``backtest_vs_official``, on rows up to
``as_of - days`` with their archive weather (what the live model trains on). A target day is
scored only if the last training hour could have been published by its issue
(``issue - ACTUALS_PUBLISHED_LAG >= train_end``), so the model has no load information from
after any issue it is scored on. Its last few training days do carry archive weather that the
live job would still have held as forecasts (the archive lags by days): about 5 of some 1,700
training days, and none of it about a scored hour. Only the holdout's five weather columns are
replaced; its load lags (>= 48 h for D+1) and calendar features are already known at the
issue. The same cities, variables and national averaging as live are used
(``src.data.weather_client``).

Uncertainty. Hourly errors are not independent: every hour of a target day shares one issue
and one weather run, and weather-forecast errors come in regimes that last several days. The
95 % interval of the MAE improvement therefore comes from a non-overlapping block bootstrap
over blocks of ``BLOCK_DAYS`` whole target days (a week also holds each weekday once), with
``N_BOOT`` resamples and a fixed seed. With fewer than ``MIN_BOOTSTRAP_BLOCKS`` blocks the
interval is not reliable: no verdict is given and a warning is logged. The summary shows the
evidence for the block length: the autocorrelation of the daily mean loss difference (on a
calendar, so lag k pairs days k apart even across a dropped day), and the narrower interval
that 1-day blocks would give.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.data.weather_client import (
    PREVIOUS_RUN_OFFSETS,
    WEATHER_VARS,
    WeatherClient,
    national_average,
)
from src.features.build_features import TARGET, WEATHER_COLUMNS, build_features
from src.features.horizons import D_PLUS_1, LOCAL_TZ, Horizon
from src.monitoring.metrics import MIN_WINDOW_HOURS, HeadToHead, score_pair
from src.monitoring.performance import read_official, split_for_backtest

log = logging.getLogger(__name__)

ISSUE_TIME_UTC = pd.Timedelta(hours=5, minutes=40)  # the external scheduler starts the job
PUBLICATION_LATENCY = pd.Timedelta(hours=6)  # model initialisation -> public on Open-Meteo
# The actual stamped h (hour-beginning) is complete at h + 1 h, and SMARD publishes it about an
# hour later; so at a 05:40 issue the actuals reach ~04:00 UTC, as src.features.horizons says.
# A target day is scored only if the last training hour was public by its issue.
ACTUALS_PUBLISHED_LAG = pd.Timedelta(hours=2)

NEWEST, DAY2, MEASURED = "newest", "day2", "measured"
MODES: tuple[str, ...] = (NEWEST, DAY2)  # the issue-time weather variants
LABELS = {
    NEWEST: (
        "issue-time weather: _previous_day1 where public by the issue, else _previous_day2 "
        "(headline)"
    ),
    DAY2: "issue-time weather: _previous_day2 for every hour",
    MEASURED: "measured weather (reanalysis), on the same hours as the issue-time rows",
}

BLOCK_DAYS = 7
N_BOOT = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95
MIN_BOOTSTRAP_BLOCKS = 10  # fewer -> the percentile interval means little: no verdict
TOO_FEW_BLOCKS = "too few blocks for a reliable interval"
AUTOCORR_LAGS = 7
MAX_DROPPED_SHARE = 0.01  # warn when more hours than this lack archived forecasts


# --- Issue times and which archived forecast each hour may read ---
def target_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Berlin calendar day (naive midnight) of each UTC hour: the day its issue targets."""
    return index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()


def issue_times(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """05:40 UTC on the day before each hour's Berlin target day."""
    day_before = target_days(index) - pd.Timedelta(days=1)
    return pd.DatetimeIndex((day_before + ISSUE_TIME_UTC).tz_localize("UTC"), name="issue_time")


def choose_offsets(
    index: pd.DatetimeIndex, mode: str, *, latency: pd.Timedelta = PUBLICATION_LATENCY
) -> pd.DataFrame:
    """Per hour: the ``_previous_day<N>`` to read and when that value was public.

    Columns ``issue_time``, ``offset_days``, ``run_init_bound`` (the newest the run behind the
    value can have been initialised) and ``available_by`` (``run_init_bound + latency``).
    ``newest``: N = 1 where that run was public by the issue, else 2. ``day2``: N = 2.
    Raises ``ValueError`` if any chosen value was not public by its issue.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    issue = issue_times(index)
    if mode == NEWEST:
        day1_out = (index - pd.Timedelta(days=1) + latency) <= issue
        offset = np.where(day1_out, 1, 2)
    else:
        offset = np.full(len(index), 2)
    run_init_bound = index - pd.to_timedelta(offset * 24, unit="h")
    out = pd.DataFrame(
        {
            "issue_time": issue,
            "offset_days": offset.astype("int64"),
            "run_init_bound": run_init_bound,
            "available_by": run_init_bound + latency,
        },
        index=index,
    )
    late = (out["available_by"] > out["issue_time"]) | (out["offset_days"] < 1)
    if late.any():
        raise ValueError(
            f"{int(late.sum())} hour(s) would read a forecast published after their issue "
            f"(first {out.index[late][0]}; mode {mode}, latency {latency})"
        )
    return out


def issue_time_weather(
    previous_runs: pd.DataFrame,
    index: pd.DatetimeIndex,
    mode: str,
    *,
    n_cities: int,
    latency: pd.Timedelta = PUBLICATION_LATENCY,
) -> pd.DataFrame:
    """The national-average weather each hour's issue could read, and where it came from.

    ``previous_runs`` is ``WeatherClient.fetch_previous_runs`` output. Returns the
    ``choose_offsets`` columns, ``WEATHER_COLUMNS`` and ``complete`` (every one of the
    ``n_cities`` cities has every variable for the chosen offset), indexed like ``index``.
    """
    out = choose_offsets(index, mode, latency=latency)
    for col in WEATHER_COLUMNS:
        out[col] = np.nan
    out["complete"] = False
    for n, rows in previous_runs.groupby("offset_days"):
        rows_here = out["offset_days"] == n
        if not rows_here.any():
            continue
        avg = national_average(rows).set_index("timestamp_utc").reindex(index)
        cells = rows.groupby("timestamp_utc")[list(WEATHER_VARS)].count().sum(axis=1)
        full = cells.reindex(index).eq(n_cities * len(WEATHER_VARS)).to_numpy()
        for col in WEATHER_COLUMNS:
            out.loc[rows_here, col] = avg.loc[rows_here.to_numpy(), col].to_numpy()
        out.loc[rows_here, "complete"] = full[rows_here.to_numpy()]
    out["complete"] = out["complete"].astype(bool)
    return out


# --- Uncertainty: block bootstrap on the paired hourly errors ---
def target_day_blocks(index: pd.DatetimeIndex, block_days: int = BLOCK_DAYS) -> np.ndarray:
    """Block id per hour: consecutive runs of ``block_days`` whole Berlin target days."""
    days = target_days(index)
    return np.asarray((days - days.min()).days // block_days)


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate and percentile interval of the model's MAE gain over the official one."""

    improvement_pct: float  # (official MAE - model MAE) / official MAE x 100
    low: float
    high: float
    diff_mw: float  # official MAE - model MAE
    diff_low: float
    diff_high: float
    n_blocks: int
    n_boot: int
    seed: int
    level: float

    @property
    def reliable(self) -> bool:
        """At least ``MIN_BOOTSTRAP_BLOCKS`` blocks were resampled."""
        return self.n_blocks >= MIN_BOOTSTRAP_BLOCKS

    @property
    def verdict(self) -> str:
        if not self.reliable:
            return TOO_FEW_BLOCKS
        if self.low > 0:
            return "clearly beats"
        if self.high < 0:
            return "clearly loses to"
        return "does not clearly beat"


def block_bootstrap_ci(
    model_ae: np.ndarray | pd.Series,
    official_ae: np.ndarray | pd.Series,
    blocks: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    level: float = CI_LEVEL,
) -> BootstrapCI:
    """Resample whole blocks with replacement (as many as there are) and recompute the gain.

    Each resample's gain is sum(official_ae - model_ae) / sum(official_ae) over the hours
    of its blocks - the MAE improvement % of that resample. Same seed, same interval.
    """
    m = np.asarray(model_ae, dtype="float64")
    o = np.asarray(official_ae, dtype="float64")
    b = np.asarray(blocks)
    if not len(m) == len(o) == len(b) or len(m) == 0:
        raise ValueError("model_ae, official_ae and blocks must be non-empty and equally long")
    _, inv = np.unique(b, return_inverse=True)
    k = int(inv.max()) + 1
    if k < MIN_BOOTSTRAP_BLOCKS:
        log.warning(
            "only %d block(s) to resample (fewer than %d): the bootstrap interval is not "
            "reliable and no verdict is given",
            k,
            MIN_BOOTSTRAP_BLOCKS,
        )
    diff_sum = np.bincount(inv, weights=o - m, minlength=k)
    off_sum = np.bincount(inv, weights=o, minlength=k)
    count = np.bincount(inv, minlength=k).astype("float64")

    draw = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    d, s, c = diff_sum[draw].sum(axis=1), off_sum[draw].sum(axis=1), count[draw].sum(axis=1)
    alpha = (1 - level) / 2
    pct_low, pct_high = np.quantile(100 * d / s, [alpha, 1 - alpha])
    mw_low, mw_high = np.quantile(d / c, [alpha, 1 - alpha])
    return BootstrapCI(
        improvement_pct=float(100 * (o - m).sum() / o.sum()),
        low=float(pct_low),
        high=float(pct_high),
        diff_mw=float((o - m).mean()),
        diff_low=float(mw_low),
        diff_high=float(mw_high),
        n_blocks=k,
        n_boot=n_boot,
        seed=seed,
        level=level,
    )


def daily_loss_autocorr(
    model_ae: pd.Series, official_ae: pd.Series, lags: int = AUTOCORR_LAGS
) -> list[float]:
    """Autocorrelation of the daily mean loss difference (official - model) at 1..``lags`` days.

    The daily series is put on a calendar first (``asfreq("D")``): a target day with no scored
    hours becomes a gap, so lag k always pairs days that are k calendar days apart.
    """
    daily = (official_ae - model_ae).groupby(target_days(model_ae.index)).mean().asfreq("D")
    return [float(daily.autocorr(lag)) for lag in range(1, lags + 1)]


# --- Report ---
@dataclass(kw_only=True)
class FairRow(HeadToHead):
    """One weather variant: the head-to-head on the scored hours, its interval, days won."""

    key: str
    label: str
    n_hours: int
    days_won: int
    n_days: int
    ci: BootstrapCI

    def line(self) -> str:
        return _table_row(
            self.label,
            f"{self.model_mae:,.0f} MW / {self.model_mape:.2f} %",
            f"{self.official_mae:,.0f} MW / {self.official_mape:.2f} %",
            f"{self.improvement_pct:+.1f} %",
            f"[{self.ci.low:+.1f}, {self.ci.high:+.1f}] %",
            f"{self.days_won}/{self.n_days}",
        )


# Label left-aligned (as wide as the longest label), the rest right-aligned.
_COLUMN_WIDTHS = (max(map(len, LABELS.values())) + 2, 20, 22, 10, 24, 9)


def _table_row(*cells: str) -> str:
    label, *rest = cells
    return f"{label:<{_COLUMN_WIDTHS[0]}}" + "".join(
        f"{cell:>{width}}" for cell, width in zip(rest, _COLUMN_WIDTHS[1:], strict=True)
    )


def score_variant(
    hours: pd.DataFrame, key: str, *, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED
) -> FairRow:
    """Score the prediction column ``key`` of ``hours`` against the official forecast."""
    s = score_pair(hours.rename(columns={key: "model_mw"}))
    model_ae = (hours[key] - hours["actual_mw"]).abs()
    official_ae = (hours["official_mw"] - hours["actual_mw"]).abs()
    ci = block_bootstrap_ci(
        model_ae, official_ae, target_day_blocks(hours.index), n_boot=n_boot, seed=seed
    )
    by_day = (
        pd.DataFrame({"model": model_ae, "official": official_ae})
        .groupby(target_days(hours.index))
        .mean()
    )
    return FairRow(
        key=key,
        label=LABELS[key],
        n_hours=len(hours),
        days_won=int((by_day["model"] < by_day["official"]).sum()),
        n_days=len(by_day),
        ci=ci,
        **s,
    )


@dataclass(kw_only=True)
class FairBacktestReport:
    start: pd.Timestamp  # first and last scored hour (UTC)
    end: pd.Timestamp
    n_hours: int
    n_days: int  # Berlin target days
    n_dropped: int  # hours without complete archived forecasts for both variants
    train_rows: int
    train_end: pd.Timestamp
    latency: pd.Timedelta
    rows: dict[str, FairRow]  # NEWEST, DAY2, MEASURED
    day_block_ci: BootstrapCI  # the headline with 1-day blocks, for comparison
    autocorr: list[float]  # daily mean loss difference, lags 1..AUTOCORR_LAGS days
    hours: pd.DataFrame  # per scored hour: actual, official, predictions, issue, offsets
    weather: dict[str, pd.DataFrame]  # per variant: the weather fed to the model + its origin

    @property
    def headline(self) -> FairRow:
        return self.rows[NEWEST]

    def summary(self) -> str:
        h = self.headline
        days = target_days(pd.DatetimeIndex([self.start, self.end]))
        n_day1 = int((self.hours[f"offset_{NEWEST}"] == 1).sum())
        acf = " ".join(f"{v:+.2f}" for v in self.autocorr)
        dropped = f"{self.n_dropped} dropped for missing archived forecasts"
        lines = [
            "Fair backtest: D+1 issues replayed with the weather forecast available at "
            f"{_hhmm(ISSUE_TIME_UTC)} UTC on D-1 (Open-Meteo Previous Runs, default model mix)",
            f"  target days {days[0]:%Y-%m-%d} -> {days[1]:%Y-%m-%d} ({self.n_days} days, "
            f"{self.n_hours:,} hours; {dropped})",
            f"  model trained once on {self.train_rows:,} rows up to "
            f"{self.train_end:%Y-%m-%d %H:%M} UTC; forecast publication latency assumed "
            f"{self.latency / pd.Timedelta(hours=1):g} h",
            "  "
            + _table_row(
                "weather",
                "model MAE / MAPE",
                "official MAE / MAPE",
                "MAE lower",
                f"{h.ci.level:.0%} CI ({BLOCK_DAYS}-day blocks)",
                "days won",
            ),
            *(f"  {self.rows[k].line()}" for k in (NEWEST, DAY2, MEASURED)),
            self._verdict_line(),
            "     bracket: issue-time weather is about a day older than the live job's (a lower "
            "bound for the model); measured weather is an upper bound",
            f"  headline hours on _previous_day1: {n_day1:,} of {self.n_hours:,} "
            f"({100 * n_day1 / self.n_hours:.1f} %); all others _previous_day2",
            f"  bootstrap: {h.ci.n_boot:,} resamples of {h.ci.n_blocks} blocks of {BLOCK_DAYS} "
            f"target days, seed {h.ci.seed}, percentile interval; with 1-day blocks the headline "
            f"CI would be [{self.day_block_ci.low:+.1f}, {self.day_block_ci.high:+.1f}] %",
            f"  daily mean loss difference (official - model), autocorrelation at lags "
            f"1-{AUTOCORR_LAGS} days: {acf}",
        ]
        return "\n".join(lines)

    def _verdict_line(self) -> str:
        h = self.headline
        gain = (
            f"{h.improvement_pct:+.1f} % MAE, {h.ci.level:.0%} CI "
            f"[{h.ci.low:+.1f}, {h.ci.high:+.1f}] % ({h.ci.diff_mw:+,.0f} MW, "
            f"[{h.ci.diff_low:+,.0f}, {h.ci.diff_high:+,.0f}] MW)"
        )
        if h.ci.reliable:
            return (
                f"  -> with issue-time weather the model {h.ci.verdict} the official "
                f"forecast: {gain}"
            )
        return (
            f"  -> {TOO_FEW_BLOCKS} ({h.ci.n_blocks} blocks of {BLOCK_DAYS} target days, need "
            f"at least {MIN_BOOTSTRAP_BLOCKS}), so no verdict; with issue-time weather: {gain}"
        )

    def monthly(self) -> pd.DataFrame:
        """MAE (MW) per Berlin target month: official and the model with each weather."""
        hrs = self.hours
        month = target_days(hrs.index).to_period("M").astype(str)
        err = pd.DataFrame(
            {
                "official": (hrs["official_mw"] - hrs["actual_mw"]).abs(),
                **{k: (hrs[k] - hrs["actual_mw"]).abs() for k in (NEWEST, DAY2, MEASURED)},
            },
            index=hrs.index,
        )
        out = err.groupby(month.to_numpy()).mean()
        out.insert(0, "hours", err.groupby(month.to_numpy()).size())
        return out.rename_axis("month").reset_index()


def _hhmm(delta: pd.Timedelta) -> str:
    minutes = int(delta / pd.Timedelta(minutes=1))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _measured_line(row: FairRow, start: pd.Timestamp, end: pd.Timestamp) -> str:
    first, last = target_days(pd.DatetimeIndex([start, end]))
    note = "" if row.ci.reliable else f"; {TOO_FEW_BLOCKS}"
    return (
        f"Measured-weather backtest (reanalysis weather; scored before the issue-time weather is "
        f"fetched, and again below on exactly the issue-time rows' hours): target days "
        f"{first:%Y-%m-%d} -> {last:%Y-%m-%d}, {row.n_hours:,} hours: model MAE "
        f"{row.model_mae:,.0f} MW / {row.model_mape:.2f} %, official {row.official_mae:,.0f} MW / "
        f"{row.official_mape:.2f} % -> {row.improvement_pct:+.1f} % MAE "
        f"({row.ci.level:.0%} CI [{row.ci.low:+.1f}, {row.ci.high:+.1f}] %, {BLOCK_DAYS}-day "
        f"blocks{note})"
    )


# --- The backtest ---
def fair_backtest_vs_official(
    engine: Engine,
    *,
    days: int = 365,
    as_of: pd.Timestamp | None = None,
    params: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    horizon: Horizon = D_PLUS_1,
    weather_client: WeatherClient | None = None,
    latency: pd.Timedelta = PUBLICATION_LATENCY,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    progress: Callable[[str], None] | None = None,
) -> FairBacktestReport:
    """Train once before the window; score it with issue-time and with measured weather.

    The measured-weather row is scored and passed to ``progress`` (default: the log)
    *before* the archived forecasts are fetched, so an Open-Meteo outage still leaves the
    old number on record; the fetch failure is then raised as ``RuntimeError``.
    """
    from src.models.train import fit_model

    if not horizon.calendar_day:
        raise ValueError(f"the fair backtest replays the D+1 issue; {horizon.name} has none")
    say = progress or (lambda line: log.info("%s", line))

    frame = frame if frame is not None else build_features(engine, output=None)
    train, holdout = split_for_backtest(frame, days, as_of)
    train_end = train.index.max()
    model, cols = fit_model(train, horizon, params)

    # The model must not have seen anything the live job could not have known at the issue.
    holdout = holdout[issue_times(holdout.index) - ACTUALS_PUBLISHED_LAG >= train_end]
    if holdout.empty:
        raise ValueError("no holdout target day is issued after the last training hour is public")
    official = read_official(engine, holdout.index.min(), holdout.index.max())
    hours = holdout[[TARGET]].rename(columns={TARGET: "actual_mw"})
    hours = hours.join(official.rename("official_mw"), how="inner")
    if len(hours) < MIN_WINDOW_HOURS:
        raise ValueError(f"only {len(hours)} hours have an official forecast in the window")
    holdout = holdout.loc[hours.index]
    hours["issue_time"] = issue_times(hours.index)
    hours[MEASURED] = model.predict(holdout[cols])
    say(
        _measured_line(
            score_variant(hours, MEASURED, n_boot=n_boot, seed=seed),
            hours.index.min(),
            hours.index.max(),
        )
    )

    # Only now the network.
    client = weather_client or WeatherClient()
    try:
        runs = client.fetch_previous_runs(
            hours.index.min(), hours.index.max(), offsets=PREVIOUS_RUN_OFFSETS
        )
    except Exception as err:
        raise RuntimeError(f"Previous Runs weather fetch failed: {err}") from err

    weather_cols = [c for c in WEATHER_COLUMNS if c in cols]
    weather: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        w = issue_time_weather(
            runs, hours.index, mode, n_cities=len(client.cities), latency=latency
        )
        X = holdout[cols].copy()
        X[weather_cols] = w[weather_cols]
        hours[mode] = model.predict(X)
        for col in ("offset_days", "run_init_bound", "available_by"):
            hours[f"{col.removesuffix('_days')}_{mode}"] = w[col]
        weather[mode] = w

    complete = weather[NEWEST]["complete"] & weather[DAY2]["complete"]
    n_dropped = int((~complete).sum())
    if n_dropped > MAX_DROPPED_SHARE * len(hours):
        log.warning(
            "%d of %d hours lack complete archived forecasts and are left out of every row",
            n_dropped,
            len(hours),
        )
    hours = hours[complete.to_numpy()]
    if len(hours) < MIN_WINDOW_HOURS:
        raise ValueError(f"only {len(hours)} hours have complete issue-time weather")
    weather = {mode: w.loc[hours.index] for mode, w in weather.items()}

    rows = {k: score_variant(hours, k, n_boot=n_boot, seed=seed) for k in (NEWEST, DAY2, MEASURED)}
    model_ae = (hours[NEWEST] - hours["actual_mw"]).abs()
    official_ae = (hours["official_mw"] - hours["actual_mw"]).abs()
    day_block_ci = block_bootstrap_ci(
        model_ae, official_ae, target_day_blocks(hours.index, 1), n_boot=n_boot, seed=seed
    )
    autocorr = daily_loss_autocorr(model_ae, official_ae)

    report = FairBacktestReport(
        start=hours.index.min(),
        end=hours.index.max(),
        n_hours=len(hours),
        n_days=rows[NEWEST].n_days,
        n_dropped=n_dropped,
        train_rows=len(train),
        train_end=train_end,
        latency=latency,
        rows=rows,
        day_block_ci=day_block_ci,
        autocorr=autocorr,
        hours=hours,
        weather=weather,
    )
    h = report.headline
    log.info(  # one line: callers print the full summary (performance CLI)
        "fair backtest: %d hours, model %+.1f %% MAE vs official, %s CI [%+.1f, %+.1f] %%",
        report.n_hours,
        h.improvement_pct,
        f"{h.ci.level:.0%}",
        h.ci.low,
        h.ci.high,
    )
    return report
