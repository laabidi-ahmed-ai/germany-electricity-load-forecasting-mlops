"""Promotion policy around champion / challenger (README §9, §16.12).

Cooldown. After a challenger is rejected - a real ``kept``, by the margin or by the CV gate
(``src.monitoring.retrain``) - the monitor trains no new challenger for ``days`` UTC
calendar days. The triggers are still evaluated, scored and recorded on every pass; only
the comparison waits. A pass the cooldown holds records a ``retrain`` event with decision
``cooldown``.

Why. A trigger that keeps firing (a hard week, persistent drift) used to retrain every day:
five ``kept`` retrains between 5 and 11 October 2026 (README §16.3). Each retry compared an
almost identical candidate with the same champion on a holdout that had moved by one day
out of 14 - not new evidence, another draw of the same noise. With enough draws, one passes
the 3 % margin by luck: multiple testing. The cooldown bounds the looks to about one a week,
and each new look has at least 7 new days in its 14-day holdout. It does not make the looks
independent, and it does not give the margin a fixed error rate.

Calendar days, not 7 x 24 h: the daily pass starts at a slightly different time each day,
and a pass a few minutes earlier than the rejecting one would otherwise be held one more
day. A rejection on day D holds the passes of days D to D + days - 1; the pass of day
D + days is free.

What starts it: a real rejection, or a rollback (below). ``skipped`` compared nothing, and
a dry run's decision (``kept (dry-run)``) must not change what later passes do. ``--force``
bypasses the cooldown; the bypass is logged and stored with the pass.

Only against the champion that serves now. A rejection says "this champion is not beaten
by this recipe on this data"; it says nothing about another champion. So it counts only if
it was decided against the current champion's version (the event's ``model_version``) and
that champion has not been promoted since (its ``promoted_at`` in ``model_artifacts``, which
also catches a version promoted again by hand). Any promotion - automatic, forced, or by hand
through ``bootstrap`` or ``registry --promote`` - therefore ends a running cooldown. A
rollback holds the champion it restored, by the same rule: the rollback's own promotion
comes just before its event, a later one ends the cooldown.

Post-promotion check. Neither gate of a promotion tests the model that serves: it is the
winning recipe refitted on all data, holdout included (``src.monitoring.retrain``). Its first
out-of-sample evidence is its live record, so that is what is checked:

* which promotion: the newest ``retrain`` / ``promoted`` event, if no ``post_promotion``
  event is newer, its challenger is the champion that serves now, and that champion has not
  been promoted again since. One verdict per promotion. A champion promoted by hand
  (``bootstrap``, ``registry --promote``) is never checked: that is a human decision, made on
  more evidence than a week, it is how a horizon or the training code changes, and it is how
  a rollback is undone - the job must never fight the operator;
* which hours: the new champion's scored hours - the latest issue made by its deadline, with
  an actual and an official forecast - on its first ``days`` complete Berlin target days. A
  day is complete when it is over and at least ``MIN_DAY_COVERAGE`` of its 23, 24 or 25 hours
  are scored; a shorter day is skipped, not counted. Only issues made after the promotion
  count, and the wait is measured from it (``promoted_at`` in ``model_artifacts``; the
  promoting event's time only if that is missing);
* against what: the previous champion, loaded from the database export
  (``registry.load_version``). Both champions are replayed on exactly those hours from the
  same rebuilt inputs (below), and the verdict compares the two replays;
* the rule: if the new champion's replayed MAE is more than ``margin_pct`` above the
  previous one's, the previous champion is promoted back through ``registry.promote`` (MLflow
  alias first, database pointer last), a ``post_promotion`` / ``rolled_back`` event is
  recorded and the cooldown starts. Otherwise the event says ``confirmed``. ``inconclusive``
  closes a check that cannot be made fairly: another horizon (a deliberate horizon change is
  never rolled back), a pruned or corrupt export, more than ``MAX_UNREPLAYABLE_SHARE`` of the
  hours that cannot be rebuilt, or too few complete days within ``max_wait_days``. Nothing is
  rolled back on doubt;
* failures: any other error loading or replaying the previous champion fails the job (red),
  and the next pass tries again; after ``max_wait_days`` it closes the check
  ``inconclusive`` with the error as the reason, so a broken export cannot hold retraining
  for ever;
* a rollback whose event could not be written: the pass still trains nothing and fails. The
  next pass sees the champion is the previous version of the latest automatic promotion,
  promoted after it, with no verdict since, and records the missing ``rolled_back`` verdict,
  marked as recovered, dated at that promotion - so the record and the cooldown are right.
  It cannot tell that case from the previous version promoted back by hand before the
  verdict; the reason says so.

``days`` complete days are one block of weakly dependent days: there is no interval to give
and no significance test to claim. The margin is a practical threshold for "clearly worse",
sized with ``src.monitoring.margin_replay`` (README §16.12).

The replay. The previous champion issued no forecast after it was demoted, so both
champions' forecasts are rebuilt per issue of the new champion, from what that issue read:

* weather: the per-city rows stored with the issue (``load_forecast_model_weather``), through
  the same ``national_average`` - never ``weather_hourly``, where the archive has since
  replaced the forecasts;
* load: ``load_actual`` cut at the issue time, the target hours appended with an unknown
  load, then ``build_feature_frame`` - the serving code path;
* calendar: known for any hour.

No leakage, in five steps. (1) A horizon-valid feature uses load at least ``min_lag_hours``
old, and the live issue exists only because ``serving.forecast.target_window`` accepted it,
so every lag it needs is stamped at or before that issue's latest actual. (2) The replay cuts
load at the issue time anyway: a lag that would need later load is missing and the hour is
dropped, never filled. (3) Weather comes only from the issue's own stored rows. (4) Both
models' ``train_end`` must lie before the first scored hour, or the comparison is refused.
(5) The actuals of the scored hours are used for scoring only.

Why both are replayed. ``load_actual`` keeps one value per hour: the hourly re-ingest
overwrites it in place and re-stamps ``ingested_at``, and SMARD revises recent values. So the
replay reads today's value of each pre-issue hour, not the value the issue read, and that
cannot be recovered. On the Neon rehearsal (v3 vs v2, 7 days from 5 Nov 2026) only 103 of
168 hours of v3's replay matched its stored forecast within 1 MW (largest difference
2,278 MW): the lag inputs had been revised after the issues. Comparing v3's live forecasts
with v2's replay would compare two models on different inputs; dropping the hours that
differ would leave too few. Replaying both on the same rebuilt inputs compares the models
and nothing else, which is also how ``margin_replay`` sized the margin (two models scored on
the same stored inputs). A revision is a later value of an hour before the issue, never of a
scored hour, and both models read it.

Parity is kept as information, not a gate: how many hours the new champion's replay
reproduces its stored forecast within ``PARITY_TOLERANCE_MW``, the largest difference, and
the new champion's live MAE next to its replayed MAE on the same hours, so the effect of the
revisions is visible. An exact replay needs each issue's load inputs stored with the issue,
like its weather (a migration; README §16.12).

Report only (reads the database, writes nothing; any two exported versions of one horizon)::

    python -m src.monitoring.promotion --version 3 --previous 2 [--since 2026-11-04] [--days 7]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from config.log import configure_logging
from src.data import db
from src.data.weather_client import national_average
from src.features.build_features import TARGET, build_feature_frame
from src.features.horizons import LOCAL_TZ, Horizon, most_recent_load_lag, select_features
from src.models import registry
from src.models.registry import LoadedModel
from src.monitoring.metrics import (
    actual_vs_official,
    compute_metrics,
    improvement_pct,
    latest_on_time_issues,
    model_issues,
)

log = logging.getLogger(__name__)

DEFAULT_COOLDOWN_DAYS = 7
DEFAULT_POST_CHECK_DAYS = 7
DEFAULT_ROLLBACK_MARGIN_PCT = 5.0
DEFAULT_MAX_WAIT_DAYS = 21

PARITY_TOLERANCE_MW = 1.0  # the new champion's replay vs its stored forecast (reported only)
MIN_DAY_COVERAGE = 0.90  # share of a target day's hours that must be scored for it to count
MAX_UNREPLAYABLE_SHARE = 0.10  # more scored hours than this not rebuildable -> no verdict
REPLAY_HISTORY = pd.Timedelta(days=8)  # load before the first target hour (lag 168 h + slack)

NO_WEATHER = "no weather stored with the issue"
NO_LAG = "a load lag is missing"

NOT_APPLICABLE, PENDING, CONFIRMED, ROLL_BACK, INCONCLUSIVE = (
    "not_applicable",
    "pending",
    "confirmed",
    "roll_back",
    "inconclusive",
)
# verdict -> the decision of the post_promotion event that closes the check
_DECISIONS = {
    CONFIRMED: db.DECISION_CONFIRMED,
    ROLL_BACK: db.DECISION_ROLLED_BACK,
    INCONCLUSIVE: db.DECISION_INCONCLUSIVE,
}


# --- Cooldown ---
@dataclass(frozen=True)
class Cooldown:
    """Whether a new challenger may be trained at ``as_of``, and which event says not."""

    active: bool
    days: int
    rejected_at: pd.Timestamp | None = None  # as_of of the rejecting (or rollback) pass
    until: pd.Timestamp | None = None  # first moment a new challenger may be trained
    rejection: str | None = None  # the reason of that event
    ended_by: str | None = None  # why an event on record does not hold this champion
    cause: str = "challenger rejected"
    rollback: bool = False  # the event is a rollback, not a rejected challenger

    def detail(self) -> str:
        if self.rejected_at is None:
            return "no rejected challenger on record"
        if self.ended_by is not None:
            what = "rollback" if self.rollback else "rejection"
            return f"latest {what} ({self.rejected_at:%Y-%m-%d}) {self.ended_by}"
        return (
            f"{self.cause} on {self.rejected_at:%Y-%m-%d %H:%M}Z ({self.rejection}); "
            f"no new challenger before {self.until:%Y-%m-%d} UTC ({self.days}-day cooldown)"
        )


def _details(event: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(event.get("details") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _one_cooldown(
    event: dict[str, Any],
    as_of: pd.Timestamp,
    days: int,
    champion_version: str,
    champion_promoted_at: pd.Timestamp | None,
) -> Cooldown:
    """The cooldown one event (a rejection or a rollback) puts on the current champion."""
    rollback = event["kind"] == db.EVENT_POST_PROMOTION
    # A rejection holds the champion it was decided against; a rollback the one it restored.
    held = _details(event).get("previous_version") if rollback else event["model_version"]
    rejected_at = event["as_of"]
    until = rejected_at.normalize() + pd.Timedelta(days=days)
    ended_by = None
    if str(held) != str(champion_version):
        ended_by = f"was against v{held}; the champion is now v{champion_version}"
    elif champion_promoted_at is not None and db.to_utc(champion_promoted_at) > event["created_at"]:
        ended_by = (
            f"predates the promotion of v{champion_version} on "
            f"{db.to_utc(champion_promoted_at):%Y-%m-%d}"
        )
    return Cooldown(
        active=ended_by is None and as_of < until,
        days=days,
        rejected_at=rejected_at,
        until=until,
        rejection=event["reason"],
        ended_by=ended_by,
        cause=f"v{event['model_version']} rolled back" if rollback else "challenger rejected",
        rollback=rollback,
    )


def cooldown_state(
    engine: Engine,
    as_of: pd.Timestamp,
    days: int,
    champion_version: str,
    champion_promoted_at: pd.Timestamp | None = None,
) -> Cooldown:
    """The cooldown at ``as_of`` for champion ``champion_version``.

    Active if the latest real rejection of a challenger against this champion version, or
    the latest rollback that restored it, is under ``days`` UTC calendar days old (``days=0``
    disables it) and the champion has not been promoted since (``champion_promoted_at``,
    when known, later than that event was recorded). Events evaluated after ``as_of`` are
    ignored.
    """
    as_of = db.to_utc(as_of)
    events = [
        db.latest_monitoring_event(engine, [match], as_of_at_most=as_of)
        for match in (
            (db.EVENT_RETRAIN, db.DECISION_KEPT),
            (db.EVENT_POST_PROMOTION, db.DECISION_ROLLED_BACK),
        )
    ]
    states = [
        _one_cooldown(e, as_of, days, champion_version, champion_promoted_at)
        for e in events
        if e is not None
    ]
    if not states:
        return Cooldown(active=False, days=days)
    active = [s for s in states if s.active]
    return max(active or states, key=lambda s: s.rejected_at)


# --- Post-promotion check: which promotion ---
@dataclass(frozen=True)
class Promotion:
    """An automatic promotion whose post-promotion check is still open."""

    new_version: str
    previous_version: str
    promoted_at: pd.Timestamp  # when the promoting pass was recorded


def open_promotion(
    engine: Engine,
    champion_version: str,
    as_of: pd.Timestamp,
    champion_promoted_at: pd.Timestamp | None = None,
) -> tuple[Promotion | None, str]:
    """``(promotion, "")`` if the champion's promotion still awaits its verdict, else
    ``(None, why not)``."""
    event = db.latest_monitoring_event(
        engine,
        [(db.EVENT_RETRAIN, db.DECISION_PROMOTED), (db.EVENT_POST_PROMOTION, None)],
        as_of_at_most=as_of,
    )
    if event is None:
        return None, "no automatic promotion on record"
    if event["kind"] == db.EVENT_POST_PROMOTION:
        return None, (
            f"the latest automatic promotion (v{event['model_version']}) already has its "
            f"verdict: {event['decision']}"
        )
    new = (_details(event).get("outcome") or {}).get("challenger_version")
    if new is None or str(new) != str(champion_version):
        return None, (
            f"champion v{champion_version} was not promoted by the retrain loop "
            f"(its latest promotion was v{new}): promotions made by hand are not checked"
        )
    if champion_promoted_at is not None and db.to_utc(champion_promoted_at) > event["created_at"]:
        return None, (
            f"champion v{champion_version} was promoted again by hand on "
            f"{db.to_utc(champion_promoted_at):%Y-%m-%d}: promotions made by hand are not checked"
        )
    # The real promotion time: the champion's promoted_at (set by registry.promote, a few
    # minutes before the promoting pass records its event); the event's time only if missing.
    when = event["created_at"] if champion_promoted_at is None else db.to_utc(champion_promoted_at)
    return Promotion(str(new), str(event["model_version"]), when), ""


def missing_rollback(
    engine: Engine,
    champion_version: str,
    as_of: pd.Timestamp,
    champion_promoted_at: pd.Timestamp | None,
) -> Promotion | None:
    """The automatic promotion that was rolled back without its verdict recorded, or None.

    That is: the latest automatic promotion has no ``post_promotion`` event since, the
    serving champion is the version it replaced, and that version was promoted after the
    promotion was recorded. (The previous version promoted back by hand before the verdict
    looks the same.)
    """
    event = db.latest_monitoring_event(
        engine,
        [(db.EVENT_RETRAIN, db.DECISION_PROMOTED), (db.EVENT_POST_PROMOTION, None)],
        as_of_at_most=as_of,
    )
    if event is None or event["kind"] == db.EVENT_POST_PROMOTION or champion_promoted_at is None:
        return None
    new = (_details(event).get("outcome") or {}).get("challenger_version")
    if new is None or str(new) == str(champion_version):
        return None
    if str(event["model_version"]) != str(champion_version):
        return None
    if db.to_utc(champion_promoted_at) <= event["created_at"]:
        return None
    return Promotion(str(new), str(champion_version), event["created_at"])


# --- Post-promotion check: the replay ---
class ReplayRefusedError(ValueError):
    """The two models cannot be compared fairly (horizon, features, training window)."""


def target_day(hours: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Berlin calendar day (naive midnight) of each UTC hour."""
    return hours.tz_convert(LOCAL_TZ).tz_localize(None).normalize()


def _first_hour_of(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The first UTC hour of each Berlin calendar day."""
    return days.tz_localize(LOCAL_TZ).tz_convert("UTC")


def _last_hour_of(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last UTC hour of each Berlin calendar day (23, 24 or 25 hours long)."""
    next_midnight = (days + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ).tz_convert("UTC")
    return next_midnight - pd.Timedelta(hours=1)


def replay_features(
    engine: Engine, scored: pd.DataFrame, horizon: Horizon
) -> tuple[pd.DataFrame, pd.Series]:
    """The feature rows each scored hour's issue could build, and why some cannot be rebuilt.

    ``scored`` has ``timestamp_utc`` and ``issued_at`` (one row per hour). Returns the feature
    rows that were built (an issue without stored weather builds none), indexed by hour, and,
    per scored hour, ``None`` or the reason it is not replayable (``NO_WEATHER``, ``NO_LAG``).
    Per issue: load stamped at or before the issue time, the weather rows stored with that
    issue, nothing else.

    The rows keep the dtypes the feature code gives them, as serving does: never reindexed
    onto hours that were not built, whose NaN rows would turn the integer calendar columns
    into float64, which an MLflow model's input schema refuses.
    """
    hours = pd.DatetimeIndex(scored["timestamp_utc"])
    reasons = pd.Series([None] * len(hours), index=hours, dtype=object)
    load = db.read_load(engine, start=hours.min() - REPLAY_HISTORY, end=scored["issued_at"].max())
    weather = db.read_table(engine, db.LoadForecastModelWeather, start=hours.min(), end=hours.max())
    frames = []
    for issued_at, rows in scored.groupby("issued_at"):
        target = pd.DatetimeIndex(rows["timestamp_utc"]).sort_values()
        inputs = weather[weather["issued_at"] == issued_at]
        if inputs.empty:
            reasons[target] = NO_WEATHER
            continue
        known = load[(load["timestamp_utc"] <= issued_at) & ~load["timestamp_utc"].isin(target)]
        unknown = pd.DataFrame({"timestamp_utc": target, TARGET: np.nan})
        frame = build_feature_frame(
            pd.concat([known, unknown], ignore_index=True), national_average(inputs), dropna=False
        )
        frames.append(frame.loc[target])
    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")), reasons
    features = pd.concat(frames)  # one code path per issue: the same dtypes in every frame
    lags = [
        c for c in select_features(features.columns, horizon) if most_recent_load_lag(c) is not None
    ]
    reasons[features.index[features[lags].isna().any(axis=1)]] = NO_LAG
    return features, reasons


@dataclass
class Comparison:
    """New vs previous champion, both replayed from the same rebuilt inputs, on the new
    champion's scored hours; parity with the new champion's live forecasts is reported."""

    new_version: str
    previous_version: str
    days_required: int
    n_days: int = 0  # complete target days (over, >= MIN_DAY_COVERAGE scored), at most required
    n_short_days: int = 0  # days that were over but had too few scored hours: skipped
    n_hours: int = 0  # the new champion's scored hours on those days
    n_compared: int = 0  # of which rebuilt and replayed for both models: the verdict's hours
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> hours not rebuildable
    n_parity: int = 0  # of n_compared: the new champion's replay within PARITY_TOLERANCE_MW
    max_parity_diff_mw: float | None = None  # largest |new replay - stored forecast|
    first_hour: pd.Timestamp | None = None
    last_hour: pd.Timestamp | None = None
    new_mae: float | None = None  # replayed
    new_live_mae: float | None = None  # its stored (live) forecasts, same hours: information
    previous_mae: float | None = None  # replayed
    official_mae: float | None = None
    improvement_pct: float | None = None  # replay vs replay; +x% = the new MAE is x% lower

    def short_days_text(self) -> str:
        return (
            f"{self.n_short_days} day(s) skipped with fewer than {MIN_DAY_COVERAGE * 100:.0f} % of "
            "their hours scored"
        )

    @property
    def unreplayable_share(self) -> float:
        return 1.0 - self.n_compared / self.n_hours if self.n_hours else 0.0

    def summary(self) -> str:
        lines = [
            f"  v{self.new_version} vs v{self.previous_version}, both replayed from each "
            f"issue's stored inputs on v{self.new_version}'s scored hours: {self.n_days} of "
            f"{self.days_required} complete target days, {self.n_hours} scored hours"
            + (f"; {self.short_days_text()}" if self.n_short_days else "")
        ]
        if self.n_hours:
            dropped = "; ".join(f"{n} h: {why}" for why, n in self.dropped.items()) or "none"
            lines.append(f"  hours left out (cannot be rebuilt): {dropped}")
        if self.new_mae is not None:
            lines.append(
                f"  {self.first_hour:%Y-%m-%d %H:%M}Z -> {self.last_hour:%Y-%m-%d %H:%M}Z, "
                f"{self.n_compared} hours: v{self.new_version} MAE {self.new_mae:,.0f} MW, "
                f"v{self.previous_version} MAE {self.previous_mae:,.0f} MW (both replayed), "
                f"official {self.official_mae:,.0f} MW -> new champion "
                f"{self.improvement_pct:+.1f} % (+ = lower MAE than the previous one)"
            )
            diff = "n/a" if self.max_parity_diff_mw is None else f"{self.max_parity_diff_mw:,.3f}"
            lines.append(
                f"  replay parity (information, not a gate): {self.n_parity} of "
                f"{self.n_compared} hours reproduce v{self.new_version}'s stored forecast within "
                f"{PARITY_TOLERANCE_MW:g} MW (largest difference {diff} MW); "
                f"v{self.new_version} MAE live {self.new_live_mae:,.0f} MW vs replayed "
                f"{self.new_mae:,.0f} MW"
            )
        return "\n".join(lines)


def compare_on_live_hours(
    engine: Engine,
    new: LoadedModel,
    previous: LoadedModel,
    *,
    as_of: pd.Timestamp,
    days: int,
    since: pd.Timestamp | None = None,
) -> Comparison:
    """Replay ``new`` and ``previous`` from the same rebuilt inputs on ``new``'s scored hours.

    Only issues made after ``since`` count (default: all), on the first ``days`` target days
    that are over (their last hour has an actual by ``as_of``). The MAEs compared are both
    replays; ``new``'s live forecasts give the parity figures and its live MAE, for
    information. Raises ``ReplayRefusedError`` when the comparison would not be fair.
    """
    if new.horizon != previous.horizon:
        raise ReplayRefusedError(
            f"v{new.version} forecasts {new.horizon.name}, v{previous.version} "
            f"{previous.horizon.name}: models of different horizons are never compared"
        )
    as_of = db.to_utc(as_of)
    out = Comparison(new.version, previous.version, days)
    start = since
    if start is None:
        start = db.earliest_timestamp(engine, db.LoadForecastModel, model_version=new.version)
    base = actual_vs_official(engine, start=start, end=as_of)
    if base.empty:
        return out
    scored = latest_on_time_issues(model_issues(engine, base), new.version)
    if since is not None:
        scored = scored[scored["issued_at"] > db.to_utc(since)]
    hours = base.merge(scored[["timestamp_utc", "forecast_mw", "issued_at"]], on="timestamp_utc")
    if hours.empty:
        return out

    # The first `days` complete target days: over (their last hour has an actual) and
    # scored for at least MIN_DAY_COVERAGE of their 23, 24 or 25 hours. Shorter days are skipped.
    day = target_day(pd.DatetimeIndex(hours["timestamp_utc"]))
    over = _last_hour_of(day) <= base["timestamp_utc"].max()
    scored_per_day = pd.Series(1, index=day[over]).groupby(level=0).size()
    days_over = scored_per_day.index
    length = (_last_hour_of(days_over) - _first_hour_of(days_over)) / pd.Timedelta(hours=1) + 1
    full = scored_per_day.to_numpy() >= MIN_DAY_COVERAGE * length.to_numpy()
    chosen = days_over[full].sort_values()[:days]
    short = days_over[~full]
    if len(chosen) == days:  # only the short days among those the verdict passed over
        short = short[short < chosen.max()]
    out.n_short_days = len(short)
    hours = hours[day.isin(chosen)].sort_values("timestamp_utc").reset_index(drop=True)
    out.n_days, out.n_hours = len(chosen), len(hours)
    if hours.empty:
        return out

    first = hours["timestamp_utc"].min()
    for model in (new, previous):
        if model.train_end is None or model.train_end >= first:
            raise ReplayRefusedError(
                f"v{model.version}'s training window (to {model.train_end}) is not known to "
                f"end before the first scored hour {first}"
            )

    # Only hours that cannot be rebuilt are left out. A replay that differs from the stored
    # forecast is kept: both models read the same (possibly revised) inputs.
    features, reasons = replay_features(engine, hours, new.horizon)
    out.dropped = {str(k): int(v) for k, v in reasons.dropna().value_counts().items()}
    keep = reasons.isna().to_numpy()
    out.n_compared = int(keep.sum())
    if not keep.any():
        return out
    try:
        new.check_features(list(features.columns))
        previous.check_features(list(features.columns))
    except ValueError as err:
        raise ReplayRefusedError(f"today's feature code no longer matches a model: {err}") from err

    kept = hours[keep]
    X = features.loc[pd.DatetimeIndex(kept["timestamp_utc"])]  # built rows, built dtypes
    actual = kept["actual_mw"].to_numpy()
    live = kept["forecast_mw"].to_numpy(dtype="float64")
    new_replay = np.asarray(new.predict(X), dtype="float64")
    previous_replay = np.asarray(previous.predict(X), dtype="float64")

    diff = np.abs(new_replay - live)
    out.n_parity = int((diff <= PARITY_TOLERANCE_MW).sum())  # NaN counts as different
    out.max_parity_diff_mw = float(np.nanmax(diff)) if not np.isnan(diff).all() else None
    out.first_hour, out.last_hour = kept["timestamp_utc"].min(), kept["timestamp_utc"].max()
    out.new_mae = compute_metrics(actual, new_replay)["mae"]
    out.previous_mae = compute_metrics(actual, previous_replay)["mae"]
    out.new_live_mae = compute_metrics(actual, live)["mae"]
    out.official_mae = compute_metrics(actual, kept["official_mw"])["mae"]
    out.improvement_pct = improvement_pct(out.previous_mae, out.new_mae)
    return out


# --- Post-promotion check: the verdict ---
def judge(
    comparison: Comparison, *, margin_pct: float, waited_days: float, max_wait_days: int
) -> tuple[str, str]:
    """``(verdict, reason)`` for a comparison: pending, inconclusive, roll back or confirmed."""
    c = comparison
    if c.n_days < c.days_required:
        progress = f"{c.n_days} of {c.days_required} complete target days scored"
        if c.n_short_days:
            progress += f" ({c.short_days_text()})"
        if waited_days > max_wait_days:
            return INCONCLUSIVE, f"{progress} after {waited_days:.0f} days (limit {max_wait_days})"
        return PENDING, progress
    if c.n_compared == 0 or c.unreplayable_share > MAX_UNREPLAYABLE_SHARE:
        return INCONCLUSIVE, (
            f"only {c.n_compared} of {c.n_hours} scored hours could be rebuilt and replayed "
            f"(more than {MAX_UNREPLAYABLE_SHARE:.0%} left out)"
        )
    numbers = (
        f"v{c.new_version} MAE {c.new_mae:,.0f} MW vs v{c.previous_version} "
        f"{c.previous_mae:,.0f} MW, both replayed, on the same {c.n_compared} hours "
        f"({c.improvement_pct:+.1f} %)"
    )
    if c.improvement_pct < -margin_pct:
        return ROLL_BACK, f"{numbers}: worse by more than the {margin_pct:g} % margin"
    return CONFIRMED, f"{numbers}: not worse by more than the {margin_pct:g} % margin"


@dataclass
class PostPromotionReport:
    verdict: str  # NOT_APPLICABLE | PENDING | CONFIRMED | ROLL_BACK | INCONCLUSIVE
    reason: str
    new_version: str
    margin_pct: float
    days: int
    previous_version: str | None = None
    promoted_at: pd.Timestamp | None = None
    comparison: Comparison | None = None
    acted: bool = False  # the verdict was carried out (for ROLL_BACK: the rollback happened)
    recorded: bool = False  # its post_promotion event was written
    recovered: bool = False  # a verdict a previous pass carried out but could not record

    @property
    def decision(self) -> str | None:
        """The decision of the event that closes the check; None while there is no verdict."""
        return _DECISIONS.get(self.verdict)

    @property
    def pending(self) -> bool:
        return self.verdict == PENDING

    @property
    def rolled_back(self) -> bool:
        """The champion was rolled back *in this pass* (recorded or not)."""
        return self.acted and self.verdict == ROLL_BACK and not self.recovered

    @property
    def label(self) -> str:
        """One word for tags and the dashboard; a verdict not carried out says so."""
        if self.verdict in (NOT_APPLICABLE, PENDING):
            return self.verdict
        if not self.acted:
            return f"{self.decision} (report only)"
        return self.decision if self.recorded else f"{self.decision} (not recorded)"

    def summary(self) -> str:
        head = f"Post-promotion check: {self.label.upper()} - {self.reason}"
        return head if self.comparison is None else f"{head}\n{self.comparison.summary()}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        return json.loads(json.dumps(d, default=str))


def post_promotion_check(
    engine: Engine,
    champion: LoadedModel,
    as_of: pd.Timestamp,
    *,
    days: int = DEFAULT_POST_CHECK_DAYS,
    margin_pct: float = DEFAULT_ROLLBACK_MARGIN_PCT,
    max_wait_days: int = DEFAULT_MAX_WAIT_DAYS,
    champion_promoted_at: pd.Timestamp | None = None,
    act: bool = False,
) -> PostPromotionReport:
    """Check the serving champion's first live days against the champion it replaced.

    With ``act`` a verdict is carried out and recorded: a rollback through
    ``registry.promote``, then one ``post_promotion`` event that closes the check. Without it
    (dry run, check-only) nothing is written and nothing moves.
    """
    as_of = db.to_utc(as_of)
    lost = missing_rollback(engine, champion.version, as_of, champion_promoted_at)
    if lost is not None:
        return recover_rollback(engine, champion, lost, champion_promoted_at, as_of, act=act)

    report = PostPromotionReport(NOT_APPLICABLE, "", champion.version, margin_pct, days)
    promotion, why_not = open_promotion(engine, champion.version, as_of, champion_promoted_at)
    if promotion is None:
        report.reason = why_not
        return report
    report.previous_version = promotion.previous_version
    report.promoted_at = promotion.promoted_at
    waited = (as_of - promotion.promoted_at) / pd.Timedelta(days=1)
    try:
        previous = registry.load_version(engine, promotion.previous_version, name=champion.name)
        report.comparison = compare_on_live_hours(
            engine, champion, previous, as_of=as_of, days=days, since=promotion.promoted_at
        )
    except (LookupError, ReplayRefusedError, registry.CorruptBundleError) as err:
        report.verdict, report.reason = INCONCLUSIVE, str(err)  # will not change: close now
    except Exception as err:
        if waited <= max_wait_days:
            raise  # perhaps transient: the job fails and the next pass tries again
        report.verdict, report.reason = (
            INCONCLUSIVE,
            (
                f"v{promotion.previous_version} could not be loaded or replayed for "
                f"{waited:.0f} days (limit {max_wait_days}): {type(err).__name__}: {err}"
            ),
        )
    else:
        report.verdict, report.reason = judge(
            report.comparison,
            margin_pct=margin_pct,
            waited_days=waited,
            max_wait_days=max_wait_days,
        )
    if act and report.decision is not None:
        carry_out(engine, champion.name, report, as_of)
    return report


class VerdictNotRecordedError(RuntimeError):
    """A verdict was carried out (a rollback included) but its event could not be written.

    ``report`` says what happened, so the pass can still refuse to train a challenger.
    """

    def __init__(self, report: PostPromotionReport, cause: Exception) -> None:
        super().__init__(f"post-promotion verdict not recorded: {cause}")
        self.report = report


def recover_rollback(
    engine: Engine,
    champion: LoadedModel,
    lost: Promotion,
    champion_promoted_at: pd.Timestamp,
    as_of: pd.Timestamp,
    *,
    act: bool,
) -> PostPromotionReport:
    """Record the ``rolled_back`` verdict a previous pass carried out but could not write.

    Dated at the rollback's promotion (never after this pass), so the cooldown runs from it.
    """
    when = min(db.to_utc(champion_promoted_at), as_of)
    report = PostPromotionReport(
        ROLL_BACK,
        (
            f"recovered: v{champion.version} has served again since {when:%Y-%m-%d %H:%M}Z, "
            f"after the automatic promotion of v{lost.new_version}, and no verdict was "
            "recorded - a rollback whose event could not be written (or "
            f"v{champion.version} promoted back by hand before the verdict)"
        ),
        lost.new_version,
        DEFAULT_ROLLBACK_MARGIN_PCT,
        DEFAULT_POST_CHECK_DAYS,
        previous_version=champion.version,
        promoted_at=lost.promoted_at,
        recovered=True,
    )
    if act:
        report.acted = True
        record_verdict(engine, report, when)
        report.recorded = True
        tag_version(champion.name, lost.new_version, db.DECISION_ROLLED_BACK)
    return report


def carry_out(engine: Engine, name: str, report: PostPromotionReport, as_of: pd.Timestamp) -> None:
    """Act on a verdict: roll back if it says so, then record the one event that closes the
    check, then tag the MLflow version. In that order: if the rollback fails nothing is
    recorded and the next pass tries again; a recorded verdict is never left without its
    rollback. If the event cannot be written after a rollback, ``VerdictNotRecordedError``
    carries the report (marked rolled back) so the pass trains nothing and fails; the next
    pass records the verdict (``recover_rollback``)."""
    if report.verdict == ROLL_BACK:
        registry.promote(report.previous_version, engine=engine, name=name)
        log.warning(
            "rolled back: v%s is champion again, v%s demoted (undo: registry --promote %s)",
            report.previous_version,
            report.new_version,
            report.new_version,
        )
    report.acted = True  # before the event: a rollback is real whether or not it is recorded
    try:
        record_verdict(engine, report, as_of)
    except Exception as err:
        raise VerdictNotRecordedError(report, err) from err
    report.recorded = True
    tag_version(name, report.new_version, report.decision)


def record_verdict(engine: Engine, report: PostPromotionReport, as_of: pd.Timestamp) -> int:
    """The one ``post_promotion`` event that closes a check (``model_version`` = the champion
    that was checked; ``details.previous_version`` = the one it replaced)."""
    return db.insert_monitoring_event(
        engine,
        as_of=db.to_utc(as_of).to_pydatetime(),
        kind=db.EVENT_POST_PROMOTION,
        triggered=False,
        decision=report.decision,
        reason=report.reason,
        model_version=report.new_version,
        details=json.dumps(report.to_dict()),
    )


def tag_version(name: str, version: str, decision: str) -> None:
    """Write the verdict on the MLflow model version (lineage only: a failure is logged, never
    raised - the rollback and its event are already done)."""
    try:
        import mlflow

        from src.models.tracking import setup_mlflow

        setup_mlflow()
        mlflow.MlflowClient().set_model_version_tag(name, str(version), "post_promotion", decision)
    except Exception:
        log.exception("could not tag %s v%s with post_promotion=%s", name, version, decision)


# --- Report-only CLI ---
def main(argv: list[str] | None = None) -> int:
    configure_logging()
    p = argparse.ArgumentParser(
        prog="python -m src.monitoring.promotion",
        description="Compare two exported model versions on the newer one's live hours. "
        "Reads the database; writes nothing.",
    )
    p.add_argument("--version", required=True, help="the newer version (its live forecasts)")
    p.add_argument("--previous", required=True, help="the version it replaced (replayed)")
    p.add_argument("--since", default=None, help="only issues made after this UTC time")
    p.add_argument("--days", type=int, default=DEFAULT_POST_CHECK_DAYS)
    p.add_argument("--margin", type=float, default=DEFAULT_ROLLBACK_MARGIN_PCT)
    p.add_argument("--as-of", default=None)
    p.add_argument("--database-url", default=None)
    args = p.parse_args(argv)

    engine = db.get_engine(args.database_url)
    as_of = db.to_utc(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC").floor("h")
    new = registry.load_version(engine, args.version)
    previous = registry.load_version(engine, args.previous)
    try:
        comparison = compare_on_live_hours(
            engine,
            new,
            previous,
            as_of=as_of,
            days=args.days,
            since=db.to_utc(args.since) if args.since else None,
        )
    except ReplayRefusedError as err:
        print(f"Comparison refused: {err}")
        return 1
    verdict, reason = judge(
        comparison, margin_pct=args.margin, waited_days=0, max_wait_days=DEFAULT_MAX_WAIT_DAYS
    )
    print(f"Report only (nothing written), as of {as_of:%Y-%m-%d %H:%M}Z")
    print(comparison.summary())
    print(f"  -> the automatic check would say: {verdict.upper()} - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
