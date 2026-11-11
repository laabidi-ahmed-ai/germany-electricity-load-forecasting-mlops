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

What starts it: only a real rejection. ``skipped`` compared nothing, and a dry run's
decision (``kept (dry-run)``) must not change what later passes do. ``--force`` bypasses the
cooldown; the bypass is logged and stored with the pass.

Only against the champion that serves now. A rejection says "this champion is not beaten
by this recipe on this data"; it says nothing about another champion. So it counts only if
it was decided against the current champion's version (the event's ``model_version``) and
that champion has not been promoted since (its ``promoted_at`` in ``model_artifacts``, which
also catches a version promoted again by hand). Any promotion - automatic, forced, or by hand
through ``bootstrap`` or ``registry --promote`` - therefore ends a running cooldown.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy.engine import Engine

from src.data import db

DEFAULT_COOLDOWN_DAYS = 7


@dataclass(frozen=True)
class Cooldown:
    """Whether a new challenger may be trained at ``as_of``, and which rejection says not."""

    active: bool
    days: int
    rejected_at: pd.Timestamp | None = None  # as_of of the rejecting pass
    until: pd.Timestamp | None = None  # first moment a new challenger may be trained
    rejection: str | None = None  # the reason of that rejection
    ended_by: str | None = None  # why a rejection on record does not hold this champion

    def detail(self) -> str:
        if self.rejected_at is None:
            return "no rejected challenger on record"
        if self.ended_by is not None:
            return f"latest rejection ({self.rejected_at:%Y-%m-%d}) {self.ended_by}"
        return (
            f"challenger rejected on {self.rejected_at:%Y-%m-%d %H:%M}Z ({self.rejection}); "
            f"no new challenger before {self.until:%Y-%m-%d} UTC ({self.days}-day cooldown)"
        )


def cooldown_state(
    engine: Engine,
    as_of: pd.Timestamp,
    days: int,
    champion_version: str,
    champion_promoted_at: pd.Timestamp | None = None,
) -> Cooldown:
    """The cooldown at ``as_of`` for champion ``champion_version``.

    Active if the latest real rejection is under ``days`` UTC calendar days old (``days=0``
    disables it), was decided against this champion version, and the champion has not been
    promoted since (``champion_promoted_at``, when known, later than the rejection was
    recorded). Events evaluated after ``as_of`` are ignored.
    """
    as_of = db.to_utc(as_of)
    event = db.latest_monitoring_event(
        engine, [(db.EVENT_RETRAIN, db.DECISION_KEPT)], as_of_at_most=as_of
    )
    if event is None:
        return Cooldown(active=False, days=days)
    rejected_at = event["as_of"]
    until = rejected_at.normalize() + pd.Timedelta(days=days)
    ended_by = None
    if str(event["model_version"]) != str(champion_version):
        ended_by = f"was against v{event['model_version']}; the champion is now v{champion_version}"
    elif champion_promoted_at is not None and db.to_utc(champion_promoted_at) > event["created_at"]:
        ended_by = f"predates the promotion of v{champion_version} on {db.to_utc(champion_promoted_at):%Y-%m-%d}"
    return Cooldown(
        active=ended_by is None and as_of < until,
        days=days,
        rejected_at=rejected_at,
        until=until,
        rejection=event["reason"],
        ended_by=ended_by,
    )
