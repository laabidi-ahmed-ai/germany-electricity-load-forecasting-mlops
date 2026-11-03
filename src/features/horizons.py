"""Horizon-aware feature selection: which columns a model may see for a given lead time.

Day-ahead: the forecast for every hour of day D+1 is issued once day D is complete,
so for a target hour t on D+1 the load is known only up to t-24h (for 00:00) and
t-47h (for 23:00). We use the standard simplification of a fixed 24-hour
information cutoff: a load-derived feature is day-ahead valid only if the most
recent observation it uses is at least 24 hours before t.

For the frame built in ``build_features`` that means:

* ``load_lag_24 / 48 / 168``            -> valid
* ``load_lag_1``                        -> excluded (unobserved at issue time)
* ``load_roll_*_24`` and ``load_roll_*_168`` -> excluded: both windows end at t-1,
  so they contain the unobserved last 24 hours (the 168h window is long, but its
  most recent value is not)
* calendar features                     -> always valid (known for any future hour)
* weather features                      -> valid; at serving time they are weather
  forecasts for t, which is a train/serve skew (forecast error), not leakage

Calendar day D+1 (``d_plus_1``): the forecast issued on the morning of day D for every
hour of day D+1 in local (Berlin) time - the same target as the official day-ahead
forecast and the noon auction. Issued around 05:40 UTC with actuals up to ~04:00 UTC,
the last target hour is ~43 h after the latest actual (44 h on the 25-hour autumn day),
so the information cutoff is 48 h: ``load_lag_48 / 168`` stay, ``load_lag_24`` goes, and
the 4-5 h of slack absorb a late SMARD publication. The target hours come from the
calendar (``Horizon.calendar_day``), not from the latest actual.

Nowcast (a possible future model): information cutoff of 1 hour, the full feature
set is valid.

The frame keeps every column; selection happens here, by column-name contract:
``load_lag_{k}`` and ``load_roll_{stat}_{w}`` are the load-derived names.

Each horizon also sets the deadline by which a forecast must be issued to be scored
(``Horizon.issue_deadline``): for D+1, 12:00 Berlin on the day before the target day - the
day-ahead auction's gate closure - and for the rolling horizons, the target hour itself.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

import pandas as pd

LOCAL_TZ = "Europe/Berlin"  # calendar features, the D+1 target day and its issue deadline
# The day-ahead auction closes at 12:00 Berlin: a D+1 forecast issued later is too late to use.
AUCTION_CLOSE_HOUR = 12

_LAG_RE = re.compile(r"^load_lag_(\d+)$")
_ROLL_RE = re.compile(r"^load_roll_[a-z]+_(\d+)$")

# Rolling windows in ``build_features`` end at t-1 (computed on ``load.shift(1)``).
_ROLLING_MOST_RECENT_LAG = 1

Times = TypeVar("Times", pd.Timestamp, pd.DatetimeIndex, pd.Series)


@dataclass(frozen=True)
class Horizon:
    """A forecast horizon = the minimum age (hours) of any load observation a feature may use.

    ``calendar_day``: the target is the next local calendar day instead of the hours right
    after the latest actual.
    """

    name: str
    min_lag_hours: int
    calendar_day: bool = False

    def is_valid(self, column: str) -> bool:
        """True if ``column`` may be used by a model for this horizon."""
        lag = most_recent_load_lag(column)
        return lag is None or lag >= self.min_lag_hours

    def issue_deadline(self, target: Times) -> Times:
        """The latest issue time at which a forecast for the UTC hour(s) ``target`` is scored.

        Calendar-day horizon: 12:00 Berlin on the day before the target hour's local day -
        11:00 UTC when that day is in winter time, 10:00 UTC in summer time; every hour of
        a 23- or 25-hour clock-change day gets the same deadline. Rolling horizons: the
        target hour itself. Same kind out as in (Timestamp, DatetimeIndex or Series).
        """
        if not self.calendar_day:
            return target
        if isinstance(target, pd.Series):
            deadline = self.issue_deadline(pd.DatetimeIndex(target))
            return pd.Series(deadline, index=target.index, name=target.name)
        if isinstance(target, pd.Timestamp):
            return self.issue_deadline(pd.DatetimeIndex([target]))[0]
        local_day = target.tz_convert(LOCAL_TZ).tz_localize(None).normalize()  # wall-clock dates
        noon_before = local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=AUCTION_CLOSE_HOUR)
        return noon_before.tz_localize(LOCAL_TZ).tz_convert("UTC")  # noon is never ambiguous


DAY_AHEAD = Horizon("day_ahead", min_lag_hours=24)
D_PLUS_1 = Horizon("d_plus_1", min_lag_hours=48, calendar_day=True)
NOWCAST = Horizon("nowcast", min_lag_hours=1)

HORIZONS: dict[str, Horizon] = {h.name: h for h in (DAY_AHEAD, D_PLUS_1, NOWCAST)}


def most_recent_load_lag(column: str) -> int | None:
    """Age (hours) of the newest load value a feature uses; None for non-load features."""
    if m := _LAG_RE.match(column):
        return int(m.group(1))
    if _ROLL_RE.match(column):
        return _ROLLING_MOST_RECENT_LAG
    return None


def select_features(
    columns: Iterable[str], horizon: Horizon, *, target: str = "load_mw"
) -> list[str]:
    """Columns usable as model inputs for ``horizon`` (order preserved, target excluded)."""
    return [c for c in columns if c != target and horizon.is_valid(c)]


def excluded_features(
    columns: Iterable[str], horizon: Horizon, *, target: str = "load_mw"
) -> list[str]:
    """The complement of ``select_features`` - handy for logging what a model must not see."""
    return [c for c in columns if c != target and not horizon.is_valid(c)]
