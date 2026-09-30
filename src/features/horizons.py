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

Nowcast (a possible future model): information cutoff of 1 hour, the full feature
set is valid.

The frame keeps every column; selection happens here, by column-name contract:
``load_lag_{k}`` and ``load_roll_{stat}_{w}`` are the load-derived names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_LAG_RE = re.compile(r"^load_lag_(\d+)$")
_ROLL_RE = re.compile(r"^load_roll_[a-z]+_(\d+)$")

# Rolling windows in ``build_features`` end at t-1 (computed on ``load.shift(1)``).
_ROLLING_MOST_RECENT_LAG = 1


@dataclass(frozen=True)
class Horizon:
    """A forecast horizon = the minimum age (hours) of any load observation a feature may use."""

    name: str
    min_lag_hours: int

    def is_valid(self, column: str) -> bool:
        """True if ``column`` may be used by a model for this horizon."""
        lag = most_recent_load_lag(column)
        return lag is None or lag >= self.min_lag_hours


DAY_AHEAD = Horizon("day_ahead", min_lag_hours=24)
NOWCAST = Horizon("nowcast", min_lag_hours=1)

HORIZONS: dict[str, Horizon] = {h.name: h for h in (DAY_AHEAD, NOWCAST)}


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
