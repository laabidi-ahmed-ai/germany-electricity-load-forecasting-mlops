"""Margin replay tests (offline, synthetic data)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import TARGET
from src.features.horizons import D_PLUS_1
from src.monitoring import margin_replay
from tests.conftest import FAST_LGBM, synthetic_frame


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame(days=90)


def test_replay_scores_both_models_on_the_week_after_each_cut(frame, monkeypatch) -> None:
    from src.models import train

    cutoffs: list[pd.Timestamp] = []
    real_fit = train.fit_model

    def spy(df, horizon, params=None):
        cutoffs.append(df.index.max())
        assert "load_lag_24" not in df[[c for c in df.columns if horizon.is_valid(c)]]
        return real_fit(df, horizon, params)

    monkeypatch.setattr(train, "fit_model", spy)
    out = margin_replay.replay(
        frame, D_PLUS_1, cuts=3, step_days=7, gap_days=14, score_days=7, params=FAST_LGBM
    )

    assert list(out.columns) == margin_replay.COLUMNS and len(out) == 3
    last = frame.index.max() - pd.Timedelta(days=7)
    assert list(out["cut"]) == [last - pd.Timedelta(days=14), last - pd.Timedelta(days=7), last]
    assert (out["n_hours"] == 24 * 7).all()
    # No model has seen the week it is scored on: every training cutoff is a cut or older.
    assert max(cutoffs) == last
    # B of one cut is A of the cut two steps later: 3 cuts need 5 fits, not 6.
    assert len(cutoffs) == len(set(cutoffs)) == 5
    worse = (out["mae_new"] - out["mae_previous"]) / out["mae_previous"] * 100
    pd.testing.assert_series_equal(out["worse_pct"], worse, check_names=False)


def test_scored_week_is_strictly_after_the_newer_models_training(frame, monkeypatch) -> None:
    """Corrupting the target after the last cut must not change either model."""
    from src.models import train

    seen: list[float] = []
    real_fit = train.fit_model

    def spy(df, horizon, params=None):
        seen.append(float(df[TARGET].sum()))
        return real_fit(df, horizon, params)

    monkeypatch.setattr(train, "fit_model", spy)
    kwargs = dict(cuts=1, step_days=7, gap_days=7, score_days=7, params=FAST_LGBM)
    margin_replay.replay(frame, D_PLUS_1, **kwargs)
    before = list(seen)
    seen.clear()
    tampered = frame.copy()
    tampered.loc[tampered.index > frame.index.max() - pd.Timedelta(days=7), TARGET] += 9_999
    margin_replay.replay(tampered, D_PLUS_1, **kwargs)
    assert seen == before


def test_summary_counts_the_cuts_over_each_margin() -> None:
    result = pd.DataFrame({"worse_pct": [-4.0, 0.0, 3.5, 5.0, 6.0, 9.0]})
    s = margin_replay.summarize(result)
    assert s["n_cuts"] == 6 and s["median"] == 4.25 and (s["min"], s["max"]) == (-4.0, 9.0)
    assert s["over"] == {3.0: 4, 5.0: 2, 8.0: 1}  # exactly 5 % is not over 5 %
    text = margin_replay.format_summary(result)
    assert "worse by more than 5 %: 2 of 6 (33 %)" in text


def test_replay_refuses_impossible_settings(frame) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        margin_replay.replay(frame, cuts=0)
    with pytest.raises(ValueError, match="fewer cuts"):
        margin_replay.replay(frame, cuts=30, step_days=14, params=FAST_LGBM)
