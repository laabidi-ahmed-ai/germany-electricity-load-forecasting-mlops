"""Post-promotion check tests (offline): the replay, the verdict, the rollback, no flapping.

The world: 40 days of load, official forecast and per-city weather; champion v6, promoted
automatically over v5 on 25 Jan 2024, issues a D+1 forecast every morning through the real
serving code, which stores each issue with the weather it read.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete, update

from src.data import db
from src.features.build_features import TARGET
from src.features.horizons import D_PLUS_1, DAY_AHEAD, select_features
from src.monitoring import metrics, promotion, retrain
from src.serving.forecast import build_serving_features, make_day_ahead_forecast, store_forecast
from tests.conftest import fake_loaded_model, load_curve
from tests.test_observability import enabled, value
from tests.test_registry import alias_version, env, store  # noqa: F401 - fixtures
from tests.test_serving import _weather_rows

START = pd.Timestamp("2024-01-01T00:00Z")
N_DAYS = 40
PROMOTED = pd.Timestamp("2024-01-25T12:00Z")  # the promoting pass (v5 -> v6)
TRAIN_END = pd.Timestamp("2024-01-20T00:00Z")
ISSUE_DAYS = pd.date_range("2024-01-26", "2024-02-06", freq="D", tz="UTC")  # targets 27 Jan on
AS_OF_DUE = pd.Timestamp("2024-02-04T06:00Z")  # target days 27 Jan - 3 Feb are over: 8 >= 7
AS_OF_EARLY = pd.Timestamp("2024-01-30T06:00Z")  # only 27 - 29 Jan are over
WEEK_HOURS = 7 * 24  # the first 7 target days, winter time


class Stub:
    """A model that depends on a load lag and on the weather: last week's load + 50 x T + bias."""

    def __init__(self, bias: float) -> None:
        self.bias = bias

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["load_lag_168"] + 50.0 * X["temperature_2m_de_avg"] + self.bias).to_numpy()


def make_world(
    engine,
    monkeypatch,
    *,
    new_bias: float = 1_500.0,  # v6 is clearly worse than v5 by default
    previous_bias: float = -500.0,
    late_days: tuple[str, ...] = (),
    previous_horizon=D_PLUS_1,
    as_pyfunc: Callable[[Stub, pd.DataFrame], Any] | None = None,
) -> SimpleNamespace:
    """``as_pyfunc(stub, features)`` turns each stub into the model object that predicts
    (default: the stub itself), given the served feature frame it will be called with."""
    wrap = as_pyfunc or (lambda stub, X: stub)
    ts = pd.date_range(START, periods=24 * N_DAYS, freq="h", tz="UTC")
    load = load_curve(len(ts))
    rng = np.random.default_rng(3)
    db.upsert_dataframe(
        engine, db.LoadActual, pd.DataFrame({"timestamp_utc": ts, TARGET: load, "source": "smard"})
    )
    db.upsert_dataframe(
        engine,
        db.LoadForecastOfficial,
        pd.DataFrame(
            {
                "timestamp_utc": ts,
                "forecast_mw": load + rng.normal(0, 800, len(ts)),
                "source": "smard",
            }
        ),
    )
    db.upsert_dataframe(engine, db.WeatherHourly, _weather_rows(ts, "archive"))

    served, _, _ = build_serving_features(engine, horizon=D_PLUS_1, as_of=ISSUE_DAYS[0])
    features = select_features(served.columns, D_PLUS_1)
    previous_features = select_features(served.columns, previous_horizon)
    new = fake_loaded_model(
        features,
        version="6",
        pyfunc=wrap(Stub(new_bias), served[features]),
        train_end=TRAIN_END,
        horizon=D_PLUS_1,
    )
    previous = fake_loaded_model(
        previous_features,
        version="5",
        pyfunc=wrap(Stub(previous_bias), served[previous_features]),
        train_end=TRAIN_END - pd.Timedelta(days=14),
        horizon=previous_horizon,
    )

    results = {}
    for day in ISSUE_DAYS:  # the daily job: actuals to 04:00, issued at 05:40 UTC
        result = make_day_ahead_forecast(engine, new, as_of=day + pd.Timedelta(hours=4))
        late = f"{day + pd.Timedelta(days=1):%Y-%m-%d}" in late_days  # after the 11:00 deadline
        result.issued_at = day + (
            pd.Timedelta(hours=12) if late else pd.Timedelta(hours=5, minutes=40)
        )
        store_forecast(engine, result)
        results[result.issued_at] = result

    db.insert_monitoring_event(
        engine,
        created_at=PROMOTED.to_pydatetime(),
        as_of=PROMOTED.to_pydatetime(),
        kind=db.EVENT_RETRAIN,
        triggered=True,
        decision=db.DECISION_PROMOTED,
        model_version="5",
        details=json.dumps({"outcome": {"decision": "promoted", "challenger_version": "6"}}),
    )

    models = {"5": previous, "6": new}
    state = {"champion": new, "promoted_at": PROMOTED - pd.Timedelta(minutes=5)}
    promoted: list[str] = []
    tags: list[tuple] = []

    def load_version(engine, version, **kwargs):
        if str(version) not in models:
            raise LookupError(f"v{version} is not exported in the database")
        return models[str(version)]

    def promote(version, engine=None, **kwargs):
        promoted.append(str(version))
        state["champion"] = models[str(version)]
        state["promoted_at"] = pd.Timestamp.now(tz="UTC")

    monkeypatch.setattr(promotion.registry, "load_version", load_version)
    monkeypatch.setattr(promotion.registry, "promote", promote)
    monkeypatch.setattr(promotion.registry, "load_champion", lambda engine: state["champion"])
    monkeypatch.setattr(
        promotion.registry, "champion_record", lambda engine: {"promoted_at": state["promoted_at"]}
    )
    monkeypatch.setattr(promotion, "tag_version", lambda *a: tags.append(a))
    return SimpleNamespace(
        engine=engine,
        new=new,
        previous=previous,
        models=models,
        state=state,
        promoted=promoted,
        tags=tags,
        results=results,
        bias_gap=new_bias - previous_bias,
    )


@pytest.fixture
def world(engine, monkeypatch) -> SimpleNamespace:
    return make_world(engine, monkeypatch)


def compare(w, as_of=AS_OF_DUE, days=7) -> promotion.Comparison:
    return promotion.compare_on_live_hours(
        w.engine, w.new, w.previous, as_of=as_of, days=days, since=PROMOTED
    )


def check(w, as_of=AS_OF_DUE, **kwargs) -> promotion.PostPromotionReport:
    return promotion.post_promotion_check(
        w.engine,
        w.state["champion"],
        as_of,
        champion_promoted_at=w.state["promoted_at"],
        **kwargs,
    )


def events(engine, kind=db.EVENT_POST_PROMOTION) -> pd.DataFrame:
    all_events = db.read_monitoring_events(engine)
    return all_events[all_events["kind"] == kind].reset_index(drop=True)


def scored_hours(w, as_of=AS_OF_DUE) -> pd.DataFrame:
    """v6's scored hours as the live record has them (``with_model_forecast``)."""
    base = metrics.actual_vs_official(w.engine, start=PROMOTED, end=as_of)
    return metrics.with_model_forecast(w.engine, base, model_version="6")


# --- The replay ---
def test_replay_reproduces_every_live_forecast_from_its_stored_inputs(world) -> None:
    c = compare(world)
    assert (c.n_days, c.n_hours, c.n_compared) == (7, WEEK_HOURS, WEEK_HOURS)
    assert c.dropped == {} and c.max_parity_diff_mw < 1e-6
    assert c.n_parity == WEEK_HOURS and c.new_live_mae == pytest.approx(c.new_mae)
    assert c.first_hour == pd.Timestamp("2024-01-26T23:00Z")  # 27 Jan 00:00 Berlin
    assert c.last_hour == pd.Timestamp("2024-02-02T22:00Z")

    # The replayed features are the served ones, issue by issue.
    issued_at, result = next(iter(world.results.items()))
    scored = pd.DataFrame({"timestamp_utc": result.frame.index, "issued_at": issued_at})
    features, reasons = promotion.replay_features(world.engine, scored, D_PLUS_1)
    assert reasons.isna().all()
    pd.testing.assert_frame_equal(
        features[world.new.features], result.frame[world.new.features], check_freq=False
    )


def test_both_models_are_scored_on_exactly_the_new_champions_scored_hours(
    engine, monkeypatch
) -> None:
    """A late day and hours without an official forecast are scored for neither model."""
    w = make_world(engine, monkeypatch, late_days=("2024-01-29",))
    gone = pd.date_range("2024-01-30T10:00Z", periods=2, freq="h")  # 22 of 24: still complete
    with engine.begin() as conn:
        conn.execute(
            delete(db.LoadForecastOfficial).where(
                db.LoadForecastOfficial.timestamp_utc.in_(list(gone.to_pydatetime()))
            )
        )
    c = compare(w)

    live = scored_hours(w)
    day = promotion.target_day(pd.DatetimeIndex(live["timestamp_utc"]))
    assert "2024-01-29" not in set(day.strftime("%Y-%m-%d"))  # its only issue was late
    # the first 7 scored days that are over: 27, 28, 30, 31 Jan, 1 - 3 Feb
    chosen = [f"2024-01-{d}" for d in (27, 28, 30, 31)] + [f"2024-02-0{d}" for d in (1, 2, 3)]
    expected = live[day.strftime("%Y-%m-%d").isin(chosen)]
    assert c.n_days == 7 and c.n_hours == c.n_compared == len(expected) == WEEK_HOURS - 2
    assert c.n_short_days == 0
    assert not expected["timestamp_utc"].isin(gone).any()

    actual = expected["actual_mw"]
    assert c.new_mae == pytest.approx(metrics.compute_metrics(actual, expected["model_mw"])["mae"])
    # The stubs differ by a constant, so v5's replay is v6's stored forecast minus the gap.
    assert c.previous_mae == pytest.approx(
        metrics.compute_metrics(actual, expected["model_mw"] - w.bias_gap)["mae"]
    )
    assert c.official_mae == pytest.approx(
        metrics.compute_metrics(actual, expected["official_mw"])["mae"]
    )
    assert c.improvement_pct == pytest.approx((c.previous_mae - c.new_mae) / c.previous_mae * 100)


def test_only_issues_made_after_the_promotion_count(world) -> None:
    after_first = min(world.results) + pd.Timedelta(minutes=1)
    c = promotion.compare_on_live_hours(
        world.engine, world.new, world.previous, as_of=AS_OF_DUE, days=7, since=after_first
    )
    assert c.first_hour == pd.Timestamp("2024-01-27T23:00Z")  # the first issue is left out


def remove_official(engine, start: str, hours: int) -> None:
    gone = pd.date_range(pd.Timestamp(start), periods=hours, freq="h")
    with engine.begin() as conn:
        conn.execute(
            delete(db.LoadForecastOfficial).where(
                db.LoadForecastOfficial.timestamp_utc.in_(list(gone.to_pydatetime()))
            )
        )


def test_a_day_with_too_few_scored_hours_is_skipped_not_counted(world) -> None:
    """30 Jan has 18 of 24 hours scored (75 % < 90 %): the next full day is used instead."""
    remove_official(world.engine, "2024-01-30T05:00Z", 6)
    c = compare(world)
    assert c.n_days == 7 and c.n_short_days == 1 and c.n_hours == WEEK_HOURS
    assert c.last_hour == pd.Timestamp("2024-02-03T22:00Z")  # 3 Feb replaces 30 Jan
    assert "1 day(s) skipped with fewer than 90 % of their hours scored" in c.summary()

    # With no eighth full day yet the check waits, and the wait limit still closes it.
    verdict, reason = promotion.judge(
        compare(world, days=8), margin_pct=5.0, waited_days=9, max_wait_days=21
    )
    assert verdict == promotion.PENDING
    assert reason == (
        "7 of 8 complete target days scored (1 day(s) skipped with fewer than 90 % of their "
        "hours scored)"
    )
    late = check(world, days=8, max_wait_days=3)
    assert late.verdict == promotion.INCONCLUSIVE and "(limit 3)" in late.reason


def test_a_day_at_exactly_ninety_percent_counts() -> None:
    """The coverage rule on a 23-, 24- and 25-hour day: at least 90 % of its own hours."""
    days = pd.DatetimeIndex(["2024-03-31", "2024-04-01", "2024-10-27"])  # 23 h, 24 h, 25 h
    length = (promotion._last_hour_of(days) - promotion._first_hour_of(days)) / pd.Timedelta(
        hours=1
    ) + 1
    assert list(length) == [23, 24, 25]
    assert [int(np.ceil(promotion.MIN_DAY_COVERAGE * n)) for n in length] == [21, 22, 23]


def test_the_wait_and_the_since_filter_start_at_the_real_promotion_time(world) -> None:
    """registry.promote sets promoted_at minutes before the pass records its event; an issue
    made in between belongs to the new champion and counts."""
    first_issue = min(world.results)
    with world.engine.begin() as conn:  # the promoting event recorded after the first issue
        conn.execute(
            update(db.MonitoringEvent)
            .where(db.MonitoringEvent.decision == db.DECISION_PROMOTED)
            .values(created_at=(first_issue + pd.Timedelta(minutes=10)).to_pydatetime())
        )
    report = check(world)  # champion_promoted_at = PROMOTED - 5 min
    assert report.promoted_at == PROMOTED - pd.Timedelta(minutes=5)
    assert report.comparison.first_hour == pd.Timestamp("2024-01-26T23:00Z")  # first issue in

    # Without promoted_at the event's time is the fallback: the first issue is then left out.
    fallback = promotion.post_promotion_check(world.engine, world.new, AS_OF_DUE)
    assert fallback.promoted_at == first_issue + pd.Timedelta(minutes=10)
    assert fallback.comparison.first_hour == pd.Timestamp("2024-01-27T23:00Z")


# --- No leakage ---
def test_the_replay_reads_nothing_from_after_the_issue(world) -> None:
    """Load after an issue and today's weather table cannot change that issue's replay."""
    issues = sorted(world.results)
    middle, later = issues[3], issues[6]

    def features_of(issued_at) -> pd.DataFrame:
        scored = pd.DataFrame(
            {"timestamp_utc": world.results[issued_at].frame.index, "issued_at": issued_at}
        )
        return promotion.replay_features(world.engine, scored, D_PLUS_1)[0]

    before_middle, before_later = features_of(middle), features_of(later)

    with world.engine.begin() as conn:  # every actual after the middle issue, targets included
        conn.execute(
            update(db.LoadActual)
            .where(db.LoadActual.timestamp_utc > middle.to_pydatetime())
            .values(load_mw=db.LoadActual.load_mw + 9_999)
        )
        conn.execute(update(db.WeatherHourly).values(temperature_2m=-30.0))  # "the archive"

    pd.testing.assert_frame_equal(features_of(middle), before_middle)  # bit-identical
    # The check has teeth: an issue made after the changed hours does see them.
    assert not features_of(later)["load_lag_48"].equals(before_later["load_lag_48"])
    # ... and still never the weather table: its weather is the stored one.
    pd.testing.assert_series_equal(
        features_of(later)["temperature_2m_de_avg"], before_later["temperature_2m_de_avg"]
    )


def test_a_lag_that_would_need_load_from_after_the_issue_is_dropped_not_filled(world) -> None:
    """Pretend an issue was made a day earlier: lag 48 h of its last hours is not known yet.

    A later issue is replayed in the same call, so the load it may read is in memory: only
    the cut at each issue's own time keeps it out of the earlier one.
    """
    issues = sorted(world.results)
    issued_at, result = issues[0], world.results[issues[0]]
    pretended = issued_at - pd.Timedelta(days=1)
    with world.engine.begin() as conn:  # the stored weather follows the pretended issue time
        conn.execute(
            update(db.LoadForecastModelWeather)
            .where(db.LoadForecastModelWeather.issued_at == issued_at.to_pydatetime())
            .values(issued_at=pretended.to_pydatetime())
        )
    later = world.results[issues[5]]
    scored = pd.concat(
        [
            pd.DataFrame({"timestamp_utc": result.frame.index, "issued_at": pretended}),
            pd.DataFrame({"timestamp_utc": later.frame.index, "issued_at": issues[5]}),
        ],
        ignore_index=True,
    )
    features, reasons = promotion.replay_features(world.engine, scored, D_PLUS_1)
    early = features.loc[result.frame.index]
    unknown = early["load_lag_48"].isna().to_numpy()
    assert unknown.any() and not unknown.all()
    assert (reasons[result.frame.index][unknown] == promotion.NO_LAG).all()
    assert reasons[result.frame.index][~unknown].isna().all()
    assert reasons[later.frame.index].isna().all()  # the later issue itself replays in full


@pytest.mark.parametrize("which", ["new", "previous"])
def test_a_model_trained_into_the_scored_hours_is_refused(world, which) -> None:
    getattr(world, which).train_end = pd.Timestamp("2024-01-28T00:00Z")
    with pytest.raises(promotion.ReplayRefusedError, match="training window"):
        compare(world)
    report = check(world, act=True)
    assert report.verdict == promotion.INCONCLUSIVE and world.promoted == []
    assert events(world.engine).iloc[0]["decision"] == db.DECISION_INCONCLUSIVE


def test_a_model_without_a_recorded_training_window_is_refused(world) -> None:
    world.previous.train_end = None
    with pytest.raises(promotion.ReplayRefusedError, match="not known to end before"):
        compare(world)


# --- Parity: inputs that are no longer what the issue read ---
def revise_lags(engine, day: str, hours: int) -> pd.DatetimeIndex:
    """SMARD revises load one week before the first ``hours`` hours of Berlin target ``day``,
    after their issue (the table keeps only the latest value). Returns the affected target
    hours (their ``load_lag_168`` changed)."""
    first = pd.Timestamp(day).tz_localize("Europe/Berlin").tz_convert("UTC")
    stamps = pd.date_range(first - pd.Timedelta(days=7), periods=hours, freq="h")
    with engine.begin() as conn:
        conn.execute(
            update(db.LoadActual)
            .where(db.LoadActual.timestamp_utc.in_(list(stamps.to_pydatetime())))
            .values(load_mw=db.LoadActual.load_mw + 2_000)
        )
    return stamps + pd.Timedelta(days=7)


def test_load_revised_after_the_issue_keeps_the_hours_and_compares_both_replays(world) -> None:
    """20 of 168 hours (12 %, above the old 10 % parity limit) read a revised lag: no hour is
    left out, the verdict compares the two replays, and parity shows the revisions."""
    revised = revise_lags(world.engine, "2024-01-31", 20)
    c = compare(world)
    assert c.dropped == {} and c.n_hours == c.n_compared == WEEK_HOURS
    assert c.n_parity == WEEK_HOURS - 20 and c.max_parity_diff_mw == pytest.approx(2_000)

    # Expected numbers from the live record: both stubs read the revised lag (+2,000 MW).
    live = scored_hours(world)
    live = live[live["timestamp_utc"].between(c.first_hour, c.last_hour)]
    assert len(live) == WEEK_HOURS
    actual = live["actual_mw"]
    shift = np.where(live["timestamp_utc"].isin(revised), 2_000.0, 0.0)
    new_replay = live["model_mw"] + shift
    mae = lambda pred: metrics.compute_metrics(actual, pred)["mae"]  # noqa: E731
    assert c.new_mae == pytest.approx(mae(new_replay))
    assert c.previous_mae == pytest.approx(mae(new_replay - world.bias_gap))
    assert c.new_live_mae == pytest.approx(mae(live["model_mw"]))
    assert c.new_live_mae != pytest.approx(c.new_mae)  # the verdict is not the live MAE
    assert c.improvement_pct == pytest.approx((c.previous_mae - c.new_mae) / c.previous_mae * 100)

    report = check(world, act=True)
    assert report.verdict == promotion.ROLL_BACK and world.promoted == ["5"]
    assert f"both replayed, on the same {WEEK_HOURS} hours" in report.reason
    summary = report.summary()
    assert "hours left out (cannot be rebuilt): none" in summary
    assert (
        f"replay parity (information, not a gate): {WEEK_HOURS - 20} of {WEEK_HOURS} hours "
        "reproduce v6's stored forecast within 1 MW (largest difference 2,000.000 MW)"
    ) in summary
    assert f"v6 MAE live {c.new_live_mae:,.0f} MW vs replayed {c.new_mae:,.0f} MW" in summary
    details = json.loads(events(world.engine).iloc[0]["details"])["comparison"]
    assert details["n_parity"] == WEEK_HOURS - 20 and details["n_compared"] == WEEK_HOURS


def test_a_revision_that_flips_the_live_ranking_does_not_decide_the_verdict(
    engine, monkeypatch
) -> None:
    """v6 is 300 MW better than v5 on the same inputs. Revisions make v6's stored forecasts
    look far worse than v5's replay; compared live, that would roll v6 back."""
    w = make_world(engine, monkeypatch, new_bias=0.0, previous_bias=300.0)
    for day in ("2024-01-28", "2024-01-30", "2024-02-01"):
        revise_lags(engine, day, 24)  # a whole day each: the replay moves, the stored does not
    with engine.begin() as conn:  # and the actuals of those days follow the revised level
        for day in ("2024-01-28", "2024-01-30", "2024-02-01"):
            first = pd.Timestamp(day).tz_localize("Europe/Berlin").tz_convert("UTC")
            stamps = pd.date_range(first, periods=24, freq="h")
            conn.execute(
                update(db.LoadActual)
                .where(db.LoadActual.timestamp_utc.in_(list(stamps.to_pydatetime())))
                .values(load_mw=db.LoadActual.load_mw + 2_000)
            )
    c = compare(w)
    assert c.n_compared == WEEK_HOURS and c.n_parity == WEEK_HOURS - 3 * 24
    assert c.new_live_mae > 1.05 * c.previous_mae  # live vs replay: "worse by more than 5 %"
    assert c.new_mae < c.previous_mae  # replay vs replay: v6 is better
    assert check(w, act=True).verdict == promotion.CONFIRMED and w.promoted == []


def drop_stored_weather(w) -> None:
    with w.engine.begin() as conn:
        conn.execute(
            delete(db.LoadForecastModelWeather).where(
                db.LoadForecastModelWeather.issued_at == sorted(w.results)[2].to_pydatetime()
            )
        )


def remove_lag_hours(w) -> None:
    first = pd.Timestamp("2024-01-24T23:00Z")  # one week before 1 Feb 00:00 Berlin
    stamps = pd.date_range(first, periods=20, freq="h")
    with w.engine.begin() as conn:
        conn.execute(
            delete(db.LoadActual).where(
                db.LoadActual.timestamp_utc.in_(list(stamps.to_pydatetime()))
            )
        )


@pytest.mark.parametrize(
    ("damage", "reason", "n_dropped"),
    [
        (drop_stored_weather, promotion.NO_WEATHER, 24),
        (remove_lag_hours, promotion.NO_LAG, 40),  # lag 168 h of 1 Feb and lag 48 h of 27 Jan
    ],
)
def test_too_many_unreplayable_hours_give_no_verdict_and_no_rollback(
    world, damage, reason, n_dropped
) -> None:
    damage(world)
    report = check(world, act=True)
    assert report.comparison.dropped == {reason: n_dropped}
    assert report.verdict == promotion.INCONCLUSIVE and "could be rebuilt" in report.reason
    assert world.promoted == []  # v6 is clearly worse here, and still not rolled back on doubt
    assert events(world.engine).iloc[0]["decision"] == db.DECISION_INCONCLUSIVE
    assert "hours left out" in report.summary() and reason in report.summary()


class StubPythonModel(mlflow.pyfunc.PythonModel):
    """The stub as a real MLflow pyfunc: MLflow enforces its input schema on every predict."""

    def __init__(self, bias: float) -> None:
        self.bias = bias

    def predict(self, context, model_input, params=None):
        return Stub(self.bias).predict(model_input)


def test_real_mlflow_models_replay_when_an_issue_has_no_stored_weather(
    engine, monkeypatch, tmp_path
) -> None:
    """A production model's signature is inferred from its input example, so the calendar
    columns are integers and MLflow refuses float64 for them. An issue without stored weather
    once turned every replayed row into float64 (NaN rows reindexed in) and the job crashed."""

    def as_pyfunc(stub: Stub, X: pd.DataFrame):
        path = tmp_path / f"model-{len(list(tmp_path.iterdir()))}"
        mlflow.pyfunc.save_model(
            path, python_model=StubPythonModel(stub.bias), input_example=X.head(5)
        )
        return mlflow.pyfunc.load_model(str(path))

    w = make_world(engine, monkeypatch, as_pyfunc=as_pyfunc)
    schema = {c.name: c.type.name for c in w.new.pyfunc.metadata.get_input_schema().inputs}
    assert schema["hour"] in ("integer", "long") and schema["is_weekend"] in ("integer", "long")

    # The test has teeth: the old frame (a NaN row reindexed in, then left out) is refused.
    issues = sorted(w.results)
    served = w.results[issues[0]].frame[w.new.features]
    extra = pd.DatetimeIndex([served.index.max() + pd.Timedelta(hours=1)])
    as_before = served.reindex(served.index.append(extra)).iloc[:-1]
    assert as_before["hour"].dtype == "float64"
    with pytest.raises(mlflow.exceptions.MlflowException, match="Incompatible input types"):
        w.new.predict(as_before)

    # The replayed rows keep the served dtypes, next to an issue that builds none.
    drop_stored_weather(w)  # the third issue
    scored = pd.concat(
        pd.DataFrame({"timestamp_utc": w.results[i].frame.index, "issued_at": i})
        for i in issues[:3]
    )
    features, reasons = promotion.replay_features(engine, scored, D_PLUS_1)
    assert (reasons == promotion.NO_WEATHER).sum() == 24 and len(features) == 48
    pd.testing.assert_series_equal(features[w.new.features].dtypes, served.dtypes)

    # Both replays run through MLflow; 24 of 168 hours left out is > 10 %: no verdict.
    c = compare(w)
    assert c.dropped == {promotion.NO_WEATHER: 24} and c.n_compared == WEEK_HOURS - 24
    assert c.new_mae is not None and c.previous_mae is not None
    assert c.n_parity == c.n_compared  # unrevised inputs: every stored forecast reproduced
    report = check(w, act=True)
    assert report.verdict == promotion.INCONCLUSIVE and w.promoted == []
    assert events(engine).iloc[0]["decision"] == db.DECISION_INCONCLUSIVE


# --- The verdict ---
def comparison(improvement: float, **kwargs) -> promotion.Comparison:
    defaults = dict(
        days_required=7,
        n_days=7,
        n_hours=168,
        n_compared=168,
        new_mae=1_000.0 * (1 - improvement / 100),
        previous_mae=1_000.0,
        official_mae=2_000.0,
        improvement_pct=improvement,
    )
    return promotion.Comparison("6", "5", **{**defaults, **kwargs})


def judge(c: promotion.Comparison, waited_days: float = 9) -> str:
    return promotion.judge(c, margin_pct=5.0, waited_days=waited_days, max_wait_days=21)[0]


def test_margin_edges_exactly_five_percent_worse_is_confirmed() -> None:
    assert judge(comparison(+3.0)) == promotion.CONFIRMED
    assert judge(comparison(-4.99)) == promotion.CONFIRMED
    assert judge(comparison(-5.0)) == promotion.CONFIRMED  # "more than" the margin
    assert judge(comparison(-5.01)) == promotion.ROLL_BACK
    verdict, reason = promotion.judge(
        comparison(-8.0), margin_pct=5.0, waited_days=9, max_wait_days=21
    )
    assert verdict == promotion.ROLL_BACK
    assert reason == (
        "v6 MAE 1,080 MW vs v5 1,000 MW, both replayed, on the same 168 hours (-8.0 %): "
        "worse by more than the 5 % margin"
    )


def test_fewer_than_k_days_is_pending_until_the_wait_limit() -> None:
    few = comparison(-50.0, n_days=6)
    assert judge(few) == promotion.PENDING  # however bad it looks: not yet
    assert judge(few, waited_days=21) == promotion.PENDING
    assert judge(few, waited_days=21.5) == promotion.INCONCLUSIVE


def test_unreplayable_share_edge() -> None:
    assert judge(comparison(-50.0, n_hours=170, n_compared=153)) == promotion.ROLL_BACK  # 10 %
    assert judge(comparison(-50.0, n_hours=170, n_compared=152)) == promotion.INCONCLUSIVE
    assert judge(comparison(-50.0, n_compared=0)) == promotion.INCONCLUSIVE


def test_a_worse_champion_is_rolled_back_once_and_the_event_says_so(world) -> None:
    report = check(world, act=True)
    assert report.verdict == promotion.ROLL_BACK and report.rolled_back
    assert report.label == db.DECISION_ROLLED_BACK
    assert report.comparison.improvement_pct < -5
    assert world.promoted == ["5"]  # through registry.promote
    assert world.tags == [("germany-load-day-ahead", "6", db.DECISION_ROLLED_BACK)]

    (event,) = events(world.engine).to_dict("records")
    assert event["decision"] == db.DECISION_ROLLED_BACK and event["model_version"] == "6"
    assert event["as_of"] == AS_OF_DUE and "worse by more than the 5 % margin" in event["reason"]
    details = json.loads(event["details"])
    assert details["previous_version"] == "5" and details["acted"] is True
    assert details["comparison"]["n_compared"] == WEEK_HOURS
    assert "ROLLED_BACK" in report.summary()
    assert "replay parity (information, not a gate): 168 of 168" in report.summary()


def test_a_champion_that_holds_up_is_confirmed_and_never_checked_again(engine, monkeypatch) -> None:
    w = make_world(engine, monkeypatch, new_bias=-500.0, previous_bias=1_500.0)
    report = check(w, act=True)
    assert report.verdict == promotion.CONFIRMED and not report.rolled_back
    assert report.comparison.improvement_pct > 0 and w.promoted == []
    assert events(engine).iloc[0]["decision"] == db.DECISION_CONFIRMED
    assert w.tags == [("germany-load-day-ahead", "6", db.DECISION_CONFIRMED)]

    again = check(w, AS_OF_DUE + pd.Timedelta(days=1), act=True)
    assert again.verdict == promotion.NOT_APPLICABLE
    assert "already has its verdict: confirmed" in again.reason
    assert len(events(engine)) == 1


def test_before_k_days_the_check_is_pending_and_writes_nothing(world) -> None:
    report = check(world, AS_OF_EARLY, act=True)
    assert report.pending and report.reason == "3 of 7 complete target days scored"
    assert report.decision is None and not report.acted
    assert world.promoted == [] and events(world.engine).empty


def test_a_check_still_open_after_the_wait_limit_is_closed(world) -> None:
    report = check(world, AS_OF_EARLY, act=True, max_wait_days=3)
    assert report.verdict == promotion.INCONCLUSIVE and "after 5 days (limit 3)" in report.reason
    assert world.promoted == []


# --- What is never rolled back ---
def test_a_horizon_change_is_never_rolled_back(engine, monkeypatch) -> None:
    """v5 day_ahead -> v6 d_plus_1, v6 far worse: a deliberate change, never compared."""
    w = make_world(engine, monkeypatch, new_bias=20_000.0, previous_horizon=DAY_AHEAD)
    report = check(w, act=True)
    assert report.verdict == promotion.INCONCLUSIVE and w.promoted == []
    assert "different horizons are never compared" in report.reason
    assert report.comparison is None
    assert check(w, AS_OF_DUE + pd.Timedelta(days=1), act=True).verdict == promotion.NOT_APPLICABLE
    assert w.promoted == [] and len(events(engine)) == 1


def test_a_pruned_previous_champion_gives_no_verdict(world) -> None:
    del world.models["5"]
    report = check(world, act=True)
    assert report.verdict == promotion.INCONCLUSIVE and "not exported" in report.reason
    assert world.promoted == []


def test_a_corrupt_previous_export_closes_the_check_at_once(world, monkeypatch) -> None:
    def corrupt(engine, version, **kwargs):
        raise promotion.registry.CorruptBundleError(f"corrupt model bundle for x v{version}")

    monkeypatch.setattr(promotion.registry, "load_version", corrupt)
    report = check(world, AS_OF_EARLY, act=True)  # long before the wait limit
    assert report.verdict == promotion.INCONCLUSIVE and "corrupt model bundle" in report.reason
    assert world.promoted == [] and events(world.engine).iloc[0]["decision"] == "inconclusive"


def test_any_other_replay_failure_fails_until_the_wait_limit_then_closes(
    world, monkeypatch
) -> None:
    def broken(engine, version, **kwargs):
        raise OSError("cannot load the old model's environment")

    monkeypatch.setattr(promotion.registry, "load_version", broken)
    with pytest.raises(OSError, match="environment"):  # the job fails; next pass retries
        check(world, act=True)
    assert events(world.engine).empty and world.promoted == []

    # About 10 days after the promotion, with a 7-day limit: closed, with the error.
    report = check(world, act=True, max_wait_days=7)
    assert report.verdict == promotion.INCONCLUSIVE and world.promoted == []
    assert report.reason == (
        "v5 could not be loaded or replayed for 10 days (limit 7): OSError: cannot load the old "
        "model's environment"
    )
    assert events(world.engine).iloc[0]["decision"] == db.DECISION_INCONCLUSIVE


def test_a_champion_promoted_by_hand_is_never_checked(world) -> None:
    # No automatic promotion of the serving champion: v7 came from bootstrap.
    v7 = fake_loaded_model(world.new.features, version="7", horizon=D_PLUS_1)
    report = promotion.post_promotion_check(world.engine, v7, AS_OF_DUE, act=True)
    assert report.verdict == promotion.NOT_APPLICABLE and world.promoted == []
    assert "promotions made by hand are not checked" in report.reason

    # v6 promoted again by hand after its automatic promotion: a human decision now.
    world.state["promoted_at"] = PROMOTED + pd.Timedelta(days=2)
    report = check(world, act=True)
    assert report.verdict == promotion.NOT_APPLICABLE and "again by hand" in report.reason
    assert world.promoted == [] and events(world.engine).empty


def test_promoted_at_equal_to_the_promoting_event_is_the_automatic_promotion(world) -> None:
    """The rehearsal sets the champion's promoted_at to its fabricated event's time: that is
    still the automatic promotion, and the check starts there. A moment later is by hand."""
    world.state["promoted_at"] = PROMOTED  # the promoting event's created_at, exactly
    report = check(world)
    assert report.verdict == promotion.ROLL_BACK and report.promoted_at == PROMOTED

    world.state["promoted_at"] = PROMOTED + pd.Timedelta(microseconds=1)
    report = check(world)
    assert report.verdict == promotion.NOT_APPLICABLE and "again by hand" in report.reason


def test_no_promotion_on_record(engine) -> None:
    v1 = fake_loaded_model(["load_lag_168"], version="1")
    report = promotion.post_promotion_check(engine, v1, AS_OF_DUE, act=True)
    assert report.verdict == promotion.NOT_APPLICABLE
    assert report.reason == "no automatic promotion on record" and report.label == "not_applicable"


# --- Report only ---
def test_without_act_nothing_is_written_and_nothing_moves(world) -> None:
    before = len(db.read_monitoring_events(world.engine))
    report = check(world)  # act=False: dry run, check-only
    assert report.verdict == promotion.ROLL_BACK and not report.acted and not report.rolled_back
    assert report.label == "rolled_back (report only)"
    assert world.promoted == [] and world.tags == []
    assert len(db.read_monitoring_events(world.engine)) == before
    assert check(world, act=True).rolled_back  # the real pass behaves as if it never ran


def test_report_only_cli_prints_parity_and_writes_nothing(world, capsys) -> None:
    before = len(db.read_monitoring_events(world.engine))
    args = ["--version", "6", "--previous", "5", "--database-url", str(world.engine.url)]
    assert promotion.main([*args, "--as-of", str(AS_OF_DUE), "--since", str(PROMOTED)]) == 0
    out = capsys.readouterr().out
    assert "Report only (nothing written)" in out
    assert (
        "replay parity (information, not a gate): 168 of 168 hours reproduce v6's stored "
        "forecast within 1 MW"
    ) in out
    assert "hours left out (cannot be rebuilt): none" in out and "would say: ROLL_BACK" in out
    assert world.promoted == [] and len(db.read_monitoring_events(world.engine)) == before

    world.previous.horizon = DAY_AHEAD
    assert promotion.main([*args, "--as-of", str(AS_OF_DUE)]) == 1
    assert "Comparison refused" in capsys.readouterr().out


# --- The cooldown after a rollback ---
def test_a_rollback_starts_a_cooldown_for_the_restored_champion(world) -> None:
    check(world, act=True)
    restored_at = world.state["promoted_at"]  # the rollback's own promotion
    next_day = AS_OF_DUE + pd.Timedelta(days=1)

    def state(as_of, version="5", promoted_at=restored_at) -> promotion.Cooldown:
        return promotion.cooldown_state(world.engine, as_of, 7, version, promoted_at)

    held = state(next_day)
    assert held.active and held.until == pd.Timestamp("2024-02-11T00:00Z")
    assert held.detail().startswith("v6 rolled back on 2024-02-04 06:00Z")
    assert state(pd.Timestamp("2024-02-10T23:00Z")).active
    assert not state(pd.Timestamp("2024-02-11T00:00Z")).active  # exactly 7 days: free
    # Any later promotion ends it: v6 put back by hand, or v5 promoted again.
    assert not state(next_day, version="6").active
    assert not state(next_day, promoted_at=pd.Timestamp.now(tz="UTC")).active


# --- The monitoring pass (CLI) ---
@pytest.fixture
def passes(world, monkeypatch):
    """Run monitoring passes on the world; ``fire`` decides whether a trigger fires."""
    fire = {"on": False}
    trained: list[str] = []

    def triggers(engine, champion, cfg, *, as_of=None, frame=None):
        check_ = retrain.Check("error", fire["on"], 5.0, 4.0, "7d MAPE 5.00% over 168h")
        return retrain.TriggerDecision(db.to_utc(as_of), champion.version, [check_], None, None)

    def challenger(engine, champion, cfg, **kwargs):
        trained.append(champion.version)
        return retrain.ChallengerOutcome("skipped", "spy", champion.version)

    monkeypatch.setattr(retrain, "evaluate_triggers", triggers)
    monkeypatch.setattr(retrain, "run_champion_challenger", challenger)
    monkeypatch.setattr(retrain, "build_features", lambda *a, **k: pd.DataFrame())

    def run(as_of, *extra: str) -> pd.Series:
        url = str(world.engine.url)
        assert (
            retrain.main(["--no-mlflow", "--database-url", url, "--as-of", str(as_of), *extra]) == 0
        )
        return events(
            world.engine,
            db.EVENT_RETRAIN
            if (fire["on"] or "--force" in extra) and "--check-only" not in extra
            else db.EVENT_CHECK,
        ).iloc[0]

    return SimpleNamespace(run=run, fire=fire, trained=trained)


def test_no_flapping_one_rollback_then_the_cooldown_then_hands_off(world, passes, capsys) -> None:
    # Pass 1: v6 is worse -> rolled back to v5. A fired trigger trains nothing in this pass.
    passes.fire["on"] = True
    first = passes.run(AS_OF_DUE)
    assert world.promoted == ["5"] and passes.trained == []
    assert first["decision"] == "skipped" and "rolled back to v5 in this pass" in first["reason"]
    assert json.loads(first["details"])["post_promotion"]["label"] == db.DECISION_ROLLED_BACK
    assert "Post-promotion check: ROLLED_BACK" in capsys.readouterr().out

    # Pass 2: v5 serves. No second check, no second rollback; the trigger meets the cooldown.
    second = passes.run(AS_OF_DUE + pd.Timedelta(days=1))
    assert world.promoted == ["5"] and passes.trained == []
    assert second["model_version"] == "5" and second["decision"] == db.DECISION_COOLDOWN
    assert "v6 rolled back" in second["reason"]
    assert "already has its verdict: rolled_back" in capsys.readouterr().out
    assert len(events(world.engine)) == 1

    # The operator undoes the rollback by hand: v6 again. Never rolled back a second time,
    # and the promotion ended the cooldown, so a fired trigger trains again.
    world.state.update(champion=world.new, promoted_at=pd.Timestamp.now(tz="UTC"))
    third = passes.run(AS_OF_DUE + pd.Timedelta(days=2))
    assert world.promoted == ["5"] and len(events(world.engine)) == 1
    assert third["model_version"] == "6" and passes.trained == ["6"]


def test_a_rollback_pass_trains_no_challenger_even_when_forced(world, passes) -> None:
    event = passes.run(AS_OF_DUE, "--force")
    assert world.promoted == ["5"] and passes.trained == []
    assert event["decision"] == "skipped" and "no challenger in the same pass" in event["reason"]


def test_a_rollback_whose_event_cannot_be_written_trains_nothing_and_is_recovered(
    world, passes, monkeypatch, capsys
) -> None:
    passes.fire["on"] = True
    real_record = promotion.record_verdict

    def lost(*a, **k):
        raise OSError("connection dropped")

    monkeypatch.setattr(promotion, "record_verdict", lost)
    url = str(world.engine.url)
    with pytest.raises(RuntimeError, match="post-promotion check failed"):  # the job is red
        retrain.main(["--no-mlflow", "--database-url", url, "--as-of", str(AS_OF_DUE), "--force"])
    assert world.promoted == ["5"] and passes.trained == []  # rolled back; no challenger
    assert "ROLLED_BACK (NOT RECORDED)" in capsys.readouterr().out
    assert events(world.engine).empty  # no verdict on record ...
    held = events(world.engine, db.EVENT_RETRAIN).iloc[0]  # ... but the pass is
    assert held["decision"] == "skipped" and "rolled back to v5 in this pass" in held["reason"]

    # The next pass records the missing verdict, dated at the rollback, and the cooldown holds.
    monkeypatch.setattr(promotion, "record_verdict", real_record)
    rolled_back_at = world.state["promoted_at"]
    nxt = passes.run(AS_OF_DUE + pd.Timedelta(days=1))
    (verdict,) = events(world.engine).to_dict("records")
    assert verdict["decision"] == db.DECISION_ROLLED_BACK and verdict["model_version"] == "6"
    assert verdict["reason"].startswith("recovered: v5 has served again since")
    assert verdict["as_of"] == min(rolled_back_at, AS_OF_DUE + pd.Timedelta(days=1))
    assert json.loads(verdict["details"])["previous_version"] == "5"
    assert nxt["model_version"] == "5" and nxt["decision"] == db.DECISION_COOLDOWN
    assert "v6 rolled back" in nxt["reason"] and passes.trained == []
    assert world.promoted == ["5"]  # nothing rolled back twice
    assert ("germany-load-day-ahead", "6", db.DECISION_ROLLED_BACK) in world.tags

    # Recorded once: the pass after that has nothing to recover or check.
    passes.run(AS_OF_DUE + pd.Timedelta(days=2))
    assert len(events(world.engine)) == 1
    assert "already has its verdict: rolled_back" in capsys.readouterr().out


def test_recovery_only_reports_in_a_dry_run(world) -> None:
    world.state.update(champion=world.previous, promoted_at=PROMOTED + pd.Timedelta(days=9))
    report = check(world)  # act=False
    assert report.recovered and report.label == "rolled_back (report only)"
    assert not report.rolled_back and events(world.engine).empty


@pytest.mark.parametrize("mode", ["--dry-run", "--check-only"])
def test_dry_run_and_check_only_change_nothing(world, passes, capsys, mode) -> None:
    passes.fire["on"] = True
    event = passes.run(AS_OF_DUE, mode)
    out = capsys.readouterr().out
    assert "Post-promotion check: ROLLED_BACK (REPORT ONLY)" in out
    assert world.promoted == [] and world.tags == []  # pointer, alias and registry untouched
    assert world.state["champion"] is world.new
    assert events(world.engine).empty  # no verdict recorded: the check stays open
    post = json.loads(event["details"])["post_promotion"]
    assert post["label"] == "rolled_back (report only)" and post["acted"] is False
    # no cooldown started: the next real pass acts as if the report-only pass never ran
    record = world.state["promoted_at"]
    assert not promotion.cooldown_state(world.engine, AS_OF_DUE, 7, "6", record).active
    passes.run(AS_OF_DUE + pd.Timedelta(hours=1))
    assert world.promoted == ["5"]


def test_a_pending_check_holds_new_challengers_unless_forced(world, passes) -> None:
    passes.fire["on"] = True
    held = passes.run(AS_OF_EARLY)
    assert passes.trained == [] and held["decision"] == "skipped"
    assert (
        "post-promotion check of v6 is pending (3 of 7 complete target days scored)"
        in held["reason"]
    )
    passes.run(AS_OF_EARLY, "--force")
    assert passes.trained == ["6"]  # a human asked for the comparison
    assert world.promoted == [] and events(world.engine).empty


def test_a_failing_check_still_records_the_pass_then_fails_the_job(
    world, passes, monkeypatch
) -> None:
    passes.fire["on"] = True

    def boom(*a, **k):
        raise OSError("database went away")

    monkeypatch.setattr(promotion, "post_promotion_check", boom)
    url = str(world.engine.url)
    with pytest.raises(RuntimeError, match="post-promotion check failed: database went away"):
        retrain.main(["--no-mlflow", "--database-url", url, "--as-of", str(AS_OF_DUE)])
    event = events(world.engine, db.EVENT_RETRAIN).iloc[0]  # scored and recorded first
    assert event["decision"] == "skipped" and "post-promotion check failed" in event["reason"]
    assert passes.trained == [] and world.promoted == []


# --- The rollback against a real registry (MLflow + database, temporary stores) ---
def test_rollback_moves_the_alias_and_the_pointer_and_tags_the_version(env) -> None:  # noqa: F811
    import mlflow

    from src.models import registry

    engine = env["engine"]
    previous = registry.champion_record(engine)["version"]
    new = str(registry.register_run_model(env["run_id"]).version)
    registry.promote(new, engine=engine)
    assert str(alias_version()) == new

    report = promotion.PostPromotionReport(
        promotion.ROLL_BACK, "worse on live hours", new, 5.0, 7, previous_version=previous
    )
    promotion.carry_out(engine, registry.REGISTERED_MODEL_NAME, report, AS_OF_DUE)

    assert str(alias_version()) == previous  # MLflow alias ...
    row = registry.champion_record(engine)
    assert row["version"] == previous  # ... and the pointer serving reads
    assert db.get_model_artifact(engine, row["name"], new) is not None  # still exported: undo
    tags = mlflow.MlflowClient().get_model_version(registry.REGISTERED_MODEL_NAME, new).tags
    assert tags["post_promotion"] == db.DECISION_ROLLED_BACK
    event = events(engine).iloc[0]
    assert event["decision"] == db.DECISION_ROLLED_BACK and event["model_version"] == new
    # The rollback's own promotion is not later than its event (a coarse clock can make them
    # equal), so it does not end the cooldown it starts: the restored champion is held.
    assert row["promoted_at"] <= event["created_at"]
    next_day = AS_OF_DUE + pd.Timedelta(days=1)
    assert promotion.cooldown_state(engine, next_day, 7, previous, row["promoted_at"]).active

    # Undo by hand: the documented command path. The check is closed, the cooldown over.
    registry.promote(new, engine=engine)
    assert str(alias_version()) == new
    again = registry.champion_record(engine)
    reason = promotion.open_promotion(engine, new, next_day, again["promoted_at"])
    assert reason[0] is None
    assert not promotion.cooldown_state(engine, next_day, 7, new, again["promoted_at"]).active


def test_a_failed_rollback_records_no_verdict(world, monkeypatch) -> None:
    def broken(*a, **k):
        raise RuntimeError("alias store down")

    monkeypatch.setattr(promotion.registry, "promote", broken)
    with pytest.raises(RuntimeError, match="alias store down"):
        check(world, act=True)
    assert events(world.engine).empty and world.tags == []  # the next pass tries again


def test_a_failed_version_tag_is_logged_never_raised(tmp_path, monkeypatch, caplog) -> None:
    from config.settings import get_settings

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    get_settings.cache_clear()
    promotion.tag_version("no-such-model", "1", db.DECISION_CONFIRMED)
    assert "could not tag no-such-model v1 with post_promotion=confirmed" in caplog.text
    get_settings.cache_clear()


# --- Visibility ---
def test_rollback_and_pending_gauges_are_sent_on_every_pass(world) -> None:
    """0 or 1 on every pass, so the "Champion rolled back" monitor recovers by itself."""
    decision = SimpleNamespace(champion_version="6", performance=None, drift=None, checks=[])

    def sent(post) -> dict:
        m, rec = enabled("monitor")
        retrain.report_metrics(m, decision, None, post)
        m.flush()
        return rec.metrics()

    def flags(post) -> tuple[float, float]:
        got = sent(post)
        return value(got, "retrain.rolled_back"), value(got, "post_promotion.pending")

    assert flags(None) == (0, 0)  # e.g. the check failed
    assert flags(check(world, AS_OF_EARLY)) == (0, 1)
    assert flags(check(world)) == (0, 0)  # report only: nothing was rolled back
    rolled = check(world, act=True)
    got = sent(rolled)
    assert (value(got, "retrain.rolled_back"), value(got, "post_promotion.pending")) == (1, 0)
    assert value(got, "post_promotion.improvement_pct", previous_version="v5") < -5
    # The pass after the rollback sends 0 again.
    after = check(world, AS_OF_DUE + pd.Timedelta(days=1), act=True)
    assert after.verdict == promotion.NOT_APPLICABLE and flags(after) == (0, 0)
    assert "post_promotion.improvement_pct" not in sent(after)


def test_every_kind_and_decision_fits_its_column() -> None:
    """Free text, so no migration - but Postgres rejects an overflow that SQLite accepts."""
    columns = db.MonitoringEvent.__table__.c
    kinds = [db.EVENT_CHECK, db.EVENT_RETRAIN, db.EVENT_POST_PROMOTION]
    decisions = [
        db.DECISION_KEPT,
        db.DECISION_COOLDOWN,
        db.DECISION_PROMOTED,
        db.DECISION_CONFIRMED,
        db.DECISION_ROLLED_BACK,
        db.DECISION_INCONCLUSIVE,
        "skipped",
    ]
    assert max(map(len, kinds)) <= columns.kind.type.length == 16
    assert max(len(f"{d} (dry-run)") for d in decisions) <= columns.decision.type.length == 32
    assert set(promotion._DECISIONS.values()) <= set(decisions)
