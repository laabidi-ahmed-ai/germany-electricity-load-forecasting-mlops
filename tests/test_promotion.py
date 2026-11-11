"""Promotion-policy tests (offline): the cooldown after a rejected challenger."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data import db
from src.features.horizons import DAY_AHEAD, select_features
from src.monitoring import promotion, retrain
from tests.conftest import fake_loaded_model
from tests.test_monitoring import fake_runner, feature_frame, seed_db

REJECTED = pd.Timestamp("2026-10-05T09:00Z")


def record(
    engine,
    as_of,
    *,
    kind=db.EVENT_RETRAIN,
    decision="kept",
    reason="margin",
    version="3",
    created_at=None,
) -> None:
    """A monitoring event decided against champion ``version``."""
    db.insert_monitoring_event(
        engine,
        created_at=db.to_utc(created_at if created_at is not None else as_of).to_pydatetime(),
        as_of=db.to_utc(as_of).to_pydatetime(),
        kind=kind,
        triggered=True,
        decision=decision,
        reason=reason,
        model_version=version,
    )


def state(engine, as_of, days=7, version="3", promoted_at=None) -> promotion.Cooldown:
    return promotion.cooldown_state(engine, pd.Timestamp(as_of), days, version, promoted_at)


# --- The rule ---
@pytest.mark.parametrize(
    ("as_of", "active"),
    [
        ("2026-10-05T09:00Z", True),  # the rejecting pass itself
        ("2026-10-05T23:00Z", True),
        ("2026-10-11T23:59Z", True),  # day N - 1: still held
        ("2026-10-12T00:00Z", False),  # exactly N calendar days later: free
        ("2026-10-12T06:50Z", False),  # an earlier hour than the rejection does not matter
    ],
)
def test_cooldown_holds_for_exactly_n_utc_calendar_days(engine, as_of, active) -> None:
    record(engine, REJECTED, reason="challenger improves MAE by +1.0% < required 3.0% margin")
    got = state(engine, as_of)
    assert got.active is active
    assert got.until == pd.Timestamp("2026-10-12T00:00Z")
    assert got.rejected_at == REJECTED
    assert "+1.0% < required 3.0% margin" in got.detail()
    assert "no new challenger before 2026-10-12 UTC (7-day cooldown)" in got.detail()


@pytest.mark.parametrize(
    ("kind", "decision"),
    [
        (db.EVENT_RETRAIN, "kept (dry-run)"),  # a dry run must not change later passes
        (db.EVENT_RETRAIN, "skipped"),  # compared nothing
        (db.EVENT_RETRAIN, "promoted"),
        (db.EVENT_RETRAIN, "promoted (dry-run)"),
        (db.EVENT_RETRAIN, db.DECISION_COOLDOWN),  # a held pass does not extend the cooldown
        (db.EVENT_CHECK, None),
    ],
)
def test_only_a_real_rejection_starts_a_cooldown(engine, kind, decision) -> None:
    record(engine, REJECTED, kind=kind, decision=decision)
    assert not state(engine, REJECTED + pd.Timedelta(days=1)).active


def test_the_latest_rejection_counts_and_later_events_are_ignored(engine) -> None:
    record(engine, REJECTED)
    record(engine, REJECTED + pd.Timedelta(days=6), reason="second")
    got = state(engine, REJECTED + pd.Timedelta(days=8))
    assert got.active and got.rejection == "second"
    assert got.until == pd.Timestamp("2026-10-18T00:00Z")
    # A pass replayed with an older --as-of does not see rejections from after it.
    before = state(engine, REJECTED - pd.Timedelta(days=1))
    assert not before.active and before.rejected_at is None


def test_zero_days_disables_the_cooldown(engine) -> None:
    record(engine, REJECTED)
    assert not state(engine, REJECTED, days=0).active


def test_no_rejection_on_record(engine) -> None:
    got = state(engine, REJECTED)
    assert not got.active and got.detail() == "no rejected challenger on record"


# --- Only the champion the rejection was decided against ---
def test_a_rejection_against_another_champion_does_not_hold(engine) -> None:
    """Kept against v5, then v6 became champion: v6 is not held by v5's rejection."""
    record(engine, REJECTED, version="5")
    got = state(engine, REJECTED + pd.Timedelta(days=1), version="6")
    assert not got.active
    assert got.detail() == "latest rejection (2026-10-05) was against v5; the champion is now v6"
    assert state(engine, REJECTED + pd.Timedelta(days=1), version="5").active  # v5 still is


def test_a_rejection_against_the_current_champion_holds(engine) -> None:
    record(engine, REJECTED, version="6")
    promoted = REJECTED - pd.Timedelta(days=10)  # promoted before the rejection
    assert state(engine, REJECTED + pd.Timedelta(days=1), version="6", promoted_at=promoted).active


def test_promoting_the_same_version_again_ends_the_cooldown(engine) -> None:
    """v5 rejected, v6 promoted, v5 promoted back by hand: the old rejection is spent."""
    record(engine, REJECTED, version="5")
    again = REJECTED + pd.Timedelta(days=2)
    got = state(engine, REJECTED + pd.Timedelta(days=3), version="5", promoted_at=again)
    assert not got.active and "predates the promotion of v5 on 2026-10-07" in got.detail()


# --- The monitoring pass (CLI) ---
@pytest.fixture
def fired(engine, monkeypatch):
    """A database where the error trigger fires, and a spy in place of champion/challenger."""
    load = seed_db(engine, days=60, model_noise=100, model_bias=3_000)  # ~5.6 % MAPE
    frame = feature_frame(load)
    champ = fake_loaded_model(
        select_features(frame.columns, DAY_AHEAD), train_end=frame.index.max()
    )
    monkeypatch.setattr(retrain.registry, "load_champion", lambda engine: champ)
    monkeypatch.setattr(retrain, "build_features", lambda *a, **k: frame)
    monkeypatch.setattr(retrain.drift_mod, "run_evidently", fake_runner(set()))
    calls: list[dict] = []

    def spy(engine, champion, cfg, **kwargs):
        calls.append(kwargs)
        return retrain.ChallengerOutcome("kept", "margin not met", champion.version)

    monkeypatch.setattr(retrain, "run_champion_challenger", spy)
    as_of = load["timestamp_utc"].max()  # 2024-02-29 23:00 UTC
    return {"url": str(engine.url), "as_of": as_of, "calls": calls}


def run_pass(fired, *extra: str) -> int:
    return retrain.main(
        ["--no-mlflow", "--database-url", fired["url"], "--as-of", str(fired["as_of"]), *extra]
    )


def test_a_pass_held_by_the_cooldown_still_scores_and_records_the_triggers(
    engine, fired, capsys
) -> None:
    record(engine, fired["as_of"] - pd.Timedelta(days=3), reason="first rejection", version="1")
    assert run_pass(fired) == 0
    out = capsys.readouterr().out
    assert "RETRAIN TRIGGERED" in out and "COOLDOWN" in out
    assert fired["calls"] == []  # nothing trained

    event = db.read_monitoring_events(engine).iloc[0]
    assert event["kind"] == db.EVENT_RETRAIN and event["decision"] == db.DECISION_COOLDOWN
    assert bool(event["triggered"]) and "first rejection" in event["reason"]
    assert event["model_mape_7d"] > 4 and pd.notna(event["official_mape_7d"])
    details = json.loads(event["details"])
    assert [c["name"] for c in details["decision"]["checks"]] == ["error", "official", "drift"]
    assert details["outcome"]["cooldown"] == "held"
    assert details["outcome"]["triggers"] and "MAPE" in details["outcome"]["triggers"][0]


def test_force_bypasses_the_cooldown_and_says_so(engine, fired, capsys, caplog) -> None:
    record(engine, fired["as_of"] - pd.Timedelta(days=3), version="1")
    assert run_pass(fired, "--force") == 0
    assert len(fired["calls"]) == 1  # the comparison ran
    assert "cooldown until" in capsys.readouterr().out
    assert "cooldown bypassed by --force" in caplog.text
    details = json.loads(db.read_monitoring_events(engine).iloc[0]["details"])
    assert details["outcome"]["cooldown"] == "bypassed"


def test_the_pass_n_days_after_a_rejection_trains_again(engine, fired) -> None:
    record(engine, fired["as_of"].normalize() - pd.Timedelta(days=7), version="1")
    assert run_pass(fired) == 0
    assert len(fired["calls"]) == 1
    assert db.read_monitoring_events(engine).iloc[0]["decision"] == "kept"


def test_dry_run_rejections_never_hold_a_later_pass(engine, fired) -> None:
    record(engine, fired["as_of"] - pd.Timedelta(days=1), decision="kept (dry-run)", version="1")
    assert run_pass(fired) == 0
    assert len(fired["calls"]) == 1


def test_a_new_champion_trains_despite_its_predecessors_rejection(engine, fired) -> None:
    """The serving champion is v1 (the fixture); the rejection was against v0."""
    record(engine, fired["as_of"] - pd.Timedelta(days=1), version="0")
    assert run_pass(fired) == 0
    assert len(fired["calls"]) == 1  # trained, not held


def test_the_pass_reads_the_champions_promotion_time(engine, fired, monkeypatch) -> None:
    """A promotion of v1 after its rejection (e.g. by hand) ends the cooldown."""
    rejected = fired["as_of"] - pd.Timedelta(days=2)
    record(engine, rejected, version="1")
    monkeypatch.setattr(
        retrain.registry,
        "champion_record",
        lambda engine: {"promoted_at": rejected + pd.Timedelta(days=1)},
    )
    assert run_pass(fired) == 0
    assert len(fired["calls"]) == 1
