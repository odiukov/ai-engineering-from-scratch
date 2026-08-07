"""Тесты к уроку «Чекпоинты и откат». Правь exercise.py."""

import pytest

from exercise import (
    checkpoint,
    claim_lease,
    find_checkpoint,
    lease_expired,
    restore,
    rollback_to,
    run_step,
    snapshot,
)


def fresh_state():
    """Мини-база: два баланса и журнал отправленных переводов."""
    return {"balance_A": 1500, "balance_B": 200, "sent": []}


def transfer_step(amount=100, min_balance=200, verify_ok=True, sid="tx-001"):
    """Шаг перевода денег с precondition по минимальному остатку."""

    def precondition(state):
        return state["balance_A"] - amount >= min_balance

    def apply_(state):
        state["balance_A"] -= amount
        state["balance_B"] += amount
        state["sent"].append(sid)

    def verify(state):
        return verify_ok and sid in state["sent"]

    return {"id": sid, "precondition": precondition, "apply": apply_, "verify": verify}


# ----------------------------------------------------------------- snapshot
def test_snapshot_equals_the_original():
    state = fresh_state()
    assert snapshot(state) == state


def test_snapshot_does_not_follow_later_mutation():
    state = fresh_state()
    snap = snapshot(state)
    state["balance_A"] = 0
    assert snap["balance_A"] == 1500


def test_snapshot_copies_nested_lists():
    """Поверхностной копии здесь недостаточно — на этом ломается откат."""
    state = fresh_state()
    snap = snapshot(state)
    state["sent"].append("tx-001")
    assert snap["sent"] == []


def test_snapshot_copies_nested_dicts():
    state = {"limits": {"daily": 500}}
    snap = snapshot(state)
    state["limits"]["daily"] = 0
    assert snap["limits"]["daily"] == 500


# ------------------------------------------------------------------ restore
def test_restore_brings_back_values():
    state = fresh_state()
    snap = snapshot(state)
    state["balance_A"] = 0
    restore(state, snap)
    assert state["balance_A"] == 1500


def test_restore_removes_keys_added_after_the_snapshot():
    """Полное восстановление: мусор от неудачного действия не остаётся."""
    state = fresh_state()
    snap = snapshot(state)
    state["half_written_row"] = "oops"
    restore(state, snap)
    assert "half_written_row" not in state


def test_restore_mutates_the_same_object():
    state = fresh_state()
    alias = state
    snap = snapshot(state)
    state["balance_A"] = 0
    restore(state, snap)
    assert alias is state
    assert alias["balance_A"] == 1500


def test_restore_does_not_alias_the_snapshot():
    """После отката запись в состояние не должна портить запись в журнале."""
    state = fresh_state()
    snap = snapshot(state)
    restore(state, snap)
    state["sent"].append("tx-001")
    assert snap["sent"] == []


# --------------------------------------------------------------- checkpoint
def test_checkpoint_appends_named_entry_with_time():
    log = []
    entry = checkpoint(log, "tx:before", fresh_state(), now=5.0)
    assert log == [entry]
    assert entry["name"] == "tx:before"
    assert entry["at"] == 5.0


def test_checkpoint_stores_an_independent_copy():
    log = []
    state = fresh_state()
    checkpoint(log, "tx:before", state, now=0.0)
    state["sent"].append("tx-001")
    assert log[0]["state"]["sent"] == []


def test_checkpoint_persists_every_transition():
    log = []
    state = fresh_state()
    for name, at in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        checkpoint(log, name, state, now=at)
    assert [entry["name"] for entry in log] == ["a", "b", "c"]


# ----------------------------------------------------------- find_checkpoint
def test_find_checkpoint_returns_none_when_absent():
    assert find_checkpoint([]) is None
    assert find_checkpoint([], "tx:before") is None


def test_find_checkpoint_is_latest_wins():
    log = []
    state = fresh_state()
    checkpoint(log, "tx:before", state, now=1.0)
    state["balance_A"] = 0
    checkpoint(log, "tx:before", state, now=9.0)
    assert find_checkpoint(log, "tx:before")["at"] == 9.0


def test_find_checkpoint_without_name_returns_the_last_entry():
    log = []
    checkpoint(log, "first", fresh_state(), now=1.0)
    checkpoint(log, "second", fresh_state(), now=2.0)
    assert find_checkpoint(log)["name"] == "second"


# ------------------------------------------------------------- rollback_to
def test_rollback_restores_state_completely():
    log, state = [], fresh_state()
    checkpoint(log, "tx:before", state, now=0.0)
    state["balance_A"] = 0
    state["extra"] = "garbage"
    rollback_to(state, log, "tx:before")
    assert state == fresh_state()


def test_rollback_removes_side_effects_recorded_after_the_checkpoint():
    """Побочный эффект жил в состоянии — значит откат его убирает."""
    log, state = [], fresh_state()
    checkpoint(log, "tx:before", state, now=0.0)
    state["sent"].append("tx-001")
    rollback_to(state, log, "tx:before")
    assert state["sent"] == []


def test_rollback_to_unknown_checkpoint_raises_lookup_error():
    log, state = [], fresh_state()
    checkpoint(log, "tx:before", state, now=0.0)
    with pytest.raises(LookupError):
        rollback_to(state, log, "tx:nowhere")


def test_rollback_keeps_the_state_object_identity():
    log, state = [], fresh_state()
    alias = state
    checkpoint(log, "tx:before", state, now=0.0)
    state["balance_A"] = 0
    rollback_to(state, log, "tx:before")
    assert alias is state
    assert alias["balance_A"] == 1500


# ------------------------------------------------------------ lease_expired
def test_missing_lease_counts_as_expired():
    assert lease_expired(None, now=0.0) is True


def test_live_lease_is_not_expired():
    assert lease_expired({"worker": "w1", "until": 10.0}, now=9.0) is False


def test_lease_expires_at_its_deadline():
    assert lease_expired({"worker": "w1", "until": 10.0}, now=10.0) is True


# -------------------------------------------------------------- claim_lease
def test_free_lease_is_claimed():
    assert claim_lease(None, "w1", now=0.0, duration=30.0) == {
        "worker": "w1",
        "until": 30.0,
    }


def test_live_lease_of_another_worker_is_refused():
    held = {"worker": "w1", "until": 30.0}
    assert claim_lease(held, "w2", now=10.0, duration=30.0) is None


def test_expired_lease_is_reclaimed_by_another_worker():
    """Воркер упал, аренда истекла — работу забирает следующий."""
    dead = {"worker": "w1", "until": 30.0}
    assert claim_lease(dead, "w2", now=30.0, duration=30.0) == {
        "worker": "w2",
        "until": 60.0,
    }


def test_holder_can_renew_its_own_lease():
    held = {"worker": "w1", "until": 30.0}
    assert claim_lease(held, "w1", now=10.0, duration=30.0)["until"] == 40.0


# ----------------------------------------------------------------- run_step
def test_clean_run_applies_once_and_reports_ok():
    log, state = [], fresh_state()
    assert run_step(state, log, transfer_step(), now=1.0) == "ok"
    assert state["balance_A"] == 1400
    assert state["balance_B"] == 300
    assert state["sent"] == ["tx-001"]


def test_precondition_failure_never_applies():
    """Одобрено при балансе 1500, применяется при 250 — действие не идёт."""
    log, state = [], fresh_state()
    state["balance_A"] = 250
    assert run_step(state, log, transfer_step(), now=1.0) == "precondition-failed"
    assert state["balance_A"] == 250
    assert state["sent"] == []


def test_verify_failure_rolls_state_back_completely():
    log, state = [], fresh_state()
    before = snapshot(state)
    assert run_step(state, log, transfer_step(verify_ok=False), now=1.0) == "rolled-back"
    assert state == before


def test_verify_failure_leaves_no_side_effect_residue():
    """Откат обязан убрать и деньги, и запись об отправке."""
    log, state = [], fresh_state()
    run_step(state, log, transfer_step(verify_ok=False), now=1.0)
    assert state["sent"] == []
    assert state["balance_B"] == 200


def test_retry_after_success_does_not_apply_twice():
    log, state = [], fresh_state()
    run_step(state, log, transfer_step(), now=1.0)
    assert run_step(state, log, transfer_step(), now=2.0) == "already-done"
    assert run_step(state, log, transfer_step(), now=3.0) == "already-done"
    assert state["sent"] == ["tx-001"]
    assert state["balance_A"] == 1400


def test_every_transition_of_a_clean_run_is_checkpointed():
    log, state = [], fresh_state()
    run_step(state, log, transfer_step(), now=1.0)
    assert [entry["name"] for entry in log] == [
        "tx-001:before",
        "tx-001:applied",
        "tx-001:verified",
    ]


def test_rolled_back_run_is_checkpointed_as_such():
    log, state = [], fresh_state()
    run_step(state, log, transfer_step(verify_ok=False), now=1.0)
    assert [entry["name"] for entry in log] == [
        "tx-001:before",
        "tx-001:applied",
        "tx-001:rolled-back",
    ]
    assert find_checkpoint(log, "tx-001:verified") is None
