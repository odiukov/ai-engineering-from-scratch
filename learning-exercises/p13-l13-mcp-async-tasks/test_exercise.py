"""Тесты к уроку «Async Tasks в MCP 2025-11-25». Правь exercise.py."""

import pytest

from exercise import (
    InvalidParams,
    advance,
    cancel_task,
    choose_task_support,
    create_task_result,
    is_expired,
    is_terminal,
    new_task,
    recover_after_crash,
    tasks_get,
    tasks_result,
)

T0 = "2025-11-25T10:30:00Z"
T1 = "2025-11-25T10:30:00.999Z"
T2 = "2025-11-25T10:30:01Z"
T3 = "2025-11-25T10:30:02Z"


def test_fast_medium_and_slow_tools_choose_the_expected_support():
    assert choose_task_support(0.2) == "forbidden"
    assert choose_task_support(12) == "optional"
    assert choose_task_support(180) == "required"


def test_negative_estimate_is_rejected():
    with pytest.raises(ValueError):
        choose_task_support(-1)


def test_new_task_uses_current_wire_field_names_and_iso_times():
    record = new_task("tsk_1", 900000, T0)
    task = record["task"]
    assert task["taskId"] == "tsk_1" and task["status"] == "working"
    assert task["createdAt"] == T0 and task["lastUpdatedAt"] == T0
    assert "id" not in task and "state" not in task and "updatedAt" not in task


def test_create_task_result_wraps_the_task_not_meta():
    result = create_task_result(new_task("tsk_1", 900000, T0))
    assert set(result) == {"task"} and result["task"]["taskId"] == "tsk_1"


def test_non_iso_creation_time_is_rejected():
    with pytest.raises(TypeError):
        new_task("tsk_1", 900000, 0)


def test_terminal_statuses_and_input_required():
    assert all(is_terminal(s) for s in ("completed", "failed", "cancelled"))
    assert not is_terminal("working") and not is_terminal("input_required")


def test_ttl_counts_from_creation_and_accepts_unlimited():
    finite = new_task("t", 1000, T0)
    unlimited = new_task("u", None, T0)
    assert not is_expired(finite, T1) and is_expired(finite, T2)
    assert not is_expired(unlimited, "2035-11-25T10:30:00Z")


def test_advance_updates_status_and_last_updated_at_without_mutating_input():
    record = new_task("t", 1000, T0)
    waiting = advance(record, "input_required", T1, "Need a format")
    assert waiting["task"]["status"] == "input_required"
    assert waiting["task"]["statusMessage"] == "Need a format"
    assert waiting["task"]["lastUpdatedAt"] == T1
    assert record["task"]["status"] == "working"


def test_input_required_can_complete_directly():
    waiting = advance(new_task("t", 1000, T0), "input_required", T1)
    done = advance(waiting, "completed", T2, result={"content": []})
    assert done["task"]["status"] == "completed"


def test_terminal_task_cannot_be_revived():
    done = advance(new_task("t", 1000, T0), "completed", T1, result={})
    with pytest.raises(ValueError):
        advance(done, "working", T2)


def test_tasks_get_returns_the_full_task_shape():
    record = new_task("t", 1000, T0)
    got = tasks_get({"t": record}, "t", T1)
    assert set(got) == {
        "taskId", "status", "createdAt", "lastUpdatedAt", "ttl", "pollInterval"
    }


def test_tasks_get_rejects_unknown_or_expired_ids():
    with pytest.raises(KeyError):
        tasks_get({}, "missing", T0)
    with pytest.raises(KeyError):
        tasks_get({"t": new_task("t", 1000, T0)}, "t", T2)


def test_cancel_moves_working_task_to_cancelled():
    cancelled = cancel_task(new_task("t", 1000, T0), T1)
    assert cancelled["task"]["status"] == "cancelled"


def test_terminal_cancellation_is_invalid_params_minus_32602():
    cancelled = cancel_task(new_task("t", 1000, T0), T1)
    with pytest.raises(InvalidParams) as caught:
        cancel_task(cancelled, T2)
    assert caught.value.code == -32602


def test_tasks_result_returns_the_underlying_tool_result_exactly():
    payload = {"content": [{"type": "text", "text": "report"}], "isError": False}
    done = advance(new_task("t", 1000, T0), "completed", T1, result=payload)
    assert tasks_result({"t": done}, "t", T1) == payload


def test_tasks_result_does_not_return_404_for_working_task():
    with pytest.raises(BlockingIOError):
        tasks_result({"t": new_task("t", 1000, T0)}, "t", T1)


def test_tasks_result_waits_until_terminal_when_waiter_is_supplied():
    store = {"t": new_task("t", 1000, T0)}
    payload = {"content": [{"type": "text", "text": "ready"}], "isError": False}

    def finish(current, task_id):
        current[task_id] = advance(current[task_id], "completed", T1, result=payload)

    assert tasks_result(store, "t", T1, wait=finish) == payload


def test_failed_task_returns_the_underlying_json_rpc_error():
    error = {"code": -32000, "message": "boom"}
    failed = advance(new_task("t", 1000, T0), "failed", T1, result=error)
    assert tasks_result({"t": failed}, "t", T1) == error


def test_crash_recovery_fails_inflight_and_preserves_completed():
    working = new_task("w", 10000, T0)
    done = advance(new_task("d", 10000, T0), "completed", T1, result={"ok": True})
    recovered = recover_after_crash({"w": working, "d": done}, T2)
    assert recovered["w"]["task"]["status"] == "failed"
    assert recovered["w"]["result"]["message"] == "CRASH_RECOVERY"
    assert recovered["d"]["result"] == {"ok": True}


def test_crash_recovery_drops_expired_records_without_mutating_store():
    store = {"old": new_task("old", 1000, T0), "new": new_task("new", 10000, T0)}
    assert set(recover_after_crash(store, T2)) == {"new"}
    assert store["new"]["task"]["status"] == "working"
