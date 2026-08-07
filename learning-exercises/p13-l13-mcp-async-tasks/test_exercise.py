"""Тесты к уроку «Async Tasks в MCP (SEP-1686)». Правь exercise.py."""

import pytest

from exercise import (
    advance,
    cancel_task,
    choose_task_support,
    is_expired,
    is_terminal,
    new_task,
    recover_after_crash,
    tasks_result,
)


# ------------------------------------------------------ choose_task_support
def test_fast_tool_stays_synchronous():
    assert choose_task_support(0.2) == "forbidden"


def test_medium_tool_lets_the_client_decide():
    assert choose_task_support(12) == "optional"


def test_slow_tool_requires_task_augmentation():
    """Дольше 30 секунд — синхронный вызов гарантированно оборвётся."""
    assert choose_task_support(180) == "required"


def test_thirty_seconds_is_still_optional():
    """Граница включительная: ровно 30 — ещё optional, 31 — уже required."""
    assert choose_task_support(30) == "optional"
    assert choose_task_support(31) == "required"


def test_negative_estimate_is_rejected():
    with pytest.raises(ValueError):
        choose_task_support(-1)


# ------------------------------------------------------------------ new_task
def test_new_task_starts_working():
    task = new_task("tsk_1", 900000, 1000)
    assert task["state"] == "working"
    assert task["id"] == "tsk_1"


def test_new_task_records_creation_time_and_ttl():
    task = new_task("tsk_1", 900000, 1000)
    assert (task["createdAt"], task["ttl"]) == (1000, 900000)


def test_new_task_has_no_result_yet():
    task = new_task("tsk_1", 900000, 1000)
    assert task["result"] is None and task["error"] is None
    assert task["progress"] == 0.0


# --------------------------------------------------------------- is_terminal
def test_working_is_not_terminal():
    assert is_terminal("working") is False


def test_completed_failed_cancelled_are_terminal():
    assert all(is_terminal(s) for s in ("completed", "failed", "cancelled"))


def test_input_required_is_not_terminal():
    """Задача ждёт elicitation и вернётся в working — это не конец."""
    assert is_terminal("input_required") is False


# ---------------------------------------------------------------- is_expired
def test_task_is_alive_before_ttl():
    assert is_expired(new_task("t", 1000, 0), 999) is False


def test_task_expires_exactly_at_ttl():
    assert is_expired(new_task("t", 1000, 0), 1000) is True


def test_ttl_counts_from_creation_not_from_last_update():
    """Задача, которую долго обновляли, всё равно протухает по createdAt."""
    task = new_task("t", 1000, 0)
    busy = advance(task, "input_required", 900)
    assert busy["updatedAt"] == 900
    assert is_expired(busy, 1000) is True


# ------------------------------------------------------------------- advance
def test_advance_moves_to_completed_with_payload():
    done = advance(new_task("t", 1000, 0), "completed", 5, payload="report")
    assert (done["state"], done["result"]) == ("completed", "report")


def test_advance_stores_failure_payload_in_error_not_result():
    bad = advance(new_task("t", 1000, 0), "failed", 5, payload="boom")
    assert bad["error"] == "boom"
    assert bad["result"] is None


def test_advance_records_progress_and_time():
    task = advance(new_task("t", 1000, 0), "input_required", 7, progress=0.5)
    assert (task["progress"], task["updatedAt"]) == (0.5, 7)


def test_advance_does_not_mutate_the_input_task():
    """Состояние лежит в durable store — мутация на месте прячет момент записи."""
    task = new_task("t", 1000, 0)
    advance(task, "completed", 5, payload="x")
    assert task["state"] == "working" and task["result"] is None


def test_terminal_task_cannot_be_revived():
    """Машина состояний append-only: из completed нет ни одного перехода."""
    done = advance(new_task("t", 1000, 0), "completed", 5)
    with pytest.raises(ValueError):
        advance(done, "working", 6)


def test_working_cannot_loop_into_itself():
    with pytest.raises(ValueError):
        advance(new_task("t", 1000, 0), "working", 5)


def test_unknown_state_is_rejected():
    with pytest.raises(ValueError):
        advance(new_task("t", 1000, 0), "done", 5)


def test_elicitation_loop_goes_there_and_back():
    """working -> input_required -> working — единственный цикл в машине."""
    task = advance(new_task("t", 1000, 0), "input_required", 5)
    back = advance(task, "working", 6)
    assert back["state"] == "working"


# --------------------------------------------------------------- cancel_task
def test_cancel_moves_working_task_to_cancelled():
    assert cancel_task(new_task("t", 1000, 0), 5)["state"] == "cancelled"


def test_cancel_is_idempotent_on_a_cancelled_task():
    """tasks/cancel обязан быть no-op на терминальной задаче, а не ошибкой."""
    once = cancel_task(new_task("t", 1000, 0), 5)
    twice = cancel_task(once, 6)
    assert twice["state"] == "cancelled"


def test_cancel_does_not_overwrite_a_completed_result():
    done = advance(new_task("t", 1000, 0), "completed", 5, payload="report")
    assert cancel_task(done, 6) == done


# -------------------------------------------------------------- tasks_result
def test_unknown_task_id_is_404():
    got = tasks_result({}, "нет такой", 0)
    assert (got["status"], got["error"]) == (404, "unknown_task")


def test_unfinished_task_is_404_not_ready():
    got = tasks_result({"t": new_task("t", 1000, 0)}, "t", 5)
    assert (got["status"], got["state"], got["error"]) == (404, "working", "not_ready")


def test_completed_task_returns_its_payload():
    store = {"t": advance(new_task("t", 1000, 0), "completed", 5, payload="report")}
    got = tasks_result(store, "t", 6)
    assert (got["status"], got["result"]) == (200, "report")


def test_expired_task_forgets_even_its_state():
    """После ttl сервер не обязан помнить ничего — ни результат, ни состояние."""
    store = {"t": advance(new_task("t", 1000, 0), "completed", 5, payload="report")}
    got = tasks_result(store, "t", 1000)
    assert (got["status"], got["state"], got["error"]) == (404, None, "expired")


def test_failed_task_returns_its_error_with_200():
    store = {"t": advance(new_task("t", 1000, 0), "failed", 5, payload="boom")}
    got = tasks_result(store, "t", 6)
    assert (got["status"], got["error"], got["result"]) == (200, "boom", None)


# -------------------------------------------------------- recover_after_crash
def test_crash_marks_inflight_task_failed():
    store = {"t": new_task("t", 1000, 0)}
    assert recover_after_crash(store, 5)["t"]["error"] == "CRASH_RECOVERY"


def test_crash_recovery_preserves_completed_results():
    done = advance(new_task("t", 1000, 0), "completed", 5, payload="report")
    assert recover_after_crash({"t": done}, 6)["t"]["result"] == "report"


def test_crash_recovery_drops_expired_tasks():
    store = {"old": new_task("old", 100, 0), "new": new_task("new", 10000, 0)}
    assert set(recover_after_crash(store, 500)) == {"new"}


def test_crash_recovery_also_fails_tasks_waiting_for_input():
    """input_required тоже незавершённое: поток, который ждал ответа, мёртв."""
    waiting = advance(new_task("t", 1000, 0), "input_required", 5)
    assert recover_after_crash({"t": waiting}, 6)["t"]["state"] == "failed"


def test_crash_recovery_does_not_mutate_the_old_store():
    store = {"t": new_task("t", 1000, 0)}
    recover_after_crash(store, 5)
    assert store["t"]["state"] == "working"
