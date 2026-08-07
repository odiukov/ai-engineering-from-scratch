"""Тесты к уроку «Долгоживущие фоновые агенты: durable execution». Правь exercise.py."""

import pytest

from exercise import (
    WorkflowCrash,
    activity_key,
    deterministic_value,
    execution_count,
    find_completed,
    needs_fresh_approval,
    replay_state,
    run_activity,
    run_workflow,
)


def pipeline():
    """Три активности и список calls, куда они пишут свои побочные эффекты."""
    calls = []

    def fetch(query):
        calls.append(("fetch", query))
        return len(query) * 3

    def llm(doc_count):
        calls.append(("llm", doc_count))
        return f"summary({doc_count})"

    def write(summary):
        calls.append(("write", summary))
        return f"report://{summary}"

    activities = (("fetch", fetch), ("llm", llm), ("write", write))
    return activities, calls


def done_event(thread_id, name, args, result, at=0.0):
    """Готовое событие "done" — чтобы тестировать чтение журнала без запуска."""
    return {
        "thread_id": thread_id,
        "key": activity_key(name, args),
        "name": name,
        "args": list(args),
        "status": "done",
        "result": result,
        "at": at,
    }


# ------------------------------------------------------------ activity_key
def test_activity_key_is_stable_for_the_same_call():
    assert activity_key("double", (21,)) == activity_key("double", (21,))


def test_activity_key_separates_argument_order():
    """(a, b) и (b, a) — разные вызовы, значит и разные ключи."""
    assert activity_key("fetch", ("hi", 2)) != activity_key("fetch", (2, "hi"))


def test_activity_key_separates_names():
    assert activity_key("fetch", (1,)) != activity_key("llm", (1,))


# ---------------------------------------------------------- find_completed
def test_find_completed_returns_none_for_empty_log():
    assert find_completed([], "t-1", activity_key("double", (21,))) is None


def test_find_completed_ignores_started_without_done():
    """Активность начали и упали — результата нет, её надо выполнить заново."""
    started_only = {
        "thread_id": "t-1",
        "key": activity_key("half", (1,)),
        "name": "half",
        "args": [1],
        "status": "started",
        "result": None,
        "at": 0.0,
    }
    assert find_completed([started_only], "t-1", started_only["key"]) is None


def test_find_completed_is_scoped_by_thread_id():
    log = [done_event("t-1", "double", (21,), 42)]
    key = activity_key("double", (21,))
    assert find_completed(log, "t-1", key)["result"] == 42
    assert find_completed(log, "t-2", key) is None


# ------------------------------------------------------------ run_activity
def test_run_activity_executes_once_and_returns_result():
    log, calls = [], []

    def double(x):
        calls.append(x)
        return x * 2

    assert run_activity(log, "t-1", "double", (21,), double) == 42
    assert calls == [21]


def test_second_call_replays_from_log_without_calling_fn():
    """Реплей возвращает записанное, даже если новая fn вернула бы другое."""
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2)
    assert run_activity(log, "t-1", "double", (21,), lambda x: 0) == 42


def test_run_activity_logs_started_before_done():
    """Сбой между записями должен читаться как «начали, но не знаем результат»."""
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2, now=7.0)
    assert [event["status"] for event in log] == ["started", "done"]
    assert log[-1]["result"] == 42
    assert log[-1]["at"] == 7.0


def test_run_activity_does_not_share_results_across_threads():
    """Два одновременных сеанса делят журнал, но не результаты."""
    log, calls = [], []

    def double(x):
        calls.append(x)
        return x * 2

    run_activity(log, "t-1", "double", (21,), double)
    run_activity(log, "t-2", "double", (21,), double)
    assert calls == [21, 21]


# ------------------------------------------------------ deterministic_value
def test_deterministic_value_records_the_first_result():
    log = []
    assert deterministic_value(log, "t-1", "clock", lambda: 100.0) == 100.0


def test_deterministic_value_replays_instead_of_reproducing():
    """Workflow.now(): часы спрашивают один раз, дальше берут из журнала."""
    log = []
    deterministic_value(log, "t-1", "clock", lambda: 100.0)
    assert deterministic_value(log, "t-1", "clock", lambda: 999.0) == 100.0


def test_deterministic_value_is_scoped_by_thread_id():
    log = []
    deterministic_value(log, "t-1", "clock", lambda: 100.0)
    assert deterministic_value(log, "t-2", "clock", lambda: 999.0) == 999.0


# ------------------------------------------------------------ run_workflow
def test_run_workflow_chains_activity_outputs():
    activities, calls = pipeline()
    assert run_workflow([], "t-1", "hello", activities) == "report://summary(15)"
    assert [name for name, _ in calls] == ["fetch", "llm", "write"]


def test_run_workflow_crash_raises_workflow_crash():
    activities, _ = pipeline()
    with pytest.raises(WorkflowCrash):
        run_workflow([], "t-1", "hello", activities, crash_after=2)


def test_replay_after_crash_reaches_the_same_state():
    """Главное свойство: после сбоя восстанавливается ровно то же состояние."""
    crashed_log, activities = [], pipeline()[0]
    with pytest.raises(WorkflowCrash):
        run_workflow(crashed_log, "t-1", "hello", activities, crash_after=2)
    resumed = run_workflow(crashed_log, "t-1", "hello", activities)

    clean_log = []
    clean = run_workflow(clean_log, "t-9", "hello", pipeline()[0])

    assert resumed == clean
    assert replay_state(crashed_log, "t-1") == replay_state(clean_log, "t-9")


def test_replay_after_crash_does_not_repeat_side_effects():
    """Три активности — три побочных эффекта, сколько бы раз ни падали."""
    log = []
    activities, calls = pipeline()
    with pytest.raises(WorkflowCrash):
        run_workflow(log, "t-1", "hello", activities, crash_after=2)
    assert len(calls) == 2
    run_workflow(log, "t-1", "hello", activities)
    assert len(calls) == 3
    run_workflow(log, "t-1", "hello", activities)
    assert len(calls) == 3
    assert execution_count(log, "t-1") == 3


def test_crash_point_determines_how_much_is_replayed():
    early, late = [], []
    for log, crash in ((early, 1), (late, 2)):
        activities, _ = pipeline()
        with pytest.raises(WorkflowCrash):
            run_workflow(log, "t-1", "hello", activities, crash_after=crash)
    assert execution_count(early, "t-1") == 1
    assert execution_count(late, "t-1") == 2


# --------------------------------------------------------- execution_count
def test_execution_count_is_zero_on_empty_log():
    assert execution_count([]) == 0


def test_execution_count_ignores_replays():
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2)
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2)
    assert execution_count(log) == 1


def test_execution_count_can_be_scoped_per_thread():
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2)
    run_activity(log, "t-2", "double", (21,), lambda x: x * 2)
    assert execution_count(log) == 2
    assert execution_count(log, "t-2") == 1


# ------------------------------------------------------------ replay_state
def test_replay_state_maps_names_to_results():
    log = []
    run_workflow(log, "t-1", "hello", pipeline()[0])
    assert replay_state(log, "t-1") == {
        "fetch": 15,
        "llm": "summary(15)",
        "write": "report://summary(15)",
    }


def test_replay_state_skips_unfinished_activities():
    log, activities = [], pipeline()[0]
    with pytest.raises(WorkflowCrash):
        run_workflow(log, "t-1", "hello", activities, crash_after=1)
    assert replay_state(log, "t-1") == {"fetch": 15}


def test_replay_state_is_latest_wins():
    log = [
        done_event("t-1", "inc", (1,), 2),
        done_event("t-1", "inc", (2,), 3),
    ]
    assert replay_state(log, "t-1") == {"inc": 3}
    assert replay_state(log, "t-2") == {}


# ----------------------------------------------------- needs_fresh_approval
def test_fresh_checkpoint_needs_no_new_approval():
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2, now=100.0)
    assert needs_fresh_approval(log, "t-1", now=101.0, max_idle=60.0) is False


def test_stale_checkpoint_requires_fresh_approval():
    """Прогон, простоявший дольше max_idle, снова идёт к человеку."""
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2, now=100.0)
    assert needs_fresh_approval(log, "t-1", now=100_000.0, max_idle=60.0) is True


def test_unknown_thread_requires_approval():
    log = []
    run_activity(log, "t-1", "double", (21,), lambda x: x * 2, now=100.0)
    assert needs_fresh_approval(log, "t-2", now=101.0, max_idle=60.0) is True
    assert needs_fresh_approval([], "t-1", now=0.0, max_idle=60.0) is True
