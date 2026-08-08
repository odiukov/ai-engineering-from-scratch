"""Тесты к уроку «Продакшен-масштабирование: очереди и чекпоинты». Правь exercise.py."""

import pytest

from exercise import (
    InvalidTransition,
    TransactionalEffectSink,
    WorkerCrash,
    append_checkpoint,
    claim_task,
    dedup_effect,
    last_checkpoint,
    process_queue,
    queue_transition,
    resume_until_done,
    run_thread,
)


def add(n):
    """Супершаг: прибавить n к счётчику. Чистая функция state -> state."""
    return lambda s: {"n": s["n"] + n}


STEPS = [add(1), add(10), add(100), add(1000)]


def make_tasks(*thread_ids):
    return [
        {"thread_id": t, "state": {"n": 0}, "worker": None, "leased_at": None,
         "done": False}
        for t in thread_ids
    ]


# ------------------------------------------------------- append_checkpoint
def test_append_checkpoint_adds_one_record():
    log = []
    record = append_checkpoint(log, "t-1", 0, {"n": 1})
    assert log == [record]
    assert record == {"thread_id": "t-1", "step": 0, "state": {"n": 1}}


def test_append_checkpoint_copies_the_state():
    """Ловушка: без копии следующий шаг задним числом испортит чекпоинт."""
    log, state = [], {"n": 1}
    append_checkpoint(log, "t-1", 0, state)
    state["n"] = 999
    assert log[0]["state"] == {"n": 1}


def test_append_checkpoint_deep_copies_nested_state():
    """dict(state) недостаточно: вложенные объекты тоже не должны alias'иться."""
    log, state = [], {"plan": {"steps": ["search"]}}
    append_checkpoint(log, "t-1", 0, state)
    state["plan"]["steps"].append("tampered")
    assert log[0]["state"] == {"plan": {"steps": ["search"]}}


def test_append_checkpoint_keeps_the_log_append_only():
    log = []
    append_checkpoint(log, "t-1", 0, {"n": 1})
    append_checkpoint(log, "t-1", 1, {"n": 2})
    assert [r["step"] for r in log] == [0, 1]


# --------------------------------------------------------- last_checkpoint
def test_last_checkpoint_of_an_empty_log_is_none():
    assert last_checkpoint([], "t-1") is None


def test_last_checkpoint_returns_the_highest_step():
    log = []
    append_checkpoint(log, "t-1", 0, {"n": 1})
    append_checkpoint(log, "t-1", 1, {"n": 2})
    assert last_checkpoint(log, "t-1")["state"] == {"n": 2}


def test_last_checkpoint_does_not_mix_threads():
    """Журнал общий: записи разных тредов лежат вперемешку."""
    log = []
    append_checkpoint(log, "t-1", 0, {"n": 1})
    append_checkpoint(log, "t-2", 0, {"n": 50})
    append_checkpoint(log, "t-1", 1, {"n": 2})
    assert last_checkpoint(log, "t-2")["state"] == {"n": 50}


def test_last_checkpoint_ignores_insertion_order():
    """«Последний» — по номеру шага, а не по позиции в списке."""
    log = []
    append_checkpoint(log, "t-1", 3, {"n": 30})
    append_checkpoint(log, "t-1", 1, {"n": 10})
    assert last_checkpoint(log, "t-1")["step"] == 3


# -------------------------------------------------------------- run_thread
def test_run_thread_executes_every_step_from_scratch():
    assert run_thread(STEPS, "t-1", [], {"n": 0}) == {"n": 1111}


def test_run_thread_writes_one_checkpoint_per_step():
    log = []
    run_thread(STEPS, "t-1", log, {"n": 0})
    assert [r["step"] for r in log] == [0, 1, 2, 3]


def test_run_thread_raises_worker_crash_not_a_generic_error():
    """Свой тип исключения: RuntimeError поймал бы и пустую заготовку."""
    with pytest.raises(WorkerCrash):
        run_thread(STEPS, "t-1", [], {"n": 0}, crash_at=1)


def test_crash_happens_before_the_step_so_nothing_half_runs():
    log = []
    with pytest.raises(WorkerCrash):
        run_thread(STEPS, "t-1", log, {"n": 0}, crash_at=2)
    assert [r["step"] for r in log] == [0, 1]
    assert last_checkpoint(log, "t-1")["state"] == {"n": 11}


def test_run_thread_resumes_from_the_last_checkpoint():
    """Возобновление начинается со СЛЕДУЮЩЕГО шага, а не с записанного."""
    log = []
    with pytest.raises(WorkerCrash):
        run_thread(STEPS, "t-1", log, {"n": 0}, crash_at=2)
    assert run_thread(STEPS, "t-1", log, {"n": 0}) == {"n": 1111}


def test_run_thread_on_a_finished_thread_is_a_no_op():
    log = []
    run_thread(STEPS, "t-1", log, {"n": 0})
    before = len(log)
    assert run_thread(STEPS, "t-1", log, {"n": 0}) == {"n": 1111}
    assert len(log) == before


def test_resume_does_not_mutate_a_nested_checkpoint():
    def mutate(state):
        state["nested"]["items"].append("next")
        return state

    log = []
    append_checkpoint(log, "t", 0, {"nested": {"items": ["first"]}})
    run_thread([lambda state: state, mutate], "t", log,
               {"nested": {"items": []}})
    assert log[0]["state"] == {"nested": {"items": ["first"]}}


# -------------------------------------------------------- resume_until_done
def test_resume_until_done_survives_a_crash():
    assert resume_until_done(STEPS, "t-1", [], {"n": 0}, [2]) == {"n": 1111}


def test_resume_from_a_checkpoint_matches_a_clean_run():
    """Главное свойство durable execution — результат от падений не зависит."""
    clean = resume_until_done(STEPS, "t-1", [], {"n": 0})
    crashy = resume_until_done(STEPS, "t-1", [], {"n": 0}, [1, 2, 3])
    assert clean == crashy


def test_resume_does_not_duplicate_checkpoints():
    """Если шаг переигрывается, в журнале появится лишняя запись — это баг."""
    clean, crashy = [], []
    resume_until_done(STEPS, "t-1", clean, {"n": 0})
    resume_until_done(STEPS, "t-1", crashy, {"n": 0}, [1, 2, 3])
    assert [r["step"] for r in crashy] == [r["step"] for r in clean]


def test_resume_gives_up_after_max_attempts():
    with pytest.raises(WorkerCrash):
        resume_until_done(STEPS, "t-1", [], {"n": 0}, [0] * 5, max_attempts=3)


# -------------------------------------------------------- queue_transition
def test_queue_transition_takes_a_task():
    assert queue_transition("idle", "take") == "processing"


def test_queue_transition_finishes_and_flushes():
    assert queue_transition("processing", "finish") == "response"
    assert queue_transition("response", "flush") == "idle"


def test_queue_transition_cycles_back_to_idle():
    state = "idle"
    for event in ("take", "finish", "flush"):
        state = queue_transition(state, event)
    assert state == "idle"


def test_queue_transition_rejects_an_impossible_event():
    """Молчаливое «остаться как было» — это state drift из MAST."""
    with pytest.raises(InvalidTransition):
        queue_transition("idle", "finish")


# --------------------------------------------------------------- claim_task
def test_claim_task_leases_a_free_task():
    tasks = make_tasks("t-1")
    task = claim_task(tasks, "w1", 0, 5)
    assert task["worker"] == "w1"
    assert task["leased_at"] == 0


def test_claim_task_skips_a_task_under_active_lease():
    tasks = make_tasks("t-1")
    claim_task(tasks, "w1", 0, 5)
    assert claim_task(tasks, "w2", 1, 5) is None


def test_claim_task_reclaims_after_the_lease_expires():
    """Воркер умер, ничего не вернув — через ttl задачу забирает другой."""
    tasks = make_tasks("t-1")
    claim_task(tasks, "w1", 0, 5)
    assert claim_task(tasks, "w2", 5, 5)["worker"] == "w2"


def test_claim_task_skips_finished_tasks():
    tasks = make_tasks("t-1", "t-2")
    tasks[0]["done"] = True
    assert claim_task(tasks, "w1", 0, 5)["thread_id"] == "t-2"


def test_claim_task_returns_none_when_everything_is_done():
    tasks = make_tasks("t-1")
    tasks[0]["done"] = True
    assert claim_task(tasks, "w1", 0, 5) is None


# ------------------------------------------------------------- dedup_effect
def test_dedup_effect_runs_the_first_time():
    sink = TransactionalEffectSink()
    assert dedup_effect(sink, "pay-1", {"amount": 10}) is True
    assert sink.effects() == [{"amount": 10}]


def test_dedup_effect_swallows_the_replay():
    """Повтор супершага после падения не должен списать деньги дважды."""
    sink = TransactionalEffectSink()
    dedup_effect(sink, "pay-1", {"amount": 10})
    assert dedup_effect(sink, "pay-1", {"amount": 10}) is False
    assert len(sink.effects()) == 1


def test_dedup_effect_distinguishes_different_keys():
    sink = TransactionalEffectSink()
    dedup_effect(sink, "pay-1", {"amount": 10})
    dedup_effect(sink, "pay-2", {"amount": 10})
    assert len(sink.effects()) == 2


def test_crash_after_atomic_commit_does_not_duplicate_the_effect():
    """Явно закрываем окно «эффект случился, dedupe ещё не записан»."""
    sink = TransactionalEffectSink()
    with pytest.raises(WorkerCrash):
        dedup_effect(sink, "pay-1", {"amount": 10}, crash_after_commit=True)
    assert dedup_effect(sink, "pay-1", {"amount": 10}) is False
    assert sink.effects() == [{"amount": 10}]


# ------------------------------------------------------------ process_queue
def test_process_queue_finishes_every_thread():
    tasks = make_tasks("t-1", "t-2", "t-3")
    assert process_queue(tasks, STEPS, []) == {
        "t-1": {"n": 1111}, "t-2": {"n": 1111}, "t-3": {"n": 1111}
    }


def test_process_queue_marks_tasks_done():
    tasks = make_tasks("t-1", "t-2")
    process_queue(tasks, STEPS, [])
    assert all(t["done"] for t in tasks)


def test_crashes_do_not_change_the_result_of_the_queue():
    """Восстановление из чекпоинта даёт ровно то же, что прогон без падений."""
    clean = process_queue(make_tasks("t-1", "t-2"), STEPS, [])
    crashy = process_queue(
        make_tasks("t-1", "t-2"), STEPS, [], {"t-1": [2], "t-2": [1, 3]}
    )
    assert clean == crashy


def test_crashes_do_not_inflate_the_checkpoint_log():
    """Журнал одинаковой длины — значит ни один супершаг не переигран."""
    clean_log, crashy_log = [], []
    process_queue(make_tasks("t-1", "t-2"), STEPS, clean_log)
    process_queue(make_tasks("t-1", "t-2"), STEPS, crashy_log,
                  {"t-1": [2], "t-2": [1, 3]})
    assert len(crashy_log) == len(clean_log)


def test_process_queue_does_not_consume_the_crash_plan():
    """План копируется: иначе второй прогон пройдёт без падений и ничего не докажет."""
    plan = {"t-1": [2]}
    process_queue(make_tasks("t-1"), STEPS, [], plan)
    assert plan == {"t-1": [2]}


def test_process_queue_gives_up_on_a_thread_that_always_crashes():
    with pytest.raises(WorkerCrash):
        process_queue(make_tasks("t-1"), STEPS, [], {"t-1": [0] * 20},
                      max_rounds=5)
