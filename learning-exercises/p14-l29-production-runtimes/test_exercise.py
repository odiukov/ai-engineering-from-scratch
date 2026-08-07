"""Тесты к уроку «Production runtimes: очередь, событие, cron». Правь exercise.py."""

import random

import pytest

from exercise import (
    apply_once,
    cron_due,
    enqueue,
    percentile,
    pick_runtime_shape,
    queue_metrics,
    resume_from_checkpoint,
    run_worker,
)

RECORDS = [
    {"id": "a", "enqueued_at": 0, "finished_at": 2, "status": "done"},
    {"id": "b", "enqueued_at": 1, "finished_at": 9, "status": "done"},
    {"id": "c", "enqueued_at": 2, "finished_at": 5, "status": "dlq"},
    {"id": "d", "enqueued_at": 3, "finished_at": None, "status": "pending"},
]


# ------------------------------------------------------- pick_runtime_shape
def test_resume_cost_is_ignored_for_a_short_task():
    """Восстанавливать нечего: двухсекундную задачу дешевле повторить целиком."""
    assert pick_runtime_shape(2) == "request_response"
    assert pick_runtime_shape(2, resume_cost_high=True) == "request_response"


def test_progressive_output_asks_for_streaming():
    assert pick_runtime_shape(5, needs_progress=True) == "streaming"


def test_long_task_leaves_the_request_path():
    """Пять минут синхронного ожидания — та самая ошибка выбора формы."""
    assert pick_runtime_shape(600) == "queue"
    assert pick_runtime_shape(600, resume_cost_high=True) == "durable"


def test_schedule_wins_over_the_other_signals():
    assert pick_runtime_shape(600, event_triggered=True, periodic=True) == "cron"
    assert pick_runtime_shape(600, event_triggered=True) == "event"


def test_negative_duration_is_value_error():
    with pytest.raises(ValueError):
        pick_runtime_shape(-1)


# ------------------------------------------------------------------ enqueue
def test_new_job_lands_in_the_queue():
    assert enqueue([], "j1", "send report") == [
        {"id": "j1", "payload": "send report", "attempt": 0}
    ]


def test_same_job_id_is_not_enqueued_twice():
    """Продюсер без подтверждения шлёт повтор — отчёт не должен уйти дважды."""
    once = enqueue([], "j1", "send report")
    assert enqueue(once, "j1", "send report") == once


def test_duplicate_does_not_overwrite_the_first_payload():
    once = enqueue([], "j1", "первый")
    assert enqueue(once, "j1", "второй")[0]["payload"] == "первый"


def test_enqueue_does_not_mutate_the_input_list():
    pending = []
    enqueue(pending, "j1", "x")
    assert pending == []


# --------------------------------------------------------------- run_worker
def test_worker_drains_the_queue_without_touching_it():
    pending = enqueue(enqueue([], "j1", "a"), "j2", "b")
    result = run_worker(pending, str.upper)
    assert result["done"] == [("j1", "A"), ("j2", "B")]
    assert result["dlq"] == []
    assert pending == [
        {"id": "j1", "payload": "a", "attempt": 0},
        {"id": "j2", "payload": "b", "attempt": 0},
    ]


def test_exhausted_job_is_retried_then_lands_in_the_dlq():
    """Очередь без DLQ теряет упавшие задачи молча."""

    def always_fails(payload):
        raise LookupError(payload)

    result = run_worker(enqueue([], "j1", "яд"), always_fails, max_attempts=3)
    assert result["dlq"] == ["j1"]
    assert result["attempts"]["j1"] == 3
    assert result["done"] == []


def test_poison_job_does_not_block_the_rest():
    """Ретрай в хвост очереди, а не на месте: иначе один payload держит всех."""
    pending = enqueue(enqueue(enqueue([], "j1", "ok1"), "j2", "яд"), "j3", "ok2")

    def handler(payload):
        if payload == "яд":
            raise LookupError(payload)
        return payload

    result = run_worker(pending, handler, max_attempts=2)
    assert result["done"] == [("j1", "ok1"), ("j3", "ok2")]
    assert result["dlq"] == ["j2"]


def test_max_attempts_below_one_is_value_error():
    with pytest.raises(ValueError):
        run_worker(enqueue([], "j1", "x"), str.upper, max_attempts=0)


def test_retry_does_not_double_an_idempotent_effect():
    """At-least-once доставка плюс идемпотентный обработчик — ровно один эффект."""
    store = {}
    tries = []

    def handler(payload):
        nonlocal store
        store = apply_once(store, payload, "charged", 1)
        tries.append(payload)
        if len(tries) < 3:                  # первые две попытки падают ПОСЛЕ записи
            raise LookupError("подтверждение не дошло")
        return "ok"

    result = run_worker(enqueue([], "j1", "j1"), handler, max_attempts=3)
    assert result["done"] == [("j1", "ok")]
    assert len(tries) == 3
    assert store["counters"]["charged"] == 1


# --------------------------------------------------------------- apply_once
def test_first_delivery_applies_the_effect():
    assert apply_once({}, "j1", "emails", 1) == {
        "counters": {"emails": 1},
        "applied": ["j1"],
    }


def test_redelivery_does_not_double_the_effect():
    once = apply_once({}, "j1", "emails", 1)
    assert apply_once(once, "j1", "emails", 1) == once


def test_different_jobs_accumulate():
    store = apply_once(apply_once({}, "j1", "emails", 1), "j2", "emails", 2)
    assert store["counters"]["emails"] == 3
    assert store["applied"] == ["j1", "j2"]


def test_apply_once_does_not_mutate_the_store():
    store = apply_once({}, "j1", "emails", 1)
    apply_once(store, "j2", "emails", 5)
    assert store == {"counters": {"emails": 1}, "applied": ["j1"]}


def test_duplicated_delivery_log_matches_the_clean_one():
    """Порядок и повторы в журнале доставок не влияют на итоговый эффект."""
    rng = random.Random(11)
    clean = [(f"j{i}", "runs", 1) for i in range(8)]
    noisy = clean + [rng.choice(clean) for _ in range(12)]
    rng.shuffle(noisy)

    clean_store, noisy_store = {}, {}
    for jid, key, delta in clean:
        clean_store = apply_once(clean_store, jid, key, delta)
    for jid, key, delta in noisy:
        noisy_store = apply_once(noisy_store, jid, key, delta)

    assert noisy_store["counters"] == clean_store["counters"] == {"runs": 8}


# --------------------------------------------------------------- percentile
def test_median_by_nearest_rank():
    assert percentile([4, 1, 3, 2], 50) == 2


def test_p95_picks_the_tail():
    assert percentile(list(range(1, 101)), 95) == 95


def test_percentile_does_not_interpolate():
    """По четырём замерам «p75 = 3.25» — выдумка. Ответ обязан быть из выборки."""
    assert percentile([1, 2, 3, 4], 75) == 3


def test_empty_sample_has_no_percentile():
    assert percentile([], 50) is None


# ------------------------------------------------------------- queue_metrics
def test_depth_and_dlq_are_counted_separately():
    m = queue_metrics(RECORDS, now=10)
    assert (m["depth"], m["dlq"]) == (1, 1)


def test_latency_percentiles_come_from_finished_jobs():
    """Незавершённая задача в задержки не входит — она ещё не задержалась."""
    m = queue_metrics(RECORDS, now=10)
    assert (m["p50"], m["p95"]) == (3, 8)


def test_pending_only_queue_has_no_latency_but_a_growing_wait():
    """Ноль вместо None соврал бы про «отвечаем мгновенно»."""
    pending = [{"id": "a", "enqueued_at": 4, "finished_at": None, "status": "pending"}]
    early = queue_metrics(pending, now=6)
    late = queue_metrics(pending, now=60)
    assert early["p50"] is None and late["p50"] is None
    assert (early["oldest_wait"], late["oldest_wait"]) == (2, 56)


def test_clock_running_backwards_is_value_error():
    with pytest.raises(ValueError):
        queue_metrics(
            [{"id": "a", "enqueued_at": 5, "finished_at": None, "status": "pending"}],
            now=1,
        )
    with pytest.raises(ValueError):
        queue_metrics(
            [{"id": "a", "enqueued_at": 5, "finished_at": 2, "status": "done"}], now=9
        )


# ------------------------------------------------------- resume_from_checkpoint
def test_fresh_run_executes_every_step():
    result = resume_from_checkpoint(["a", "b"], {}, str.upper)
    assert result == {
        "completed": ["a", "b"],
        "results": {"a": "A", "b": "B"},
        "failed": None,
    }


def test_failure_stops_at_the_broken_step():
    """Следующие шаги могут зависеть от упавшего — идти дальше нельзя."""

    def runner(name):
        if name == "b":
            raise LookupError("сеть отвалилась")
        return name.upper()

    result = resume_from_checkpoint(["a", "b", "c"], {}, runner)
    assert result["failed"] == "b"
    assert result["completed"] == ["a"]
    assert "c" not in result["results"]


def test_resume_after_a_fix_does_not_repeat_completed_steps():
    """Агент, упавший на шаге 37, продолжает с 37-го, а не платит за 36 заново."""
    calls = []

    def runner(name):
        calls.append(name)
        if name == "c" and calls.count("c") == 1:
            raise LookupError("сеть отвалилась")
        return name.upper()

    steps = ["a", "b", "c", "d"]
    first = resume_from_checkpoint(steps, {}, runner)
    second = resume_from_checkpoint(steps, first, runner)
    assert first["failed"] == "c"
    assert second["failed"] is None
    assert calls == ["a", "b", "c", "c", "d"]
    assert second["results"] == {"a": "A", "b": "B", "c": "C", "d": "D"}


def test_duplicate_step_names_are_value_error():
    with pytest.raises(ValueError):
        resume_from_checkpoint(["a", "a"], {}, str.upper)


# ------------------------------------------------------------------ cron_due
def test_never_run_jobs_are_due_and_come_back_sorted():
    """Порядок ключей словаря не должен решать, что запустится первым."""
    assert cron_due({"zebra": 10, "alpha": 10}, {}, now=0) == ["alpha", "zebra"]


def test_job_becomes_due_only_after_its_interval():
    assert cron_due({"evals": 60}, {"evals": 0}, now=30) == []
    assert cron_due({"evals": 60}, {"evals": 0}, now=60) == ["evals"]


def test_missed_ticks_do_not_pile_up():
    """Сутки простоя — один запуск, а не двадцать четыре."""
    schedule = {"evals": 60}
    last_run = {"evals": 0}
    due = cron_due(schedule, last_run, now=1440)
    assert due == ["evals"]
    after = {**last_run, **{name: 1440 for name in due}}
    assert cron_due(schedule, after, now=1441) == []


def test_non_positive_interval_is_value_error():
    with pytest.raises(ValueError):
        cron_due({"evals": 0}, {}, now=5)
