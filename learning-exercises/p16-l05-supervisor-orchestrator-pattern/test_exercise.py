"""Тесты к уроку «Паттерн supervisor / orchestrator-worker». Правь exercise.py."""

import pytest

from exercise import (
    MAX_WORKERS,
    SIMPLE_TOOL_CALLS,
    SupervisorError,
    detect_conflicts,
    parallel_seconds,
    plan,
    run_workers,
    scale_effort,
    sequential_seconds,
    supervisor_run,
    synthesize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

QUERY = "what changed in multi-agent systems?"


def fast_worker(sub_question):
    return {"answer": f"summary of {sub_question}", "seconds": 0.3, "tokens": 800}


def tiny_worker(sub_question):
    return {"answer": "ok", "seconds": 0.01}


def slow_worker(sub_question):
    return {"answer": "ok", "seconds": 1.0}


def half_broken_worker(sub_question):
    if "open problems" in sub_question:
        raise TimeoutError("worker timeout")
    return {"answer": "ok", "seconds": 0.3}


def disputing_worker(sub_question):
    verdict = "yes" if "historical origins" in sub_question else "no"
    return {"answer": "ok", "seconds": 0.1, "claims": {"llm-managers-work": verdict}}


# ------------------------------------------------------------- scale_effort
def test_simple_query_gets_a_single_agent():
    assert scale_effort(3) == 1


def test_the_simple_threshold_is_inclusive():
    assert scale_effort(SIMPLE_TOOL_CALLS) == 1


def test_one_call_over_the_threshold_adds_a_worker():
    assert scale_effort(SIMPLE_TOOL_CALLS + 1) == 2


def test_worker_count_is_capped():
    """Каждый воркер стоит контекста, множить их бесконечно нельзя."""
    assert scale_effort(99_999) == MAX_WORKERS


def test_worker_count_never_decreases_with_complexity():
    counts = [scale_effort(n) for n in range(1, 200)]
    assert all(b >= a for a, b in zip(counts, counts[1:]))


# --------------------------------------------------------------------- plan
def test_single_worker_plan_keeps_the_query_unchanged():
    """Вырожденный случай: supervisor с одним исполнителем — обычный агент."""
    assert plan(QUERY, 1) == [QUERY]


def test_plan_makes_one_sub_question_per_worker():
    assert len(plan(QUERY, 3)) == 3


def test_sub_questions_are_all_different():
    assert len(set(plan(QUERY, 5))) == 5


def test_every_sub_question_still_carries_the_query():
    assert all(QUERY in sq for sq in plan(QUERY, 4))


def test_planning_for_zero_workers_is_refused():
    with pytest.raises(SupervisorError):
        plan(QUERY, 0)


def test_planning_beyond_the_aspect_list_is_refused():
    with pytest.raises(SupervisorError):
        plan(QUERY, 999)


# -------------------------------------------------------------- run_workers
def test_each_result_remembers_its_sub_question():
    results = run_workers(["a", "b"], fast_worker)
    assert [r["sub_question"] for r in results] == ["a", "b"]


def test_results_keep_the_order_of_the_plan():
    """Параллельный запуск не даёт права перепутать, кто что ответил."""
    results = run_workers(["z", "y", "x"], fast_worker)
    assert [r["answer"] for r in results] == [
        "summary of z", "summary of y", "summary of x"
    ]


def test_a_killed_worker_reports_no_answer():
    results = run_workers(["q -- open problems"], half_broken_worker)
    assert results[0]["answer"] is None
    assert "worker timeout" in results[0]["error"]


def test_one_dead_worker_does_not_stop_the_others():
    results = run_workers(["q -- open problems", "q -- state of the art"],
                          half_broken_worker)
    assert results[1]["answer"] == "ok"


# ------------------------------------------------- sequential / parallel time
def test_sequential_time_adds_every_worker():
    assert sequential_seconds([0.3, 0.3, 0.3], 0.05, 0.05) == APPROX(1.0)


def test_parallel_time_pays_only_for_the_slowest_worker():
    assert parallel_seconds([0.3, 0.3, 0.3], 0.05, 0.05, 0.02) == APPROX(0.46)


def test_parallelism_wins_on_long_workers():
    seconds = [1.0] * 5
    assert parallel_seconds(seconds, 0.05, 0.05, 0.02) < \
        sequential_seconds(seconds, 0.05, 0.05)


def test_coordination_beats_parallelism_on_tiny_workers():
    """Главная оговорка урока: на мелких подзадачах supervisor проигрывает."""
    seconds = [0.01] * 5
    assert parallel_seconds(seconds, 0.05, 0.05, 0.02) > \
        sequential_seconds(seconds, 0.05, 0.05)


def test_lead_costs_something_even_without_workers():
    assert parallel_seconds([], 0.05, 0.05, 0.02) == APPROX(0.1)


# ---------------------------------------------------------- detect_conflicts
def test_agreeing_workers_produce_no_conflict():
    results = [{"claims": {"x": "yes"}}, {"claims": {"x": "yes"}}]
    assert detect_conflicts(results) == {}


def test_disagreement_lists_both_verdicts():
    results = [{"claims": {"x": "yes"}}, {"claims": {"x": "no"}}]
    assert detect_conflicts(results) == {"x": ["no", "yes"]}


def test_results_without_claims_are_fine():
    assert detect_conflicts([{"answer": "a"}, {"answer": "b"}]) == {}


def test_three_way_disagreement_keeps_every_verdict():
    results = [{"claims": {"x": "a"}}, {"claims": {"x": "b"}}, {"claims": {"x": "c"}}]
    assert detect_conflicts(results)["x"] == ["a", "b", "c"]


def test_untouched_claims_are_not_conflicts():
    results = [{"claims": {"x": "yes", "y": "up"}}, {"claims": {"x": "yes"}}]
    assert detect_conflicts(results) == {}


# --------------------------------------------------------------- synthesize
def test_synthesis_lists_every_sub_answer():
    results = [{"sub_question": "q1", "answer": "a1"},
               {"sub_question": "q2", "answer": "a2"}]
    text = synthesize(QUERY, results, {})
    assert "- q1: a1" in text and "- q2: a2" in text


def test_synthesis_marks_a_missing_worker():
    results = [{"sub_question": "q1", "answer": None, "error": "worker timeout"}]
    assert "MISSING q1: worker timeout" in synthesize(QUERY, results, {})


def test_synthesis_surfaces_disagreement_instead_of_picking_a_side():
    """Молча выбрать одну сторону — худший исход: спор исчезает бесследно."""
    text = synthesize(QUERY, [], {"x": ["no", "yes"]})
    assert "! CONFLICT on x: no vs yes" in text


def test_synthesis_stays_quiet_when_nobody_disagrees():
    results = [{"sub_question": "q1", "answer": "a1"}]
    assert "CONFLICT" not in synthesize(QUERY, results, {})


# ------------------------------------------------------------ supervisor_run
def test_simple_query_degenerates_into_a_single_agent():
    """Один воркер, исходный вопрос, никакого разбиения."""
    answer, stats = supervisor_run(QUERY, fast_worker, 3)
    assert stats["worker_count"] == 1
    assert f"- {QUERY}: summary of {QUERY}" in answer


def test_a_single_worker_only_adds_coordination_cost():
    """Параллелить нечего — остаётся чистая накладная цена порождения."""
    _, stats = supervisor_run(QUERY, fast_worker, 3, spawn_seconds=0.02)
    assert stats["coordination_cost"] == APPROX(0.02)


def test_complex_query_spawns_several_workers_and_wins_wall_clock():
    _, stats = supervisor_run(QUERY, slow_worker, 50)
    assert stats["worker_count"] == 5
    assert stats["coordination_cost"] < 0


def test_tiny_subtasks_make_the_supervisor_lose():
    _, stats = supervisor_run(QUERY, tiny_worker, 50)
    assert stats["coordination_cost"] > 0


def test_supervisor_reports_the_workers_it_lost():
    _, stats = supervisor_run(QUERY, half_broken_worker, 50)
    assert stats["failed"] == [f"{QUERY} -- open problems"]


def test_supervisor_surfaces_worker_disagreement():
    answer, stats = supervisor_run(QUERY, disputing_worker, 20)
    assert stats["conflicts"] == {"llm-managers-work": ["no", "yes"]}
    assert "! CONFLICT on llm-managers-work" in answer
