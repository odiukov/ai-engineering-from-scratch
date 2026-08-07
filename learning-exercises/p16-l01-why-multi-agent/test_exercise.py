"""Тесты к уроку «Зачем нужны мультиагенты». Правь exercise.py."""

import pytest

from exercise import (
    MAX_SINGLE_AGENT_TOOL_CALLS,
    WINDOW,
    coordination_overhead,
    fanout_seconds,
    first_overflow,
    inbox,
    multi_agent_contexts,
    pipeline_seconds,
    recommend_topology,
    single_agent_context,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

MESSAGES = [
    {"from": "researcher", "to": "coder", "content": "notes"},
    {"from": "coder", "to": "reviewer", "content": "code"},
    {"from": "reviewer", "to": "orchestrator", "content": "review"},
    {"from": "analyst", "to": "coder", "content": "requirements"},
]


# -------------------------------------------------------------------- inbox
def test_inbox_returns_only_messages_addressed_to_the_agent():
    got = inbox(MESSAGES, "coder")
    assert [m["content"] for m in got] == ["notes", "requirements"]


def test_inbox_of_an_unknown_agent_is_empty():
    assert inbox(MESSAGES, "tester") == []


def test_inbox_preserves_send_order():
    """Порядок писем несёт смысл: research приходит раньше требований."""
    got = inbox(MESSAGES, "coder")
    assert got[0]["from"] == "researcher" and got[1]["from"] == "analyst"


def test_inbox_never_leaks_another_agents_context():
    """Главное свойство разделения: чужие 50k токенов сюда не попадают."""
    for agent in ("coder", "reviewer", "orchestrator"):
        assert all(m["to"] == agent for m in inbox(MESSAGES, agent))


# ------------------------------------------------------ single_agent_context
def test_single_agent_context_accumulates_every_stage():
    assert single_agent_context(1000, [500, 800]) == [1500, 2300]


def test_single_agent_context_of_no_stages_is_empty():
    assert single_agent_context(1000, []) == []


def test_single_agent_context_never_shrinks():
    """Одиночный агент ничего не выбрасывает — размер только растёт."""
    sizes = single_agent_context(100, [10, 20, 30, 40])
    assert all(b > a for a, b in zip(sizes, sizes[1:]))


def test_single_agent_context_has_one_entry_per_stage():
    assert len(single_agent_context(1, [1, 1, 1])) == 3


# ----------------------------------------------------- multi_agent_contexts
def test_multi_agent_contexts_shows_each_specialist_separately():
    assert multi_agent_contexts(1000, [500, 800, 300]) == [1500, 2300, 2100]


def test_first_specialist_never_sees_a_predecessor():
    assert multi_agent_contexts(7000, [1, 2, 3])[0] == 7001


def test_multi_agent_contexts_match_single_agent_on_two_stages():
    """До третьего этапа выбрасывать ещё нечего — числа обязаны совпасть."""
    assert multi_agent_contexts(1000, [500, 800]) == single_agent_context(1000, [500, 800])


def test_multi_agent_contexts_stay_below_single_agent_context():
    """Ради этого урок и затевался: разделение держит контексты маленькими."""
    outputs = [4000, 5000, 6000]
    multi = multi_agent_contexts(2000, outputs)
    single = single_agent_context(2000, outputs)
    assert max(multi) < max(single)


def test_multi_agent_contexts_of_no_stages_is_empty():
    assert multi_agent_contexts(1000, []) == []


# ------------------------------------------------------------ first_overflow
def test_first_overflow_returns_minus_one_when_everything_fits():
    assert first_overflow([10, 20], 100) == -1


def test_first_overflow_finds_the_stage_that_breaks_the_window():
    assert first_overflow([60, 120, 300], 100) == 1


def test_exactly_full_window_is_not_an_overflow():
    """Ловушка: ровно окно — это ещё влезло, строгое сравнение."""
    assert first_overflow([100], 100) == -1


def test_first_overflow_uses_the_lesson_window_by_default():
    assert first_overflow([WINDOW + 1]) == 0


# ---------------------------------------------------------- pipeline_seconds
def test_pipeline_seconds_sums_stages_and_handoffs():
    assert pipeline_seconds([1.0, 2.0, 3.0], 0.1) == APPROX(6.2)


def test_single_stage_pipeline_pays_no_handoff():
    """Ловушка: границ на одну меньше, чем этапов."""
    assert pipeline_seconds([5.0], 0.1) == APPROX(5.0)


def test_empty_pipeline_takes_no_time():
    assert pipeline_seconds([], 0.1) == APPROX(0.0)


# ------------------------------------------------------------ fanout_seconds
def test_fanout_seconds_is_max_stage_plus_split_and_merge():
    assert fanout_seconds([1.0, 2.0, 3.0], 0.1) == APPROX(3.6)


def test_fanout_beats_pipeline_on_long_stages():
    stages = [3.0, 3.0, 3.0]
    assert fanout_seconds(stages, 0.1) < pipeline_seconds(stages, 0.1)


def test_coordination_beats_parallelism_on_short_stages():
    """Ключевое свойство урока: на мелких задачах мультиагент проигрывает."""
    stages = [0.1, 0.1, 0.1]
    assert fanout_seconds(stages, 0.1) > pipeline_seconds(stages, 0.1)


def test_empty_fanout_takes_no_time():
    assert fanout_seconds([], 0.1) == APPROX(0.0)


# ------------------------------------------------------ coordination_overhead
def test_coordination_overhead_is_negative_when_contexts_pile_up():
    assert coordination_overhead(1000, [500, 800, 300], 100) == APPROX(-300)


def test_coordination_overhead_is_positive_on_a_tiny_task():
    assert coordination_overhead(1000, [10, 10], 500) == APPROX(500)


def test_coordination_overhead_grows_with_summary_price():
    cheap = coordination_overhead(1000, [10, 10, 10], 50)
    pricey = coordination_overhead(1000, [10, 10, 10], 5000)
    assert pricey > cheap


def test_single_stage_has_no_boundary_to_pay_for():
    """Один этап — границ нет, пересказ не оплачивается."""
    assert coordination_overhead(1000, [500], 9999) == APPROX(0.0)


# ------------------------------------------------------- recommend_topology
def test_small_task_stays_single_agent():
    assert recommend_topology(1000, [500], 5, False) == "single"


def test_context_overflow_forces_multi_agent():
    assert recommend_topology(90_000, [50_000], 5, False) == "multi"


def test_too_many_tool_calls_forces_multi_agent():
    assert recommend_topology(1000, [500], MAX_SINGLE_AGENT_TOOL_CALLS, False) == "multi"


def test_nineteen_tool_calls_still_fit_one_agent():
    """Порог из урока: «меньше 20» — значит 19 ещё одиночный."""
    assert recommend_topology(1000, [500], 19, False) == "single"


def test_different_system_prompts_force_multi_agent():
    """Даже крошечная задача разбивается, если этапам нужны разные роли."""
    assert recommend_topology(1000, [500], 1, True) == "multi"
