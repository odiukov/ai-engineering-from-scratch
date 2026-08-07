"""Тесты к уроку «Паттерны оркестрации». Правь exercise.py."""

import random

import pytest

from exercise import (
    PATTERNS,
    classify,
    compare_patterns,
    detect_bouncing,
    hierarchical_route,
    pick_pattern,
    run_parallel,
    supervisor_route,
    swarm_route,
)

REFUND = "I want a refund for order 4711"
BUG = "the app keeps crashing on save"
SALES = "what is the price of the team plan"


# ---------------------------------------------------------------- classify
def test_classify_finds_the_bug_intent():
    assert classify(BUG) == "bug"


def test_classify_falls_back_to_unknown():
    assert classify("добрый день") == "unknown"


def test_classify_breaks_ties_alphabetically():
    """Роутер, зависящий от порядка ключей словаря, ломает воспроизводимость трейсов."""
    assert classify("refund and bug") == "bug"


def test_classify_ignores_letter_case():
    assert classify("REFUND PLEASE") == classify("refund please") == "refund"


# --------------------------------------------------------- supervisor_route
def test_supervisor_sends_each_task_to_its_specialist():
    assert supervisor_route([REFUND, BUG, SALES])["assignments"] == [
        "billing_agent",
        "support_agent",
        "sales_agent",
    ]


def test_supervisor_costs_two_ops_per_routed_task():
    assert supervisor_route([REFUND, BUG])["ops"] == 4


def test_unroutable_task_costs_only_the_router_hop():
    result = supervisor_route(["добрый день"])
    assert (result["assignments"], result["ops"]) == ([None], 1)


# -------------------------------------------------------------- swarm_route
def test_swarm_delivers_without_a_router():
    assert swarm_route([REFUND, BUG, SALES])["assignments"] == [
        "billing_agent",
        "support_agent",
        "sales_agent",
    ]


def test_swarm_reaches_the_entry_specialist_in_one_hop():
    """Иногда swarm дешевле супервизора — попал сразу, роутер не нужен."""
    assert swarm_route([REFUND])["handoffs"] == [["billing_agent"]]


def test_swarm_walks_the_ring_to_a_distant_specialist():
    assert swarm_route([SALES])["handoffs"] == [
        ["billing_agent", "support_agent", "sales_agent"]
    ]


def test_swarm_gives_up_after_max_hops():
    """Без счётчика передач нераспознанная задача крутится по кольцу вечно."""
    result = swarm_route(["добрый день"], max_hops=2)
    assert result["assignments"] == [None]
    assert len(result["handoffs"][0]) == 3


def test_entry_agent_outside_the_ring_is_value_error():
    with pytest.raises(ValueError):
        swarm_route([REFUND], entry="legal_agent")


# ---------------------------------------------------------- detect_bouncing
def test_return_to_the_previous_agent_is_bouncing():
    assert detect_bouncing(["billing", "support", "billing"]) is True


def test_forward_only_chain_is_not_bouncing():
    assert detect_bouncing(["billing", "support", "sales"]) is False


def test_same_agent_twice_in_a_row_is_not_bouncing():
    """Агент, продолжающий работу, — это не пинг-понг."""
    assert detect_bouncing(["billing", "billing", "billing"]) is False


# ------------------------------------------------------- hierarchical_route
def test_hierarchy_reaches_the_same_specialist():
    assert hierarchical_route([REFUND, BUG])["assignments"] == [
        "billing_agent",
        "support_agent",
    ]


def test_hierarchy_costs_one_extra_hop_per_task():
    tasks = [REFUND, BUG]
    assert hierarchical_route(tasks)["ops"] == supervisor_route(tasks)["ops"] + 2


def test_hierarchy_names_the_team():
    assert hierarchical_route([REFUND, SALES])["teams"] == ["finance", "product"]


# --------------------------------------------------------------- run_parallel
def test_parallel_results_follow_task_order():
    assert run_parallel(["a", "b", "c"], str.upper) == ["A", "B", "C"]


def test_completion_order_does_not_change_the_result():
    """Кто закончил раньше — дело случая. Ответ обязан быть одним и тем же."""
    rng = random.Random(3)
    tasks = ["a", "b", "c", "d", "e"]
    reference = run_parallel(tasks, str.upper)
    for _ in range(20):
        order = list(range(len(tasks)))
        rng.shuffle(order)
        assert run_parallel(tasks, str.upper, completion_order=order) == reference


def test_worker_runs_once_per_task():
    calls = []

    def worker(task):
        calls.append(task)
        return task

    run_parallel(["a", "b", "c"], worker, completion_order=[2, 0, 1])
    assert sorted(calls) == ["a", "b", "c"]


def test_completion_order_that_is_not_a_permutation_is_value_error():
    with pytest.raises(ValueError):
        run_parallel(["a", "b"], str.upper, completion_order=[0, 0])


def test_empty_task_list_gives_empty_results():
    assert run_parallel([], str.upper) == []


# --------------------------------------------------------------- pick_pattern
def test_single_specialist_needs_no_topology():
    assert pick_pattern(1) == "single_agent"


def test_default_choice_is_supervisor():
    assert pick_pattern(3) == "supervisor"


def test_context_budget_failure_forces_hierarchical():
    """Жёсткое ограничение бьёт любые предпочтения."""
    assert pick_pattern(12, accuracy_critical=True, supervisor_context_ok=False) == (
        "hierarchical"
    )


def test_accuracy_outranks_latency_in_the_decision_order():
    assert pick_pattern(3, latency_critical=True, accuracy_critical=True) == "debate"


def test_latency_critical_picks_swarm():
    assert pick_pattern(4, latency_critical=True) == "swarm"


def test_zero_specialists_is_value_error():
    with pytest.raises(ValueError):
        pick_pattern(0)


# ------------------------------------------------------------ compare_patterns
def test_all_patterns_reach_the_same_assignments():
    """Маршрут не должен зависеть от топологии — зависеть должна только цена."""
    report = compare_patterns([REFUND, BUG, SALES, "добрый день"])
    assignments = [entry["assignments"] for entry in report.values()]
    assert all(a == assignments[0] for a in assignments)


def test_hierarchical_is_the_most_expensive():
    report = compare_patterns([REFUND, BUG])
    assert report["hierarchical"]["ops"] > report["supervisor"]["ops"]


def test_swarm_can_beat_supervisor_on_cost():
    report = compare_patterns([REFUND, REFUND, BUG])
    assert report["swarm"]["ops"] < report["supervisor"]["ops"]


def test_report_covers_every_pattern():
    assert sorted(compare_patterns([REFUND])) == sorted(PATTERNS)
