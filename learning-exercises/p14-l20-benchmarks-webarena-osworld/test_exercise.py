"""Тесты к уроку «Бенчмарки: WebArena и OSWorld». Правь exercise.py."""

import random

import pytest

from exercise import (
    STEP_GROUNDING,
    STEP_PLANNING,
    STEP_SUCCESS,
    apply_action,
    benchmark_report,
    classify_step,
    failure_breakdown,
    new_state,
    run_trajectory,
    task_succeeded,
    trajectory_efficiency,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def step(intended, clicked, plan_ok=True):
    return {"intended": intended, "clicked": clicked, "plan_ok": plan_ok}


# --------------------------------------------------------------- new_state
def test_new_state_has_an_empty_cart_and_no_orders():
    assert new_state() == {"cart": {}, "orders": []}


def test_new_state_gives_a_fresh_object_every_call():
    """Общий словарь-константа утёк бы из одного прогона в другой."""
    first = new_state()
    first["cart"]["sku-001"] = 1
    assert new_state()["cart"] == {}


def test_new_state_is_still_empty_after_a_full_purchase():
    run_trajectory([("add_to_cart", "sku-001"), ("checkout",)])
    assert new_state() == {"cart": {}, "orders": []}


# ------------------------------------------------------------ apply_action
def test_apply_action_adds_the_sku_to_the_cart():
    state, obs = apply_action(new_state(), ("add_to_cart", "sku-003"))
    assert state["cart"] == {"sku-003": 1}
    assert obs == "added sku-003"


def test_apply_action_rejects_an_unknown_sku_without_changing_the_world():
    state, obs = apply_action(new_state(), ("add_to_cart", "sku-999"))
    assert obs.startswith("error")
    assert state == new_state()


def test_apply_action_checkout_sums_the_catalog_prices():
    state, _ = run_trajectory([("add_to_cart", "sku-001"),
                               ("add_to_cart", "sku-003")])
    state, oid = apply_action(state, ("checkout",))
    assert oid == "ord-001"
    assert state["orders"][0]["total"] == 199 + 59
    assert state["cart"] == {}


def test_apply_action_checkout_of_an_empty_cart_is_an_error():
    state, obs = apply_action(new_state(), ("checkout",))
    assert obs.startswith("error")
    assert state["orders"] == []


def test_apply_action_does_not_mutate_the_state_it_was_given():
    """Harness гоняет одну стартовую точку по нескольким агентам."""
    before = new_state()
    apply_action(before, ("add_to_cart", "sku-002"))
    assert before == {"cart": {}, "orders": []}


def test_apply_action_refuses_an_unknown_action_kind():
    with pytest.raises(ValueError):
        apply_action(new_state(), ("teleport", "sku-001"))


# ---------------------------------------------------------- run_trajectory
def test_run_trajectory_returns_one_observation_per_action():
    _, obs = run_trajectory([("list_items",), ("add_to_cart", "sku-001"),
                             ("checkout",)])
    assert len(obs) == 3


def test_run_trajectory_numbers_orders_in_sequence():
    state, obs = run_trajectory([
        ("add_to_cart", "sku-001"), ("checkout",),
        ("add_to_cart", "sku-002"), ("checkout",),
    ])
    assert [o["oid"] for o in state["orders"]] == ["ord-001", "ord-002"]
    assert obs[-1] == "ord-002"


def test_run_trajectory_lets_the_agent_undo_a_wrong_pick():
    state, _ = run_trajectory([
        ("add_to_cart", "sku-002"), ("remove_from_cart", "sku-002"),
        ("add_to_cart", "sku-003"), ("checkout",),
    ])
    assert state["orders"][0]["items"] == {"sku-003": 1}


def test_run_trajectory_does_not_mutate_the_state_it_was_given():
    start = new_state()
    run_trajectory([("add_to_cart", "sku-001")], start)
    assert start == {"cart": {}, "orders": []}


# ---------------------------------------------------------- task_succeeded
def test_task_succeeded_on_the_exact_order():
    state, _ = run_trajectory([("add_to_cart", "sku-001"), ("checkout",)])
    assert task_succeeded(state, {"sku-001": 1}) is True


def test_task_succeeded_is_false_when_an_extra_item_slipped_in():
    """Клавиатура «на всякий случай» — это провал, а не «почти»."""
    state, _ = run_trajectory([("add_to_cart", "sku-001"),
                               ("add_to_cart", "sku-002"), ("checkout",)])
    assert task_succeeded(state, {"sku-001": 1}) is False


def test_task_succeeded_is_false_when_nothing_was_ordered():
    assert task_succeeded(new_state(), {"sku-001": 1}) is False


def test_task_succeeded_ignores_the_cart_and_looks_only_at_placed_orders():
    """WebArena смотрит на состояние приложения: корзина — это ещё не покупка."""
    state, _ = run_trajectory([("add_to_cart", "sku-001")])
    assert task_succeeded(state, {"sku-001": 1}) is False


# --------------------------------------------------- trajectory_efficiency
def test_trajectory_efficiency_is_steps_over_gold():
    assert trajectory_efficiency(6, 3) == APPROX(2.0)


def test_trajectory_efficiency_of_a_perfect_run_is_one():
    assert trajectory_efficiency(3, 3) == APPROX(1.0)


def test_trajectory_efficiency_refuses_a_zero_length_gold():
    with pytest.raises(ValueError):
        trajectory_efficiency(5, 0)


def test_trajectory_efficiency_refuses_negative_steps():
    with pytest.raises(ValueError):
        trajectory_efficiency(-1, 3)


# ------------------------------------------------------------ classify_step
def test_classify_step_marks_a_correct_click_as_success():
    assert classify_step(step("buy", "buy")) == STEP_SUCCESS


def test_classify_step_marks_a_missed_click_as_grounding():
    assert classify_step(step("buy", "cart")) == STEP_GROUNDING


def test_classify_step_marks_a_click_that_hit_nothing_as_grounding():
    assert classify_step(step("buy", None)) == STEP_GROUNDING


def test_classify_step_blames_planning_even_when_the_click_landed():
    """Метко кликнуть не в ту кнопку — провал плана, а не grounding."""
    assert classify_step(step("cart", "cart", plan_ok=False)) == STEP_PLANNING


# ------------------------------------------------------- failure_breakdown
def test_failure_breakdown_counts_every_class():
    records = [step("a", "a"), step("a", "b"), step("a", "a", plan_ok=False)]
    assert failure_breakdown(records) == {
        STEP_SUCCESS: 1, STEP_GROUNDING: 1, STEP_PLANNING: 1,
    }


def test_failure_breakdown_keeps_zero_categories_visible():
    """Пропавшая категория читается как «такого не бывает» — это ложь."""
    assert failure_breakdown([step("a", "a")]) == {
        STEP_SUCCESS: 1, STEP_GROUNDING: 0, STEP_PLANNING: 0,
    }


def test_failure_breakdown_does_not_depend_on_record_order():
    rng = random.Random(5)
    records = [step("a", "a"), step("a", "b"), step("a", "a", plan_ok=False)] * 4
    shuffled = list(records)
    rng.shuffle(shuffled)
    assert failure_breakdown(shuffled) == failure_breakdown(records)


# -------------------------------------------------------- benchmark_report
def test_benchmark_report_counts_the_success_rate():
    results = [
        {"task_id": "t1", "success": True, "steps": 3, "gold_steps": 3},
        {"task_id": "t2", "success": False, "steps": 9, "gold_steps": 3},
    ]
    report = benchmark_report(results)
    assert (report["tasks"], report["solved"]) == (2, 1)
    assert report["success_rate"] == APPROX(0.5)


def test_benchmark_report_measures_efficiency_only_on_solved_tasks():
    """Агент, сдавшийся на первом шаге, иначе выглядел бы «эффективнее» человека."""
    results = [
        {"task_id": "t1", "success": True, "steps": 6, "gold_steps": 3},
        {"task_id": "t2", "success": False, "steps": 1, "gold_steps": 3},
    ]
    assert benchmark_report(results)["efficiency"] == APPROX(2.0)


def test_benchmark_report_efficiency_is_zero_when_nothing_was_solved():
    results = [{"task_id": "t1", "success": False, "steps": 4, "gold_steps": 2}]
    assert benchmark_report(results)["efficiency"] == APPROX(0.0)


def test_benchmark_report_does_not_depend_on_task_order():
    rng = random.Random(11)
    results = [
        {"task_id": f"t{i}", "success": i % 2 == 0, "steps": 2 + i, "gold_steps": 3}
        for i in range(12)
    ]
    shuffled = list(results)
    rng.shuffle(shuffled)
    assert benchmark_report(shuffled) == benchmark_report(results)


def test_benchmark_report_refuses_a_duplicated_task_id():
    results = [
        {"task_id": "t1", "success": True, "steps": 3, "gold_steps": 3},
        {"task_id": "t1", "success": False, "steps": 3, "gold_steps": 3},
    ]
    with pytest.raises(ValueError):
        benchmark_report(results)
