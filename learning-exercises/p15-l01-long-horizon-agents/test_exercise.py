"""Тесты к уроку «Переход от чат-ботов к агентам с длинным горизонтом».

Правь exercise.py.
"""

import random

import pytest

from exercise import (
    BASELINE_HOURS,
    DOUBLING_MONTHS,
    deployment_horizon,
    end_to_end_reliability,
    horizon_at,
    horizon_verdict,
    max_steps_for_target,
    months_to_cross,
    simulate_run,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------------ horizon_at
def test_horizon_at_zero_months_is_the_baseline():
    assert horizon_at(0) == APPROX(BASELINE_HOURS)


def test_horizon_doubles_after_exactly_one_doubling_time():
    assert horizon_at(DOUBLING_MONTHS) == APPROX(2 * BASELINE_HOURS)


def test_horizon_quadruples_after_two_doubling_times():
    assert horizon_at(2 * DOUBLING_MONTHS) == APPROX(4 * BASELINE_HOURS)


def test_horizon_halves_going_backwards_in_time():
    assert horizon_at(-DOUBLING_MONTHS) == APPROX(BASELINE_HOURS / 2)


def test_horizon_growth_is_exponential_not_linear():
    """Прирост за второй год больше прироста за первый — линейка так не умеет."""
    first_year = horizon_at(12) - horizon_at(0)
    second_year = horizon_at(24) - horizon_at(12)
    assert second_year > first_year * 2


# ------------------------------------------------------------- months_to_cross
def test_months_to_cross_the_baseline_is_zero():
    assert months_to_cross(BASELINE_HOURS) == APPROX(0.0)


def test_months_to_cross_one_doubling():
    assert months_to_cross(2 * BASELINE_HOURS) == APPROX(DOUBLING_MONTHS)


def test_months_to_cross_is_the_inverse_of_horizon_at():
    """Прямая и обратная функции обязаны сходиться на любой точке."""
    for months in (3.0, 11.5, 29.0):
        assert months_to_cross(horizon_at(months)) == pytest.approx(months, abs=1e-9)


def test_already_passed_targets_land_in_the_past():
    assert months_to_cross(BASELINE_HOURS / 4) < 0


def test_non_positive_target_is_a_value_error_not_a_nan():
    with pytest.raises(ValueError):
        months_to_cross(0.0)


# ------------------------------------------------------ end_to_end_reliability
def test_reliability_of_a_single_step_is_the_step_reliability():
    assert end_to_end_reliability(0.99, 1) == APPROX(0.99)


def test_empty_trajectory_never_fails():
    assert end_to_end_reliability(0.9, 0) == APPROX(1.0)


def test_ninety_nine_percent_agent_is_a_coin_flip_at_seventy_steps():
    """Главное число урока: 0.99 на шаге — это меньше половины на 70 шагах."""
    assert end_to_end_reliability(0.99, 70) < 0.5


def test_reliability_decays_monotonically_with_length():
    values = [end_to_end_reliability(0.95, n) for n in (1, 10, 50, 200)]
    assert values == sorted(values, reverse=True)


def test_probability_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        end_to_end_reliability(1.5, 10)


# ------------------------------------------------------- max_steps_for_target
def test_max_steps_for_a_two_nines_agent():
    assert max_steps_for_target(0.99) == 68


def test_the_returned_length_still_clears_the_target_and_one_more_does_not():
    n = max_steps_for_target(0.99, 0.5)
    assert end_to_end_reliability(0.99, n) >= 0.5
    assert end_to_end_reliability(0.99, n + 1) < 0.5


def test_an_extra_nine_buys_far_more_than_a_constant_factor():
    """0.999 против 0.99 — прирост на порядок, а не втрое."""
    assert max_steps_for_target(0.999) > 10 * max_steps_for_target(0.99)


def test_perfect_agent_has_no_step_limit():
    assert max_steps_for_target(1.0) is None


def test_hopeless_agent_cannot_take_a_single_step():
    assert max_steps_for_target(0.0) == 0


# --------------------------------------------------------- deployment_horizon
def test_zero_gap_leaves_the_benchmark_number_alone():
    assert deployment_horizon(14.0, 0.0) == APPROX(14.0)


def test_gap_shrinks_the_horizon():
    assert deployment_horizon(14.0, 0.4) == APPROX(8.4)


def test_benchmark_is_a_ceiling_never_a_floor():
    """При любом ненулевом разрыве продакшен-горизонт строго меньше бенчмарка."""
    for gap in (0.05, 0.2, 0.78):
        assert deployment_horizon(14.0, gap) < 14.0


def test_gap_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        deployment_horizon(14.0, 1.4)


# ----------------------------------------------------------------- simulate_run
def test_a_never_failing_agent_finishes_all_steps():
    out = simulate_run(random.Random(0), 1.0, 3, 100.0, 1.0)
    assert out == {"status": "success", "steps": 3, "spent": APPROX(3.0)}


def test_an_always_failing_agent_dies_on_the_first_step():
    out = simulate_run(random.Random(0), 0.0, 3, 100.0, 1.0)
    assert out["status"] == "failed"
    assert out["steps"] == 1


def test_budget_is_checked_before_the_step_not_after():
    """Прогон не имеет права уйти за бюджет — списание идёт до действия."""
    out = simulate_run(random.Random(0), 1.0, 10, 2.5, 1.0)
    assert out["status"] == "budget_exhausted"
    assert out["steps"] == 2
    assert out["spent"] == APPROX(2.0)
    assert out["spent"] <= 2.5


def test_same_seed_reproduces_the_same_trajectory():
    """Без воспроизводимости разбор инцидента невозможен."""
    a = simulate_run(random.Random(7), 0.8, 50, 1000.0, 1.0)
    b = simulate_run(random.Random(7), 0.8, 50, 1000.0, 1.0)
    assert a == b


def test_different_seeds_give_different_trajectories():
    runs = {simulate_run(random.Random(s), 0.7, 50, 1000.0, 1.0)["steps"]
            for s in range(20)}
    assert len(runs) > 1


def test_spent_always_equals_cost_times_completed_steps():
    for seed in range(15):
        out = simulate_run(random.Random(seed), 0.85, 40, 12.0, 0.5)
        assert out["spent"] == APPROX(out["steps"] * 0.5)


# -------------------------------------------------------------- horizon_verdict
def test_short_task_with_room_to_spare_is_safe():
    assert horizon_verdict(4.0) == "safe"


def test_task_close_to_the_horizon_is_only_tight():
    assert horizon_verdict(10.0) == "tight"


def test_task_longer_than_the_horizon_is_a_runaway():
    assert horizon_verdict(40.0) == "runaway"


def test_eval_gap_is_applied_before_the_comparison():
    """Без скидки на разрыв та же задача читалась бы как safe."""
    assert horizon_verdict(4.0) == "safe"
    assert horizon_verdict(4.0, eval_gap=0.5) == "tight"


def test_verdict_only_worsens_as_the_task_grows():
    order = {"safe": 2, "tight": 1, "runaway": 0}
    ranks = [order[horizon_verdict(h)] for h in (1.0, 5.0, 12.0, 30.0)]
    assert ranks == sorted(ranks, reverse=True)
