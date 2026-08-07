"""Тесты к уроку «Оценка и бенчмарки координации». Правь exercise.py."""

import math

import pytest

from exercise import (
    CONTAMINATION_THRESHOLD,
    accuracy,
    contamination_gap,
    coordination_gain,
    cost_per_milestone,
    lift_over_random,
    mean_confidence_interval,
    milestone_score,
    scorecard,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SYSTEM = {
    "seen": [True] * 8 + [False] * 2,
    "held": [True] * 6 + [False] * 4,
    "milestones": [True, True, True, False],
    "milestone_weights": None,
    "tokens": 40000,
    "price_per_1k": 0.01,
    "n_options": 4,
    "team": [1, 1, 1, 0],
    "solos": [[1, 1, 0, 0], [0, 0, 1, 0]],
}


# ---------------------------------------------------------------- accuracy
def test_accuracy_is_the_share_of_successes():
    assert accuracy([True, True, False, False]) == APPROX(0.5)


def test_accuracy_of_an_empty_run_is_zero():
    assert accuracy([]) == APPROX(0.0)


def test_accuracy_treats_partial_progress_as_failure():
    """Всё или ничего: три из четырёх шагов — это ноль. Отсюда и нужны вехи."""
    assert accuracy([False, False, False]) == APPROX(0.0)


# --------------------------------------------------------- milestone_score
def test_milestone_score_gives_partial_credit():
    assert milestone_score([True, True, False, False]) == APPROX(0.5)


def test_milestone_score_normalises_by_the_sum_of_weights():
    """Веса задают в очках, а не в долях: нормировать надо на их сумму."""
    assert milestone_score([True, False], [3.0, 1.0]) == APPROX(0.75)


def test_milestone_score_of_no_milestones_is_zero():
    assert milestone_score([]) == APPROX(0.0)


def test_milestone_score_separates_near_miss_from_no_start():
    """Ровно тот случай, ради которого MARBLE и ввела вехи."""
    assert milestone_score([True] * 4 + [False]) > milestone_score([False] * 5)


def test_milestone_score_survives_all_zero_weights():
    assert milestone_score([True, True], [0.0, 0.0]) == APPROX(0.0)


# -------------------------------------------------------- lift_over_random
def test_random_level_scores_zero_lift():
    assert lift_over_random(0.25, 4) == APPROX(0.0)


def test_perfect_score_gets_full_lift():
    assert lift_over_random(1.0, 4) == APPROX(1.0)


def test_lift_is_normalised_by_the_remaining_headroom():
    assert lift_over_random(0.625, 4) == APPROX(0.5)


def test_below_random_performance_gives_negative_lift():
    """Результат COMMA: без явной случайной базы 0.1 выглядит «каким-то числом»."""
    assert lift_over_random(0.1, 4) < 0


def test_lift_rejects_a_degenerate_option_count():
    with pytest.raises(ValueError):
        lift_over_random(1.0, 1)


# ------------------------------------------------------- coordination_gain
def test_team_that_only_reproduces_its_best_member_gains_nothing():
    """Главное свойство метрики: копия сильнейшего одиночки — это ноль."""
    assert coordination_gain([1, 1, 0, 0], [[1, 1, 0, 0], [0, 0, 0, 0]]) == APPROX(0.0)


def test_complementary_agents_produce_positive_gain():
    assert coordination_gain([1, 1, 1, 1], [[1, 1, 0, 0], [0, 0, 1, 1]]) == APPROX(0.5)


def test_coordination_tax_shows_up_as_negative_gain():
    """MedAgentBoard находит это регулярно: команда хуже одной модели."""
    assert coordination_gain([0, 0, 0, 0], [[1, 1, 1, 1]]) == APPROX(-1.0)


def test_gain_subtracts_the_best_solo_not_the_average_solo():
    """Со средним одиночкой можно «показать» выигрыш, которого нет."""
    team = [1, 1, 0, 0]
    solos = [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert coordination_gain(team, solos) == APPROX(0.0)


def test_adding_a_weak_agent_never_inflates_the_gain():
    strong_only = coordination_gain([1, 1, 1, 0], [[1, 1, 0, 0]])
    with_weak = coordination_gain([1, 1, 1, 0], [[1, 1, 0, 0], [0, 0, 0, 0]])
    assert with_weak == APPROX(strong_only)


def test_coordination_gain_needs_at_least_one_solo_run():
    with pytest.raises(ValueError):
        coordination_gain([1, 1], [])


# ------------------------------------------------------- cost_per_milestone
def test_cost_per_milestone_divides_money_by_progress():
    assert cost_per_milestone(20000, 0.5, 0.01) == APPROX(0.4)


def test_cost_per_milestone_at_full_progress_is_just_the_cost():
    assert cost_per_milestone(20000, 1.0, 0.01) == APPROX(0.2)


def test_zero_progress_costs_infinity_not_zero():
    """Токены сожжены, вех нет — это бесконечная цена за веху."""
    assert cost_per_milestone(20000, 0.0, 0.01) == math.inf


def test_a_pricier_system_can_lose_despite_higher_progress():
    """Ранжирование по точности и по цене за веху расходится — пункт 6 чеклиста."""
    cheap = cost_per_milestone(10000, 0.6, 0.01)
    pricey = cost_per_milestone(200000, 0.9, 0.01)
    assert cheap < pricey


# -------------------------------------------------------- contamination_gap
def test_contamination_gap_is_positive_when_the_seen_split_is_easier():
    assert contamination_gap([True] * 4, [True, False, False, False]) == APPROX(0.75)


def test_no_gap_on_a_clean_benchmark():
    assert contamination_gap([True, False], [True, False]) == APPROX(0.0)


def test_a_negative_gap_is_not_evidence_of_contamination():
    """Отложенный сплит оказался легче — это шум, а не улика."""
    assert contamination_gap([True, False, False], [True, True, True]) < 0


# -------------------------------------------------- mean_confidence_interval
def test_identical_runs_have_a_zero_width_interval():
    assert mean_confidence_interval([0.5, 0.5, 0.5, 0.5]) == (APPROX(0.5), APPROX(0.0))


def test_confidence_interval_of_two_extremes():
    mean, half = mean_confidence_interval([0.0, 1.0])
    assert mean == APPROX(0.5)
    assert half == pytest.approx(0.98, abs=1e-9)


def test_a_single_run_gives_no_interval_at_all():
    """Один прогон — не «интервал нулевой ширины», а отсутствие оценки."""
    assert mean_confidence_interval([0.7]) == (APPROX(0.7), math.inf)


def test_more_runs_shrink_the_interval():
    few = mean_confidence_interval([0.0, 1.0, 0.0, 1.0])[1]
    many = mean_confidence_interval([0.0, 1.0] * 20)[1]
    assert many < few


# --------------------------------------------------------------- scorecard
def test_scorecard_reports_accuracy_from_the_held_out_split():
    """Публиковать число с виденного сплита — ровно та ошибка, ради которой есть Pro."""
    assert scorecard(SYSTEM)["accuracy"] == APPROX(0.6)


def test_scorecard_flags_contamination_above_the_threshold():
    system = dict(SYSTEM, seen=[True] * 10, held=[True] * 5 + [False] * 5)
    card = scorecard(system)
    assert card["contamination_gap"] == APPROX(0.5)
    assert card["contaminated"] is True


def test_scorecard_leaves_a_clean_benchmark_unflagged():
    system = dict(SYSTEM, seen=[True] * 6 + [False] * 4, held=[True] * 6 + [False] * 4)
    assert scorecard(system)["contaminated"] is False


def test_scorecard_threshold_default_matches_the_module_constant():
    system = dict(SYSTEM, seen=[True] * 10, held=[True] * 5 + [False] * 5)
    assert scorecard(system) == scorecard(system, CONTAMINATION_THRESHOLD)


def test_scorecard_carries_every_metric_of_the_checklist():
    card = scorecard(SYSTEM)
    assert sorted(card) == [
        "accuracy", "confidence_interval", "contaminated", "contamination_gap",
        "coordination_gain", "cost_per_milestone", "lift_over_random", "milestone",
    ]


def test_scorecard_agrees_with_the_standalone_metrics():
    card = scorecard(SYSTEM)
    assert card["milestone"] == APPROX(milestone_score(SYSTEM["milestones"]))
    assert card["coordination_gain"] == APPROX(
        coordination_gain(SYSTEM["team"], SYSTEM["solos"])
    )
    assert card["lift_over_random"] == APPROX(
        lift_over_random(accuracy(SYSTEM["held"]), SYSTEM["n_options"])
    )
