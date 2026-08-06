"""Тесты к уроку «Подбор гиперпараметров». Правь exercise.py."""

import math

import pytest

from exercise import (
    bayes_search,
    count_unique,
    expected_improvement,
    grid_search,
    log_uniform,
    random_search,
    sample_config,
    surrogate,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SPACE = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
}


def objective(config):
    """Синтетическая цель урока: максимум 10.0 при lr = 0.01 и depth = 4."""
    return (
        -(math.log10(config["learning_rate"]) + 2) ** 2
        - (config["max_depth"] - 4) ** 2
        + 10
    )


# --------------------------------------------------------------- log_uniform
def test_log_uniform_hits_both_ends():
    assert log_uniform(0.001, 1.0, 0.0) == APPROX(0.001)
    assert log_uniform(0.001, 1.0, 1.0) == APPROX(1.0)


def test_log_uniform_midpoint_is_the_geometric_mean():
    assert log_uniform(0.001, 1.0, 0.5) == pytest.approx(math.sqrt(0.001 * 1.0))


def test_log_uniform_is_not_the_linear_midpoint():
    """Ловушка всего урока: линейная середина между 0.001 и 1.0 — это 0.5,
    и весь бюджет уходит на бесполезно большие learning rate."""
    assert log_uniform(0.001, 1.0, 0.5) < 0.1


def test_log_uniform_grows_with_u():
    values = [log_uniform(0.0001, 10.0, u / 10) for u in range(11)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_log_uniform_gives_each_decade_the_same_share():
    """От 0.001 до 1.0 три порядка: треть шкалы — ровно один порядок."""
    assert log_uniform(0.001, 1.0, 1 / 3) == pytest.approx(0.01)


# ------------------------------------------------------------- sample_config
def test_sample_config_is_reproducible_for_the_same_seed():
    assert sample_config(SPACE, 0) == sample_config(SPACE, 0)


def test_sample_config_respects_integer_bounds():
    for seed in range(20):
        depth = sample_config(SPACE, seed)["max_depth"]
        assert isinstance(depth, int)
        assert 2 <= depth <= 8


def test_sample_config_keeps_log_floats_inside_the_range():
    for seed in range(20):
        assert 0.001 <= sample_config(SPACE, seed)["learning_rate"] <= 1.0


def test_sample_config_picks_from_the_choice_list():
    space = {"kernel": ("choice", ["rbf", "linear", "poly"])}
    assert all(sample_config(space, s)["kernel"] in space["kernel"][1] for s in range(20))


def test_sample_config_does_not_depend_on_dict_key_order():
    """Ловушка: без сортировки ключей тот же seed даёт разные конфигурации."""
    shuffled = {"max_depth": SPACE["max_depth"], "learning_rate": SPACE["learning_rate"]}
    assert sample_config(SPACE, 7) == sample_config(shuffled, 7)


# --------------------------------------------------------------- grid_search
def test_grid_search_evaluates_the_whole_product():
    _, _, history = grid_search(lambda c: 0.0, {"a": [1, 2], "b": [3, 4, 5]})
    assert len(history) == 6


def test_grid_search_finds_the_maximum_on_the_grid():
    best, score, _ = grid_search(
        objective, {"learning_rate": [0.001, 0.01, 0.1, 1.0], "max_depth": [2, 4, 6]}
    )
    assert best == {"learning_rate": 0.01, "max_depth": 4}
    assert score == APPROX(10.0)


def test_grid_search_returns_the_config_that_earned_the_score():
    best, score, _ = grid_search(objective, {"learning_rate": [0.001, 0.1], "max_depth": [3, 5]})
    assert objective(best) == APPROX(score)


def test_grid_search_best_score_is_the_maximum_of_its_history():
    _, score, history = grid_search(objective, {"learning_rate": [0.001, 0.1], "max_depth": [2, 4, 8]})
    assert score == APPROX(max(s for _, s in history))


# ------------------------------------------------------------- random_search
def test_random_search_spends_exactly_the_budget():
    _, _, history = random_search(objective, SPACE, n_iter=9, seed=0)
    assert len(history) == 9


def test_random_search_is_reproducible_for_the_same_seed():
    assert random_search(objective, SPACE, n_iter=9, seed=1) == random_search(
        objective, SPACE, n_iter=9, seed=1
    )


def test_random_search_explores_differently_with_another_seed():
    first = random_search(objective, SPACE, n_iter=9, seed=1)[2]
    second = random_search(objective, SPACE, n_iter=9, seed=99)[2]
    assert first != second


def test_random_search_best_score_is_the_maximum_of_its_history():
    _, score, history = random_search(objective, SPACE, n_iter=15, seed=3)
    assert score == APPROX(max(s for _, s in history))


# -------------------------------------------------------------- count_unique
def test_grid_search_wastes_its_budget_on_repeated_values():
    """9 вычислений сетки 3x3 — всего 3 разных learning rate."""
    _, _, history = grid_search(
        objective, {"learning_rate": [0.001, 0.01, 0.1], "max_depth": [3, 4, 5]}
    )
    assert len(history) == 9
    assert count_unique(history, "learning_rate") == 3


def test_random_search_gets_a_fresh_value_on_every_trial():
    """Тот же бюджет из 9 вычислений — 9 разных learning rate. Bergstra & Bengio."""
    _, _, history = random_search(objective, SPACE, n_iter=9, seed=0)
    assert count_unique(history, "learning_rate") == 9


def test_count_unique_of_a_frozen_parameter_is_one():
    _, _, history = grid_search(objective, {"learning_rate": [0.01], "max_depth": [2, 4, 6]})
    assert count_unique(history, "learning_rate") == 1


# ------------------------------------------------------ expected_improvement
def test_expected_improvement_with_no_uncertainty_is_the_plain_gain():
    assert expected_improvement(5.0, 0.0, 4.0) == APPROX(1.0)


def test_expected_improvement_of_a_certainly_worse_point_is_zero():
    """Ловушка: EI не бывает отрицательной — в плохую точку мы просто не пойдём."""
    assert expected_improvement(3.0, 0.0, 4.0) == APPROX(0.0)


def test_expected_improvement_is_never_negative():
    assert all(
        expected_improvement(mu, sigma, 5.0) >= 0.0
        for mu in (-10.0, 0.0, 5.0, 20.0)
        for sigma in (0.0, 0.1, 3.0)
    )


def test_expected_improvement_rewards_uncertainty():
    """Разведка: при одинаковом прогнозе интереснее та точка, про которую мы
    знаем меньше."""
    assert expected_improvement(4.0, 2.0, 5.0) > expected_improvement(4.0, 0.5, 5.0)


def test_expected_improvement_rewards_a_higher_prediction():
    """Эксплуатация: при одинаковой неопределённости интереснее та, где обещают больше."""
    assert expected_improvement(6.0, 1.0, 5.0) > expected_improvement(4.0, 1.0, 5.0)


# ----------------------------------------------------------------- surrogate
def test_surrogate_reproduces_an_observed_point_with_zero_uncertainty():
    assert surrogate([([0.0], 5.0)], [0.0]) == APPROX((5.0, 0.0))


def test_surrogate_without_observations_admits_it_knows_nothing():
    """Ловушка: делить не на что. Ответ — (0.0, 1.0), а не ZeroDivisionError."""
    assert surrogate([], [1.0]) == APPROX((0.0, 1.0))


def test_surrogate_between_two_observations_predicts_their_average():
    mu, _ = surrogate([([0.0], 0.0), ([1.0], 10.0)], [0.5])
    assert mu == APPROX(5.0)


def test_surrogate_uncertainty_grows_with_distance():
    _, near = surrogate([([0.0], 5.0)], [0.1])
    _, far = surrogate([([0.0], 5.0)], [2.0])
    assert near < far <= 1.0


def test_surrogate_leans_toward_the_closer_observation():
    mu, _ = surrogate([([0.0], 0.0), ([1.0], 10.0)], [0.1])
    assert mu < 5.0


# --------------------------------------------------------------- bayes_search
def test_bayes_search_spends_exactly_the_budget():
    _, _, history = bayes_search(objective, SPACE, n_iter=12, seed=0)
    assert len(history) == 12


def test_bayes_search_is_reproducible_for_the_same_seed():
    assert bayes_search(objective, SPACE, n_iter=12, seed=4) == bayes_search(
        objective, SPACE, n_iter=12, seed=4
    )


def test_bayes_search_starts_with_plain_random_points():
    """Пока наблюдений нет, суррогату не на чем учиться — первые n_initial
    конфигураций обязаны совпасть со случайным поиском на том же seed."""
    _, _, bayes_history = bayes_search(objective, SPACE, n_iter=12, n_initial=5, seed=0)
    _, _, random_history = random_search(objective, SPACE, n_iter=5, seed=0)
    assert [c for c, _ in bayes_history[:5]] == [c for c, _ in random_history]


def test_bayes_search_best_score_is_the_maximum_of_its_history():
    _, score, history = bayes_search(objective, SPACE, n_iter=12, seed=2)
    assert score == APPROX(max(s for _, s in history))


def test_bayes_search_beats_random_search_on_the_same_budget():
    """Смысл суррогата: он не тратит бюджет на заведомо плохие области."""
    seeds = (0, 10, 20, 30, 40)
    random_mean = sum(random_search(objective, SPACE, n_iter=20, seed=s)[1] for s in seeds)
    bayes_mean = sum(bayes_search(objective, SPACE, n_iter=20, seed=s)[1] for s in seeds)
    assert bayes_mean > random_mean
