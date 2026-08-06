"""Тесты к уроку «Отбор признаков». Правь exercise.py."""

import math

import pytest

from exercise import (
    correlation,
    discretize,
    l1_select,
    logistic_weights,
    mutual_information,
    rfe,
    select_k_best,
    variance_threshold,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Три признака с известной ролью, повторённые четыре раза:
#   0 — сигнал (цель = 1 при положительном значении),
#   1 — константа (нулевая дисперсия),
#   2 — шум, независимый от цели по построению.
_BLOCK_SIGNAL = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
_BLOCK_NOISE = [1.0, -1.0, 0.0, 1.0, -1.0, 0.0]
_BLOCK_Y = [0, 0, 0, 1, 1, 1]

X = [
    [_BLOCK_SIGNAL[i], 5.0, _BLOCK_NOISE[i]] for _ in range(4) for i in range(6)
]
Y = [_BLOCK_Y[i] for _ in range(4) for i in range(6)]

SIGNAL = [row[0] for row in X]
NOISE = [row[2] for row in X]

# Симметричная зависимость: цель равна единице на краях диапазона.
# Корреляция ровно ноль, а связь железная.
NONLINEAR = [v for _ in range(5) for v in (-2.0, -1.0, 1.0, 2.0)]
NONLINEAR_Y = [v for _ in range(5) for v in (1, 0, 0, 1)]


# ----------------------------------------------------- variance_threshold
def test_variance_threshold_drops_the_constant_feature():
    mask, _ = variance_threshold(X, 0.01)
    assert mask == [True, False, True]


def test_variance_threshold_reports_the_variances():
    _, variances = variance_threshold([[1.0, 5.0], [3.0, 5.0]], 0.01)
    assert variances == APPROX([1.0, 0.0])


def test_variance_threshold_uses_population_variance():
    """Делим на n, а не на n-1: у [1, 3] дисперсия 1.0, а не 2.0."""
    _, variances = variance_threshold([[1.0], [3.0]])
    assert variances[0] == APPROX(1.0)


def test_higher_threshold_keeps_fewer_features():
    loose, _ = variance_threshold(X, 0.01)
    strict, _ = variance_threshold(X, 1.0)
    assert sum(strict) <= sum(loose)


def test_variance_threshold_says_nothing_about_usefulness():
    """Шумовой признак с приличной дисперсией фильтр спокойно пропускает."""
    mask, variances = variance_threshold(X, 0.01)
    assert mask[2] is True and variances[2] > 0.5


# --------------------------------------------------------------- correlation
def test_correlation_of_proportional_lists_is_one():
    assert correlation([1, 2, 3], [2, 4, 6]) == APPROX(1.0)


def test_correlation_of_opposite_lists_is_minus_one():
    assert correlation([1, 2, 3], [3, 2, 1]) == APPROX(-1.0)


def test_correlation_with_a_constant_is_zero():
    """Ловушка: нулевое отклонение обнуляет знаменатель."""
    assert correlation([1, 1, 1], [1, 2, 3]) == APPROX(0.0)


def test_correlation_does_not_depend_on_scale_or_shift():
    a = [1.0, 4.0, 2.0, 8.0]
    b = [2.0, 3.0, 5.0, 7.0]
    shifted = [3 * x + 100 for x in a]
    assert correlation(shifted, b) == APPROX(correlation(a, b))


def test_correlation_is_blind_to_a_symmetric_dependency():
    """Ровно ноль, хотя цель однозначно определяется признаком.

    Это и есть та дыра, ради которой в отборе признаков используют
    взаимную информацию, а не корреляцию.
    """
    assert correlation(NONLINEAR, NONLINEAR_Y) == APPROX(0.0)


# --------------------------------------------------------------- discretize
def test_discretize_splits_into_equal_bins():
    assert discretize([1, 2, 3, 4], 2) == [0, 0, 1, 1]


def test_discretize_puts_the_maximum_in_the_last_bin():
    binned = discretize([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10)
    assert binned[0] == 0
    assert binned[-1] == 9


def test_discretize_of_a_constant_list_is_all_zeros():
    """Ловушка: ширина корзины нулевая, границы совпали."""
    assert discretize([5, 5, 5], 4) == [0, 0, 0]


def test_discretize_never_leaves_the_bin_range():
    binned = discretize(SIGNAL, 5)
    assert all(0 <= b <= 4 for b in binned)


# ------------------------------------------------------ mutual_information
def test_mutual_information_of_a_copy_of_the_target():
    """Признак совпадает с целью — информация равна энтропии цели, log 2."""
    scores = mutual_information([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1], 2)
    assert scores[0] == APPROX(math.log(2))


def test_mutual_information_of_an_independent_feature_is_zero():
    scores = mutual_information(X, Y, 4)
    assert scores[2] == APPROX(0.0)


def test_mutual_information_is_never_negative():
    assert all(s >= -1e-12 for s in mutual_information(X, Y, 5))


def test_signal_carries_more_information_than_noise():
    scores = mutual_information(X, Y, 6)
    assert scores[0] > scores[2]


def test_mutual_information_sees_what_correlation_misses():
    """Корреляция нулевая, взаимная информация — заметно положительная."""
    nonlinear_X = [[v] for v in NONLINEAR]
    assert correlation(NONLINEAR, NONLINEAR_Y) == APPROX(0.0)
    assert mutual_information(nonlinear_X, NONLINEAR_Y, 4)[0] > 0.3


def test_constant_feature_carries_no_information():
    assert mutual_information(X, Y, 4)[1] == APPROX(0.0)


# ------------------------------------------------------------ select_k_best
def test_select_k_best_returns_the_top_scores():
    assert select_k_best([0.1, 0.9, 0.5], 2) == [1, 2]


def test_select_k_best_returns_indices_in_ascending_order():
    assert select_k_best([0.5, 0.1, 0.9, 0.2], 2) == [0, 2]


def test_select_k_best_breaks_ties_by_index():
    """Отбор обязан быть воспроизводимым: два запуска — одна и та же модель."""
    assert select_k_best([1.0, 1.0, 0.0], 1) == [0]


def test_select_k_best_of_everything_returns_everything():
    assert select_k_best([0.3, 0.1, 0.2], 3) == [0, 1, 2]


# ------------------------------------------------------- logistic_weights
def test_logistic_weights_learn_the_direction_of_the_signal():
    w, _ = logistic_weights([[-1.0], [1.0]], [0, 1], epochs=100)
    assert w[0] > 0


def test_logistic_weights_ignore_a_constant_feature():
    """Константа не различает классы: её вес остаётся около нуля."""
    w, _ = logistic_weights(X, Y, epochs=100)
    assert abs(w[1]) < abs(w[0])


def test_signal_gets_the_largest_weight():
    w, _ = logistic_weights(X, Y, epochs=100)
    assert abs(w[0]) > abs(w[1]) and abs(w[0]) > abs(w[2])


def test_l1_penalty_shrinks_the_weights():
    plain, _ = logistic_weights(X, Y, 0.0, 0.1, 100)
    penalized, _ = logistic_weights(X, Y, 0.5, 0.1, 100)
    assert abs(penalized[0]) < abs(plain[0])


def test_logistic_weights_survive_huge_feature_values():
    """Ловушка: exp(1000) — это OverflowError, а не бесконечность."""
    w, _ = logistic_weights([[-1e6], [1e6]], [0, 1], epochs=20)
    assert all(math.isfinite(v) for v in w)


# --------------------------------------------------------------------- rfe
def test_rfe_keeps_the_informative_feature():
    mask, _ = rfe(X, Y, 1, epochs=60)
    assert mask == [True, False, False]


def test_rfe_selects_exactly_the_requested_count():
    mask, _ = rfe(X, Y, 2, epochs=60)
    assert sum(mask) == 2


def test_rfe_ranks_the_selected_features_as_one():
    mask, rankings = rfe(X, Y, 1, epochs=60)
    assert rankings[0] == 1
    assert [r == 1 for r in rankings] == mask


def test_rfe_gives_the_worst_feature_the_highest_rank():
    """Первым выбрасывают самый бесполезный, и его ранг равен числу признаков."""
    _, rankings = rfe(X, Y, 1, epochs=60)
    assert max(rankings) == len(X[0])
    assert rankings[0] == min(rankings)


def test_rfe_keeping_everything_changes_nothing():
    mask, rankings = rfe(X, Y, 3, epochs=60)
    assert mask == [True, True, True]
    assert rankings == [1, 1, 1]


# ---------------------------------------------------------------- l1_select
def test_l1_select_keeps_only_the_informative_feature():
    assert l1_select(X, Y, 0.5, 0.1, 100) == [True, False, False]


def test_bigger_alpha_selects_fewer_features():
    """Чем сильнее штраф, тем больше весов уходит ровно в ноль."""
    weak = l1_select(X, Y, 0.5, 0.1, 100)
    strong = l1_select(X, Y, 20.0, 0.1, 100)
    assert sum(strong) < sum(weak)


def test_zero_alpha_keeps_everything_that_moved():
    """Без штрафа вес просто уменьшается, но нуля не достигает."""
    assert l1_select(X, Y, 0.0, 0.1, 100)[0] is True


def test_l1_select_agrees_with_rfe_on_the_signal():
    """Разные семейства методов, один и тот же вывод — признак 0 главный."""
    rfe_mask, _ = rfe(X, Y, 1, epochs=60)
    assert l1_select(X, Y, 0.5, 0.1, 100)[0] == rfe_mask[0] is True
