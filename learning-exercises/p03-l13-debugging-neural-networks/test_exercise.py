"""Тесты к уроку «Отладка нейросетей». Правь exercise.py."""

import pytest

from exercise import (
    can_overfit_one_batch,
    dead_neurons,
    dead_relu_fractions,
    diagnose_loss_curve,
    find_bad_gradients,
    gradient_check,
    has_nan_or_inf,
    numeric_gradient,
    relative_difference,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
NUMERIC = lambda x: pytest.approx(x, abs=1e-6)

NAN = float("nan")
INF = float("inf")


# ------------------------------------------------------------ has_nan_or_inf
def test_clean_gradients_are_clean():
    assert has_nan_or_inf([1.0, 2.0, -3.5]) is False


def test_nan_is_detected():
    """NaN не равен сам себе — сравнением его не поймать, нужен math.isnan."""
    assert has_nan_or_inf([1.0, NAN]) is True


def test_nested_matrices_are_searched_to_the_bottom():
    """Градиенты приходят матрицами — обходить надо рекурсивно."""
    assert has_nan_or_inf([[1.0, 2.0], [3.0, INF]]) is True
    assert has_nan_or_inf([[1.0], [-INF]]) is True


def test_empty_gradient_is_not_broken():
    assert has_nan_or_inf([]) is False


def test_huge_but_finite_values_are_not_flagged():
    """1e300 — это ещё не поломка, это просто повод к градиентному клиппингу."""
    assert has_nan_or_inf([1e300]) is False


# -------------------------------------------------------- find_bad_gradients
def test_bad_layer_is_named():
    assert find_bad_gradients({"fc1": [1.0], "fc2": [NAN]}) == ["fc2"]


def test_healthy_network_reports_nothing():
    assert find_bad_gradients({"fc1": [1.0], "fc2": [2.0]}) == []


def test_bad_layers_come_back_sorted():
    """Порядок обхода словаря не должен просачиваться в отчёт."""
    named = {"fc3": [NAN], "fc1": [INF], "fc2": [0.5]}
    assert find_bad_gradients(named) == ["fc1", "fc3"]


# ------------------------------------------------------ dead_relu_fractions
def test_neuron_silent_on_every_sample_has_fraction_one():
    assert dead_relu_fractions([[0.0, 1.0], [0.0, 2.0]]) == APPROX([1.0, 0.0])


def test_fractions_are_counted_per_neuron_not_per_sample():
    """Нейрон — это столбец. Посчитаешь по строкам — получишь бессмыслицу."""
    assert dead_relu_fractions([[0.0, 1.0], [3.0, 2.0]]) == APPROX([0.5, 0.0])


def test_fractions_of_an_empty_batch_are_empty():
    assert dead_relu_fractions([]) == []


# ------------------------------------------------------------- dead_neurons
def test_dead_neuron_is_found():
    assert dead_neurons([[0.0, 1.0], [0.0, 2.0]]) == [0]


def test_default_threshold_demands_silence_on_every_sample():
    """Нейрон, молчащий в половине случаев, ещё не мёртв."""
    assert dead_neurons([[0.0, 1.0], [3.0, 2.0]]) == []


def test_lower_threshold_catches_half_dead_neurons():
    assert dead_neurons([[0.0, 1.0], [3.0, 2.0]], threshold=0.5) == [0]


# --------------------------------------------------------- numeric_gradient
def test_numeric_gradient_of_a_square():
    assert numeric_gradient(lambda p: p[0] ** 2, [3.0]) == NUMERIC([6.0])


def test_numeric_gradient_of_a_sum_of_squares():
    assert numeric_gradient(lambda p: p[0] ** 2 + p[1] ** 2, [3.0, 4.0]) == NUMERIC([6.0, 8.0])


def test_numeric_gradient_does_not_mutate_the_point():
    point = [1.0, 2.0]
    numeric_gradient(lambda p: p[0] * p[1], point)
    assert point == APPROX([1.0, 2.0])


def test_numeric_gradient_uses_the_central_difference():
    """Односторонняя разность на x^3 при h=1e-4 промахнётся на 3e-4."""
    assert numeric_gradient(lambda p: p[0] ** 3, [1.0]) == pytest.approx([3.0], abs=1e-6)


# ------------------------------------------------------ relative_difference
def test_identical_values_differ_by_zero():
    assert relative_difference(1.0, 1.0) == APPROX(0.0)


def test_relative_difference_is_normalised_by_the_larger_value():
    assert relative_difference(1.0, 2.0) == APPROX(0.5)


def test_two_zeros_do_not_divide_by_zero():
    """Оба градиента нулевые — это норма, а не повод падать."""
    assert relative_difference(0.0, 0.0) == APPROX(0.0)


# ---------------------------------------------------------- gradient_check
def test_correct_analytic_gradient_passes_the_check():
    assert gradient_check(lambda p: p[0] ** 2, [6.0], [3.0]) < 1e-5


def test_gradient_off_by_a_factor_of_two_is_caught():
    assert gradient_check(lambda p: p[0] ** 2, [3.0], [3.0]) > 1e-3


def test_swapped_gradient_components_are_caught():
    """Транспонированный backward: forward верный, лосс убывает, градиент врёт."""
    f = lambda p: p[0] ** 2 + 5 * p[1] ** 2
    correct = [2 * 1.0, 10 * 2.0]
    assert gradient_check(f, correct, [1.0, 2.0]) < 1e-5
    assert gradient_check(f, [correct[1], correct[0]], [1.0, 2.0]) > 1e-3


# ---------------------------------------------------- can_overfit_one_batch
def test_a_reachable_minimum_is_reached():
    assert can_overfit_one_batch(lambda p: (p[0] - 3.0) ** 2, [0.0]) is True


def test_an_irreducible_loss_floor_fails_the_test():
    """Лосс не может опуститься ниже 1.0 — значит модель мала для задачи."""
    assert can_overfit_one_batch(lambda p: (p[0] - 3.0) ** 2 + 1.0, [0.0]) is False


def test_a_loss_that_ignores_the_parameters_fails_the_test():
    """Нулевой градиент везде — оптимизатор не подключён к весам."""
    assert can_overfit_one_batch(lambda p: 5.0, [0.0]) is False


def test_a_diverging_learning_rate_fails_the_test():
    assert can_overfit_one_batch(lambda p: (p[0] - 3.0) ** 2, [0.0], lr=100.0) is False


def test_several_parameters_are_fitted_together():
    loss = lambda p: (p[0] - 1.0) ** 2 + (p[1] + 2.0) ** 2
    assert can_overfit_one_batch(loss, [0.0, 0.0]) is True


# ---------------------------------------------------- diagnose_loss_curve
def test_a_single_point_is_not_a_curve():
    assert diagnose_loss_curve([1.0]) == "NOT_ENOUGH_DATA"


def test_nan_beats_every_other_diagnosis():
    assert diagnose_loss_curve([1.0, 0.9, NAN, 0.7]) == "NAN_OR_INF"


def test_a_steadily_falling_curve_is_healthy():
    assert diagnose_loss_curve([1.0, 0.9, 0.8, 0.7]) == "HEALTHY"


def test_a_flat_curve_is_not_decreasing():
    assert diagnose_loss_curve([1.0, 1.0, 1.0, 1.0]) == "NOT_DECREASING"


def test_a_curve_that_jumps_is_oscillating():
    """Лосс вырос более чем вдвое за шаг — learning rate велик."""
    assert diagnose_loss_curve([1.0, 5.0, 1.0, 5.0]) == "OSCILLATING"


def test_a_growing_curve_is_diagnosed_before_the_flat_check():
    assert diagnose_loss_curve([0.5, 2.0, 8.0]) == "OSCILLATING"
