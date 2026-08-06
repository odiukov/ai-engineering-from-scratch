"""Тесты к уроку «Функции активации». Правь exercise.py."""

import math

import pytest

from exercise import (
    d_gelu,
    d_leaky_relu,
    d_relu,
    d_sigmoid,
    d_swish,
    d_tanh,
    dead_zone_fraction,
    gelu,
    leaky_relu,
    max_derivative,
    relu,
    sigmoid,
    softmax,
    swish,
    tanh_act,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# точки, на которых сверяем каждую пару «функция — её производная»
SMOOTH_POINTS = (-3.0, -1.2, -0.4, 0.3, 1.7, 4.0)


def numeric_derivative(f, x, h=1e-5):
    """Центральная разность — независимый ответ на вопрос «а производная ли это»."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def check_pair(f, df, points=SMOOTH_POINTS, tol=1e-6):
    for x in points:
        assert df(x) == pytest.approx(numeric_derivative(f, x), abs=tol), f"разошлись в x = {x}"


# ------------------------------------------------------- sigmoid / d_sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_survives_huge_inputs():
    assert sigmoid(-1000.0) == APPROX(0.0)
    assert sigmoid(1000.0) == APPROX(1.0)


def test_d_sigmoid_peaks_at_a_quarter():
    assert d_sigmoid(0.0) == APPROX(0.25)


def test_d_sigmoid_matches_the_numeric_derivative():
    check_pair(sigmoid, d_sigmoid)


def test_d_sigmoid_never_exceeds_a_quarter():
    """Потолок 0.25 — источник затухания градиента в сигмоидных сетях."""
    assert max_derivative(d_sigmoid) == pytest.approx(0.25, abs=1e-6)


# ------------------------------------------------------- tanh_act / d_tanh
def test_tanh_is_zero_centered():
    assert tanh_act(0.0) == APPROX(0.0)
    assert tanh_act(1.3) == APPROX(-tanh_act(-1.3))


def test_tanh_stays_between_minus_one_and_one():
    assert -1.0 < tanh_act(-8.0) < tanh_act(8.0) < 1.0


def test_tanh_survives_huge_inputs():
    """Ручная (e^x - e^-x)/(e^x + e^-x) на x = 800 падает с OverflowError."""
    assert tanh_act(-800.0) == APPROX(-1.0)
    assert tanh_act(800.0) == APPROX(1.0)


def test_d_tanh_peaks_at_one():
    assert d_tanh(0.0) == APPROX(1.0)


def test_d_tanh_matches_the_numeric_derivative():
    check_pair(tanh_act, d_tanh)


def test_tanh_passes_gradient_better_than_sigmoid():
    assert max_derivative(d_tanh) > max_derivative(d_sigmoid)


# ----------------------------------------------------------- relu / d_relu
def test_relu_passes_positive_unchanged():
    assert relu(3.0) == APPROX(3.0)


def test_relu_blocks_negative():
    assert relu(-3.0) == APPROX(0.0)


def test_d_relu_is_one_on_the_positive_side():
    assert d_relu(3.0) == APPROX(1.0)


def test_d_relu_is_zero_on_the_negative_side():
    assert d_relu(-3.0) == APPROX(0.0)


def test_d_relu_at_zero_is_zero_by_convention():
    assert d_relu(0.0) == APPROX(0.0)


def test_d_relu_matches_the_numeric_derivative_away_from_the_kink():
    """В нуле у ReLU излома нет производной, поэтому сверяем в стороне от него."""
    check_pair(relu, d_relu, points=(-4.0, -0.5, 0.5, 4.0))


# --------------------------------------------- leaky_relu / d_leaky_relu
def test_leaky_relu_passes_positive_unchanged():
    assert leaky_relu(3.0) == APPROX(3.0)


def test_leaky_relu_leaks_on_the_negative_side():
    assert leaky_relu(-3.0) == APPROX(-0.03)


def test_leaky_relu_never_fully_kills_the_gradient():
    """Это и есть лекарство от мёртвых нейронов: слева не ноль, а alpha."""
    assert d_leaky_relu(-100.0) == APPROX(0.01)
    assert d_relu(-100.0) == APPROX(0.0)


def test_d_leaky_relu_matches_the_numeric_derivative():
    check_pair(leaky_relu, d_leaky_relu, points=(-4.0, -0.5, 0.5, 4.0))


def test_leaky_relu_alpha_is_adjustable():
    assert leaky_relu(-2.0, alpha=0.5) == APPROX(-1.0)
    assert d_leaky_relu(-2.0, alpha=0.5) == APPROX(0.5)


# ----------------------------------------------------------- gelu / d_gelu
def test_gelu_at_zero_is_zero():
    assert gelu(0.0) == APPROX(0.0)


def test_gelu_almost_passes_large_positive_values():
    assert gelu(3.0) == pytest.approx(2.99595, abs=1e-5)


def test_gelu_allows_small_negative_values():
    """В отличие от ReLU, GELU не обрезает отрицательное в ноль наглухо."""
    assert -0.2 < gelu(-1.0) < 0.0


def test_d_gelu_matches_the_numeric_derivative():
    check_pair(gelu, d_gelu)


def test_d_gelu_at_zero_is_half():
    assert d_gelu(0.0) == APPROX(0.5)


def test_d_gelu_can_exceed_one():
    """GELU немонотонна, и её производная местами больше единицы — так и надо."""
    assert max_derivative(d_gelu) > 1.0


# --------------------------------------------------------- swish / d_swish
def test_swish_at_zero_is_zero():
    assert swish(0.0) == APPROX(0.0)


def test_swish_is_x_times_sigmoid():
    assert swish(2.5) == APPROX(2.5 * sigmoid(2.5))


def test_d_swish_matches_the_numeric_derivative():
    check_pair(swish, d_swish)


def test_d_swish_at_zero_is_half():
    assert d_swish(0.0) == APPROX(0.5)


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([2.0, 1.0, 0.0, -1.0])) == APPROX(1.0)


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([0.0, 0.0, 0.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_survives_huge_logits():
    """math.exp(1000) — это OverflowError. Вычитание максимума спасает."""
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])


def test_softmax_is_shift_invariant():
    """Прибавили константу ко всем логитам — распределение не изменилось."""
    assert softmax([2.0, 1.0, 0.0]) == pytest.approx(softmax([102.0, 101.0, 100.0]))


def test_softmax_keeps_the_order_of_logits():
    probs = softmax([0.5, 3.0, -2.0])
    assert probs[1] > probs[0] > probs[2]


def test_softmax_worked_example():
    assert softmax([2.0, 1.0, 0.0]) == pytest.approx([0.665241, 0.244728, 0.090031], abs=1e-6)


# ------------------------------------------ dead_zone_fraction / max_derivative
def test_relu_dead_zone_is_half_the_axis():
    """Половина входов даёт нулевой градиент — отсюда и берутся мёртвые нейроны."""
    assert 0.49 <= dead_zone_fraction(d_relu) <= 0.52


def test_smooth_activations_have_a_smaller_dead_zone_than_relu():
    assert dead_zone_fraction(d_swish) < dead_zone_fraction(d_relu)
    assert dead_zone_fraction(d_gelu) < dead_zone_fraction(d_relu)


def test_dead_zone_respects_the_scan_range():
    """Шире окно — больше насыщенных точек у tanh."""
    assert dead_zone_fraction(d_tanh, -10.0, 10.0) > dead_zone_fraction(d_tanh, -5.0, 5.0)


def test_max_derivative_of_relu_is_one():
    assert max_derivative(d_relu) == APPROX(1.0)


def test_relu_beats_sigmoid_on_gradient_throughput():
    """Десять слоёв: 1^10 против 0.25^10, то есть в миллион раз."""
    ten_relu = max_derivative(d_relu) ** 10
    ten_sigmoid = max_derivative(d_sigmoid) ** 10
    assert ten_relu / ten_sigmoid > 1e6


def test_max_derivative_uses_the_given_function():
    assert max_derivative(math.cos, 0.0, math.pi, 100) == pytest.approx(1.0, abs=1e-3)
