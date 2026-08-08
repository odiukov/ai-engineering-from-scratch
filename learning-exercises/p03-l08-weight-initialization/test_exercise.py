"""Тесты к уроку «Инициализация весов и устойчивость обучения». Правь exercise.py."""

import math

import pytest

from exercise import (
    forward_magnitudes,
    is_symmetry_broken,
    kaiming_init,
    matvec,
    random_init,
    recommend_init,
    variance,
    xavier_init,
    zero_init,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

RELU = lambda t: max(0.0, t)
TANH = math.tanh
SIGMOID = lambda t: 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, t))))


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in matrix for v in row]


def std_of(matrix):
    return math.sqrt(variance(flat(matrix)))


# -------------------------------------------------------------- zero_init
def test_zero_init_shape_is_fan_out_rows_by_fan_in_columns():
    m = zero_init(2, 3)
    assert len(m) == 3
    assert all(len(row) == 2 for row in m)


def test_zero_init_is_all_zeros():
    assert flat(zero_init(4, 4)) == APPROX([0.0] * 16)


def test_zero_init_leaves_every_neuron_identical():
    """Симметрия не сломана — слой из 8 нейронов работает как один."""
    assert is_symmetry_broken(zero_init(5, 8)) is False


# ------------------------------------------------------------ random_init
def test_random_init_shape_is_fan_out_rows_by_fan_in_columns():
    m = random_init(4, 3)
    assert len(m) == 3
    assert all(len(row) == 4 for row in m)


def test_random_init_is_reproducible_for_the_same_seed():
    """Два вызова с одним seed обязаны совпасть до последнего бита."""
    assert flat(random_init(6, 6, seed=7)) == APPROX(flat(random_init(6, 6, seed=7)))


def test_random_init_differs_for_a_different_seed():
    assert flat(random_init(6, 6, seed=1)) != APPROX(flat(random_init(6, 6, seed=2)))


def test_random_init_breaks_symmetry():
    assert is_symmetry_broken(random_init(5, 8, seed=0)) is True


def test_random_init_scale_controls_the_spread():
    """Разброс весов растёт ровно как scale: вдвое больший scale — вдвое шире."""
    narrow = std_of(random_init(40, 40, scale=0.5, seed=3))
    wide = std_of(random_init(40, 40, scale=2.0, seed=3))
    assert wide / narrow == pytest.approx(4.0, rel=0.05)


# ------------------------------------------------------------ xavier_init
def test_xavier_std_matches_the_glorot_formula():
    expected = math.sqrt(2.0 / (64 + 64))
    assert std_of(xavier_init(64, 64, seed=1)) == pytest.approx(expected, rel=0.06)


def test_xavier_uses_both_fan_in_and_fan_out():
    """Увеличили только fan_out — знаменатель вырос, веса стали уже."""
    narrow = std_of(xavier_init(64, 256, seed=1))
    wide = std_of(xavier_init(64, 64, seed=1))
    assert narrow < wide


# ----------------------------------------------------------- kaiming_init
def test_kaiming_std_matches_the_he_formula():
    expected = math.sqrt(2.0 / 64)
    assert std_of(kaiming_init(64, 64, seed=1)) == pytest.approx(expected, rel=0.06)


def test_kaiming_ignores_fan_out():
    """He считал только прямой проход, поэтому fan_out на разброс не влияет."""
    small = std_of(kaiming_init(64, 16, seed=1))
    big = std_of(kaiming_init(64, 256, seed=1))
    assert small == pytest.approx(big, rel=0.06)


def test_kaiming_is_wider_than_xavier_on_the_same_layer():
    """Та самая двойка на компенсацию ReLU и делает Kaiming шире."""
    assert std_of(kaiming_init(64, 64, seed=1)) > std_of(xavier_init(64, 64, seed=1))


# ---------------------------------------------------------------- variance
def test_variance_of_constant_list_is_zero():
    assert variance([3.0, 3.0, 3.0, 3.0]) == APPROX(0.0)


def test_variance_known_value():
    assert variance([2.0, 4.0, 4.0, 6.0]) == APPROX(2.0)


def test_variance_of_empty_list_is_zero():
    """Пустой вход не должен ронять функцию делением на ноль."""
    assert variance([]) == APPROX(0.0)


# ------------------------------------------------------------------ matvec
def test_matvec_with_identity_returns_the_input():
    assert matvec([[1, 0], [0, 1]], [3, 4]) == APPROX([3.0, 4.0])


def test_matvec_row_is_a_dot_product():
    assert matvec([[1, 1]], [3, 4]) == APPROX([7.0])


def test_matvec_length_equals_number_of_rows():
    """Длина результата — это fan_out, а не длина входного вектора."""
    assert len(matvec(zero_init(5, 3), [1.0] * 5)) == 3


# ------------------------------------------------------- is_symmetry_broken
def test_symmetry_is_broken_when_rows_differ():
    assert is_symmetry_broken([[1.0, 2.0], [3.0, 4.0]]) is True


def test_symmetry_is_not_broken_when_rows_are_equal():
    assert is_symmetry_broken([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]) is False


def test_single_row_counts_as_broken_symmetry():
    """Один нейрон не с кем сравнивать — вырожденный случай, считаем True."""
    assert is_symmetry_broken([[1.0, 2.0]]) is True


# ----------------------------------------------------- forward_magnitudes
def test_forward_magnitudes_returns_one_number_per_layer():
    assert len(forward_magnitudes(kaiming_init, RELU, n_layers=7)) == 7


def test_forward_magnitudes_is_reproducible():
    a = forward_magnitudes(kaiming_init, RELU, n_layers=5, seed=3)
    b = forward_magnitudes(kaiming_init, RELU, n_layers=5, seed=3)
    assert a == APPROX(b)


def test_kaiming_keeps_the_relu_signal_alive_through_twenty_layers():
    mags = forward_magnitudes(kaiming_init, RELU, n_layers=20)
    assert 0.1 < mags[-1] < 10.0


def test_large_random_scale_explodes_through_relu():
    """Var(w)=1 при ширине 32 умножает дисперсию на 16 за слой."""
    mags = forward_magnitudes(lambda i, o, s: random_init(i, o, 1.0, s), RELU, n_layers=20)
    assert mags[-1] > 1e6


def test_small_random_scale_vanishes_through_relu():
    mags = forward_magnitudes(lambda i, o, s: random_init(i, o, 0.01, s), RELU, n_layers=20)
    assert mags[-1] < 1e-6


def test_xavier_keeps_the_sigmoid_signal_stable():
    mags = forward_magnitudes(xavier_init, SIGMOID, n_layers=20)
    assert 0.3 < mags[-1] < 0.8


def test_xavier_survives_twenty_tanh_layers():
    mags = forward_magnitudes(xavier_init, TANH, n_layers=20)
    assert mags[-1] > 1e-3


def test_zero_init_kills_the_signal_on_the_very_first_layer():
    mags = forward_magnitudes(zero_init, RELU, n_layers=5)
    assert mags == APPROX([0.0] * 5)


def test_every_layer_gets_its_own_weights():
    """Если бы seed не сдвигался по слоям, слои были бы копиями друг друга."""
    mags = forward_magnitudes(kaiming_init, RELU, n_layers=6, seed=0)
    assert len({round(m, 9) for m in mags}) == 6


# ------------------------------------------------------------ recommend_init
def test_recommend_kaiming_for_relu():
    assert recommend_init("relu") == "kaiming"


def test_recommend_kaiming_as_a_heuristic_for_smooth_relu_like_gates():
    assert recommend_init("gelu") == "kaiming"
    assert recommend_init("swish") == "kaiming"
    assert recommend_init("silu") == "kaiming"


def test_recommend_xavier_for_saturating_activations():
    assert recommend_init("sigmoid") == "xavier"
    assert recommend_init("tanh") == "xavier"


def test_recommend_ignores_letter_case():
    assert recommend_init("ReLU") == "kaiming"


def test_recommend_falls_back_to_xavier_for_unknown_activation():
    assert recommend_init("softsign") == "xavier"
