"""Тесты к уроку «Цепное правило и автодифференцирование». Правь exercise.py."""

import math

import pytest

from exercise import (
    chain,
    chain_many,
    d_relu,
    d_sigmoid,
    forward_backward,
    relu,
    sigmoid,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------------ chain
def test_chain_multiplies():
    assert chain(2.0, 3.0) == APPROX(6.0)


def test_chain_with_zero_kills_the_gradient():
    """Один нулевой множитель обнуляет всю цепочку — так умирают нейроны."""
    assert chain(0.0, 999.0) == APPROX(0.0)


# ------------------------------------------------------------- chain_many
def test_chain_many_basic():
    assert chain_many([2.0, 3.0, 0.5]) == APPROX(3.0)


def test_chain_many_of_empty_is_one():
    """Пустое произведение равно единице, а не нулю."""
    assert chain_many([]) == APPROX(1.0)


def test_chain_many_shows_vanishing_gradient():
    """Двадцать слоёв по 0.5 — от градиента остаётся миллионная доля."""
    assert chain_many([0.5] * 20) < 1e-6


def test_chain_many_shows_exploding_gradient():
    """И наоборот: множители чуть больше единицы разносят градиент."""
    assert chain_many([1.5] * 20) > 1000


# ---------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0) == APPROX(0.5)


def test_sigmoid_is_bounded():
    assert 0 < sigmoid(-10) < sigmoid(10) < 1


def test_sigmoid_survives_large_negative_input():
    """Наивная формула 1/(1+exp(-x)) падает с OverflowError на x = -1000."""
    assert sigmoid(-1000) == APPROX(0.0)


def test_sigmoid_survives_large_positive_input():
    assert sigmoid(1000) == APPROX(1.0)


def test_sigmoid_is_symmetric_around_zero():
    assert sigmoid(2) + sigmoid(-2) == APPROX(1.0)


# -------------------------------------------------------------- d_sigmoid
def test_d_sigmoid_peaks_at_zero():
    assert d_sigmoid(0) == APPROX(0.25)


def test_d_sigmoid_never_exceeds_a_quarter():
    """Потолок 0.25 — причина затухания градиента в сигмоидных сетях."""
    assert all(d_sigmoid(x) <= 0.25 + 1e-12 for x in (-5, -1, 0, 1, 5))


def test_d_sigmoid_matches_numeric_derivative():
    h = 1e-6
    numeric = (sigmoid(1.0 + h) - sigmoid(1.0 - h)) / (2 * h)
    assert d_sigmoid(1.0) == pytest.approx(numeric, abs=1e-6)


# ------------------------------------------------------------- relu/d_relu
def test_relu_passes_positive():
    assert relu(3) == APPROX(3)


def test_relu_blocks_negative():
    assert relu(-3) == APPROX(0)


def test_d_relu_is_one_where_it_grows():
    assert d_relu(3) == APPROX(1.0)


def test_d_relu_is_zero_where_it_is_flat():
    assert d_relu(-3) == APPROX(0.0)


def test_d_relu_at_zero_is_zero_by_convention():
    assert d_relu(0) == APPROX(0.0)


def test_relu_gradient_beats_sigmoid_gradient():
    """Единица против четверти — почему ReLU вытеснила сигмоиду."""
    assert d_relu(1.0) > d_sigmoid(1.0)


# ------------------------------------------------------- forward_backward
def test_forward_backward_worked_example():
    assert forward_backward(2.0, 3.0, 4.0) == pytest.approx((24.0, 8.0, 6.0))


def test_forward_backward_blocked_by_relu():
    """Отрицательный z обнуляет и выход, и градиент по w1."""
    y, dw1, dw2 = forward_backward(2.0, -3.0, 4.0)
    assert (y, dw1, dw2) == pytest.approx((0.0, 0.0, 0.0))


def test_forward_backward_matches_numeric_gradient():
    h = 1e-6
    x, w1, w2 = 2.0, 3.0, 4.0
    num_w1 = (forward_backward(x, w1 + h, w2)[0] - forward_backward(x, w1 - h, w2)[0]) / (2 * h)
    assert forward_backward(x, w1, w2)[1] == pytest.approx(num_w1, abs=1e-4)
