"""Тесты к уроку «Перцептрон». Правь exercise.py."""

import pytest

from exercise import (
    accuracy,
    perceptron_converged,
    perceptron_output,
    step,
    train_perceptron,
    update_once,
    xor_network,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

AND_DATA = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
OR_DATA = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)]
NAND_DATA = [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
NOT_DATA = [([0], 1), ([1], 0)]
XOR_DATA = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
MAJORITY_DATA = [
    ([a, b, c], 1 if a + b + c >= 2 else 0)
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
]


# ------------------------------------------------------------------- step
def test_step_fires_on_positive():
    assert step(2.5) == 1


def test_step_stays_silent_on_negative():
    assert step(-0.001) == 0


def test_step_counts_zero_as_fired():
    """Граница принадлежит положительному классу: `z >= 0`, не `z > 0`."""
    assert step(0.0) == 1


# ------------------------------------------------------ perceptron_output
def test_perceptron_output_is_weighted_sum_through_step():
    assert perceptron_output([1.0, 1.0], -1.5, [1, 1]) == 1


def test_perceptron_bias_shifts_the_boundary():
    """Те же веса, но bias -1.5 требует обоих входов — это AND."""
    assert perceptron_output([1.0, 1.0], -1.5, [1, 0]) == 0
    assert perceptron_output([1.0, 1.0], -0.5, [1, 0]) == 1


def test_perceptron_can_fire_on_all_zero_input():
    """Без bias граница обязана проходить через ноль — вот зачем он нужен."""
    assert perceptron_output([0.0, 0.0], 0.5, [0, 0]) == 1


def test_perceptron_negative_weights_invert_the_answer():
    assert perceptron_output([-1.0, -1.0], 1.5, [1, 1]) == 0


# ------------------------------------------------------------ update_once
def test_update_does_nothing_when_prediction_is_correct():
    """error = 0 — перцептрон учится только на своих ошибках."""
    weights, bias = update_once([0.0, 0.0], 0.0, [1, 1], 1, 0.1)
    assert weights == pytest.approx([0.0, 0.0])
    assert bias == APPROX(0.0)


def test_update_raises_weights_when_output_is_too_low():
    weights, bias = update_once([0.0, 0.0], -0.5, [1, 1], 1, 0.1)
    assert weights == pytest.approx([0.1, 0.1])
    assert bias == APPROX(-0.4)


def test_update_lowers_weights_when_output_is_too_high():
    weights, bias = update_once([1.0, 1.0], 0.0, [1, 1], 0, 0.1)
    assert weights == pytest.approx([0.9, 0.9])
    assert bias == APPROX(-0.1)


def test_update_leaves_weights_of_zero_inputs_alone():
    """Вход с нулём не участвовал в ответе — его вес двигать не за что."""
    weights, _ = update_once([1.0, 1.0], 0.0, [1, 0], 0, 0.5)
    assert weights == pytest.approx([0.5, 1.0])


def test_update_moves_bias_even_on_zero_input():
    _, bias = update_once([0.0, 0.0], 0.0, [0, 0], 0, 0.1)
    assert bias == APPROX(-0.1)


# -------------------------------------------------------- train_perceptron
def test_trained_perceptron_solves_and():
    weights, bias = train_perceptron(AND_DATA)
    assert accuracy(weights, bias, AND_DATA) == APPROX(1.0)


def test_trained_perceptron_solves_or():
    weights, bias = train_perceptron(OR_DATA)
    assert accuracy(weights, bias, OR_DATA) == APPROX(1.0)


def test_trained_perceptron_solves_nand():
    weights, bias = train_perceptron(NAND_DATA)
    assert accuracy(weights, bias, NAND_DATA) == APPROX(1.0)


def test_trained_perceptron_solves_not_with_a_single_input():
    weights, bias = train_perceptron(NOT_DATA)
    assert len(weights) == 1
    assert accuracy(weights, bias, NOT_DATA) == APPROX(1.0)


def test_training_is_deterministic():
    """Старт с нулей: два прогона обязаны дать одни и те же веса."""
    assert train_perceptron(AND_DATA) == train_perceptron(AND_DATA)


def test_perceptron_never_solves_xor():
    """Хоть тысяча эпох — одна прямая XOR не разделит."""
    weights, bias = train_perceptron(XOR_DATA, epochs=1000)
    assert accuracy(weights, bias, XOR_DATA) < 1.0


# ---------------------------------------------------------------- accuracy
def test_accuracy_of_always_one_perceptron():
    """Нулевые веса и step(0) = 1: угадана только строка [1, 1]."""
    assert accuracy([0.0, 0.0], 0.0, AND_DATA) == APPROX(0.25)


def test_accuracy_is_one_for_correct_weights():
    assert accuracy([1.0, 1.0], -1.5, AND_DATA) == APPROX(1.0)


# ---------------------------------------------------- perceptron_converged
def test_perceptron_converges_on_and():
    assert perceptron_converged(AND_DATA) is True


def test_perceptron_converges_on_or():
    assert perceptron_converged(OR_DATA) is True


def test_perceptron_does_not_converge_on_xor_within_the_budget():
    assert perceptron_converged(XOR_DATA) is False


def test_perceptron_converges_on_majority_of_three():
    """«Хотя бы два из трёх» — это порог на сумме входов, значит прямая есть."""
    assert perceptron_converged(MAJORITY_DATA) is True


def test_timeout_does_not_disprove_separability_for_a_tiny_margin():
    """Близкие точки разделимы, но фиксированного малого бюджета не хватает."""
    hard_pair = [([1.0], 0), ([1.01], 1)]
    assert accuracy([1.0], -1.005, hard_pair) == APPROX(1.0)
    assert perceptron_converged(hard_pair, epochs=200) is False
    assert perceptron_converged(hard_pair, epochs=1000) is True


# ------------------------------------------------------------ xor_network
def test_xor_network_matches_the_truth_table():
    assert [xor_network(a, b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))] == [0, 1, 1, 0]


def test_xor_network_is_symmetric():
    """XOR не различает порядок аргументов."""
    assert xor_network(0, 1) == xor_network(1, 0)


def test_xor_network_beats_any_single_perceptron():
    """Два слоя дают 100% там, где один не дотягивает."""
    weights, bias = train_perceptron(XOR_DATA, epochs=1000)
    single = accuracy(weights, bias, XOR_DATA)
    two_layer = sum(1 for x, t in XOR_DATA if xor_network(x[0], x[1]) == t) / 4
    assert two_layer == APPROX(1.0)
    assert two_layer > single
