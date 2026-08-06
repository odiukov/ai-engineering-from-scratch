"""Тесты к уроку «Многослойные сети и прямой проход». Правь exercise.py."""

import pytest

from exercise import (
    count_parameters,
    init_network,
    layer_forward,
    layer_shapes,
    network_forward,
    predict_binary,
    sigmoid,
    xor_forward,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

XOR_DATA = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in matrix for x in row]


# ---------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_stays_inside_zero_and_one():
    assert 0.0 < sigmoid(-8.0) < sigmoid(8.0) < 1.0


def test_sigmoid_is_symmetric_around_zero():
    assert sigmoid(3.0) + sigmoid(-3.0) == APPROX(1.0)


def test_sigmoid_survives_huge_negative_input():
    """Наивная 1/(1+exp(-z)) на z = -1000 падает с OverflowError."""
    assert sigmoid(-1000.0) == APPROX(0.0)


def test_sigmoid_survives_huge_positive_input():
    assert sigmoid(1000.0) == APPROX(1.0)


# ---------------------------------------------------------- layer_forward
def test_layer_forward_returns_one_value_per_neuron():
    out = layer_forward([[0.0], [0.0], [0.0]], [0.0, 0.0, 0.0], [7.0])
    assert len(out) == 3


def test_layer_forward_zero_input_gives_half():
    assert layer_forward([[1.0, 1.0]], [0.0], [0.0, 0.0]) == pytest.approx([0.5])


def test_layer_forward_uses_the_bias():
    assert layer_forward([[0.0, 0.0]], [2.0], [5.0, 5.0]) == pytest.approx([sigmoid(2.0)])


def test_layer_forward_neurons_are_independent():
    """Каждая строка весов видит только свой bias и общий вход."""
    out = layer_forward([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [3.0, -3.0])
    assert out == pytest.approx([sigmoid(3.0), sigmoid(-3.0)])


def test_layer_forward_output_is_always_in_the_unit_interval():
    """Даже на z = ±30 сигмоида не долетает до чистых 0 и 1."""
    out = layer_forward([[3.0], [-3.0]], [0.0, 0.0], [10.0])
    assert all(0.0 < v < 1.0 for v in out)


# -------------------------------------------------------- network_forward
def test_network_forward_chains_layers():
    net = [([[1.0, 1.0]], [0.0]), ([[1.0]], [0.0])]
    assert network_forward(net, [0.0, 0.0]) == pytest.approx([sigmoid(0.5)])


def test_network_forward_output_length_matches_last_layer():
    net = [([[1.0, 1.0]] * 4, [0.0] * 4), ([[1.0] * 4] * 2, [0.0, 0.0])]
    assert len(network_forward(net, [0.1, 0.2])) == 2


def test_network_forward_of_one_layer_is_layer_forward():
    weights, biases = [[0.5, -0.5], [1.0, 2.0]], [0.1, -0.1]
    assert network_forward([(weights, biases)], [1.0, 2.0]) == pytest.approx(
        layer_forward(weights, biases, [1.0, 2.0])
    )


# --------------------------------------------------------- predict_binary
def test_predict_binary_above_threshold():
    assert predict_binary(0.73) == 1


def test_predict_binary_below_threshold():
    assert predict_binary(0.12) == 0


def test_predict_binary_at_the_threshold_rounds_up():
    assert predict_binary(0.5) == 1


def test_predict_binary_respects_a_custom_threshold():
    assert predict_binary(0.6, threshold=0.9) == 0


# ------------------------------------------------------------ xor_forward
def test_xor_forward_matches_the_truth_table():
    assert [predict_binary(xor_forward(a, b)) for (a, b), _ in XOR_DATA] == [0, 1, 1, 0]


def test_xor_forward_is_saturated():
    """Веса по 20 гонят сигмоиду в углы: ответы почти 0 и почти 1."""
    for (a, b), target in XOR_DATA:
        assert abs(xor_forward(a, b) - target) < 0.01


def test_xor_forward_is_symmetric():
    assert xor_forward(0, 1) == APPROX(xor_forward(1, 0))


def test_xor_forward_never_leaves_the_unit_interval():
    assert all(0.0 < xor_forward(a, b) < 1.0 for (a, b), _ in XOR_DATA)


# ----------------------------------------------------------- layer_shapes
def test_layer_shapes_reads_the_architecture():
    net = [([[1.0, 1.0]] * 3, [0.0] * 3), ([[1.0, 1.0, 1.0]], [0.0])]
    assert layer_shapes(net) == [(3, 2), (1, 3)]


def test_layer_shapes_of_neighbours_line_up():
    """Число входов слоя k+1 равно числу нейронов слоя k — иначе баг."""
    shapes = layer_shapes(init_network([4, 6, 6, 2], seed=1))
    assert all(a[0] == b[1] for a, b in zip(shapes, shapes[1:]))


# ------------------------------------------------------- count_parameters
def test_count_parameters_small_network():
    assert count_parameters([2, 3, 1]) == 13


def test_count_parameters_mnist_network():
    """784-256-128-10 — классика; 235 тысяч чисел на распознавание цифр."""
    assert count_parameters([784, 256, 128, 10]) == 235146


def test_count_parameters_of_a_single_layer_is_zero():
    """Входной слой ничего не вычисляет, значит и обучать в нём нечего."""
    assert count_parameters([5]) == 0


def test_count_parameters_matches_the_actual_network():
    net = init_network([3, 5, 2], seed=0)
    real = sum(len(flat(w)) + len(b) for w, b in net)
    assert count_parameters([3, 5, 2]) == real


# --------------------------------------------------------- init_network
def test_init_network_builds_the_requested_shapes():
    assert layer_shapes(init_network([2, 3, 1], seed=0)) == [(3, 2), (1, 3)]


def test_init_network_is_reproducible_for_one_seed():
    assert init_network([2, 4, 1], seed=7) == init_network([2, 4, 1], seed=7)


def test_init_network_differs_between_seeds():
    a = flat(init_network([2, 4, 1], seed=1)[0][0])
    b = flat(init_network([2, 4, 1], seed=2)[0][0])
    assert a != b


def test_init_network_weights_are_inside_minus_one_and_one():
    for weights, _ in init_network([3, 4, 2], seed=3):
        assert all(-1.0 <= w <= 1.0 for w in flat(weights))


def test_init_network_starts_with_zero_biases():
    assert all(all(b == 0.0 for b in biases) for _, biases in init_network([3, 4, 2], seed=3))


def test_random_network_still_runs_forward():
    """Случайные веса классифицируют плохо, но прямой проход работает."""
    out = network_forward(init_network([2, 8, 1], seed=42), [0.3, -0.4])
    assert len(out) == 1
    assert 0.0 < out[0] < 1.0
