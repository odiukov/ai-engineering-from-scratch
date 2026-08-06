"""Тесты к уроку «Собственный мини-фреймворк». Правь exercise.py."""

import pytest

from exercise import (
    Layer,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
    mse_grad,
    mse_loss,
    sgd_step,
    train_xor,
    xor_dataset,
    zero_grads,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def numeric_grad(f, values, index, h=1e-6):
    """Численная производная f по values[index] — эталон для backward."""
    original = values[index]
    values[index] = original + h
    up = f()
    values[index] = original - h
    down = f()
    values[index] = original
    return (up - down) / (2 * h)


# ------------------------------------------------------------------- Layer
def test_layer_without_weights_has_no_parameters():
    assert Layer().parameters() == []


def test_layer_forward_is_not_implemented():
    """Базовый класс — это контракт, а не рабочий слой.

    Проверяется КОНТРАСТ: база обязана отказаться считать, а рабочий
    подкласс обязан посчитать. Одной первой половины мало — она проходит
    и на пустой заготовке, где ещё ничего не написано.
    """
    with pytest.raises(NotImplementedError):
        Layer().forward([1.0])
    assert ReLU().forward([-1.0, 2.0]) == [0.0, 2.0]


# ------------------------------------------------------------------ Linear
def test_linear_output_length_is_out_features():
    assert len(Linear(2, 3, seed=0).forward([1.0, 2.0])) == 3


def test_linear_is_reproducible_for_the_same_seed():
    a = Linear(3, 4, seed=5).forward([1.0, 2.0, 3.0])
    b = Linear(3, 4, seed=5).forward([1.0, 2.0, 3.0])
    assert a == APPROX(b)


def test_linear_starts_with_zero_biases():
    """Смещения нулевые: симметрию ломают веса, а не они."""
    assert Linear(2, 3, seed=0).forward([0.0, 0.0]) == APPROX([0.0, 0.0, 0.0])


def test_linear_parameter_count_is_weights_plus_biases():
    assert len(Linear(2, 3, seed=0).parameters()) == 2 * 3 + 3


def test_linear_is_affine_in_its_input():
    """f(2x) = 2f(x) при нулевых смещениях — слой действительно линейный."""
    layer = Linear(2, 2, seed=1)
    single = layer.forward([1.0, -1.0])
    double = layer.forward([2.0, -2.0])
    assert double == pytest.approx([2 * v for v in single], abs=1e-9)


def test_linear_weight_gradients_match_numeric_derivative():
    layer = Linear(2, 2, seed=2)
    x = [0.7, -1.3]

    def loss():
        return sum(layer.forward(x))

    layer.forward(x)
    layer.backward([1.0, 1.0])
    values, index, grads = layer.parameters()[0]
    assert grads[index] == pytest.approx(numeric_grad(loss, values, index), abs=1e-5)


def test_linear_accumulates_gradients_until_zeroed():
    """Два прохода без zero_grads дают удвоенный градиент — не баг, а фича."""
    layer = Linear(2, 1, seed=3)
    x = [1.0, 2.0]
    layer.forward(x)
    layer.backward([1.0])
    once = layer.weight_grads[0][0]
    layer.forward(x)
    layer.backward([1.0])
    assert layer.weight_grads[0][0] == pytest.approx(2 * once)


# -------------------------------------------------------------------- ReLU
def test_relu_blocks_negative_and_passes_positive():
    assert ReLU().forward([-1.0, 2.0]) == APPROX([0.0, 2.0])


def test_relu_backward_uses_the_mask_from_forward():
    layer = ReLU()
    layer.forward([-1.0, 2.0])
    assert layer.backward([5.0, 5.0]) == APPROX([0.0, 5.0])


def test_relu_has_no_trainable_parameters():
    assert ReLU().parameters() == []


# ----------------------------------------------------------------- Sigmoid
def test_sigmoid_at_zero_is_half():
    assert Sigmoid().forward([0.0]) == APPROX([0.5])


def test_sigmoid_gradient_peaks_at_a_quarter():
    layer = Sigmoid()
    layer.forward([0.0])
    assert layer.backward([1.0]) == APPROX([0.25])


def test_sigmoid_survives_extreme_inputs():
    """Наивный math.exp(-x) на x = -1000 падает с OverflowError."""
    out = Sigmoid().forward([-1000.0, 1000.0])
    assert out == pytest.approx([0.0, 1.0], abs=1e-9)


# -------------------------------------------------------------- Sequential
def test_sequential_chains_shapes_left_to_right():
    model = Sequential(Linear(2, 4, seed=0), ReLU(), Linear(4, 1, seed=1))
    assert len(model.forward([1.0, 1.0])) == 1


def test_sequential_collects_parameters_of_all_layers():
    model = Sequential(Linear(2, 3, seed=0), ReLU(), Linear(3, 1, seed=1))
    assert len(model.parameters()) == (2 * 3 + 3) + (3 * 1 + 1)


def test_sequential_is_itself_a_layer_and_nests():
    """Композит: Sequential внутри Sequential работает без единой правки."""
    inner = Sequential(Linear(2, 3, seed=0), ReLU())
    outer = Sequential(inner, Linear(3, 1, seed=1))
    assert len(outer.forward([1.0, 1.0])) == 1
    assert len(outer.parameters()) == len(inner.parameters()) + 4


def test_sequential_backward_returns_gradient_of_the_input():
    model = Sequential(Linear(2, 3, seed=0), ReLU(), Linear(3, 1, seed=1))
    model.forward([0.5, -0.5])
    assert len(model.backward([1.0])) == 2


def test_sequential_gradients_match_numeric_derivative():
    model = Sequential(Linear(2, 3, seed=4), ReLU(), Linear(3, 1, seed=5), Sigmoid())
    x, target = [0.6, -0.2], [1.0]
    params = model.parameters()

    def loss():
        return mse_loss(model.forward(x), target)

    zero_grads(params)
    predicted = model.forward(x)
    model.backward(mse_grad(predicted, target))
    values, index, grads = params[0]
    assert grads[index] == pytest.approx(numeric_grad(loss, values, index), abs=1e-5)


# ---------------------------------------------------------- loss и оптимизатор
def test_mse_loss_of_a_perfect_prediction_is_zero():
    assert mse_loss([1.0, 2.0], [1.0, 2.0]) == APPROX(0.0)


def test_mse_loss_averages_over_the_outputs():
    assert mse_loss([2.0, 0.0], [0.0, 0.0]) == APPROX(2.0)


def test_mse_grad_is_the_derivative_of_mse_loss():
    predicted, target = [2.0, 0.0], [0.0, 0.0]
    assert mse_grad(predicted, target) == APPROX([2.0, 0.0])


def test_mse_grad_is_zero_at_the_minimum():
    assert mse_grad([1.0, 2.0], [1.0, 2.0]) == APPROX([0.0, 0.0])


def test_sgd_step_moves_against_the_gradient():
    values, grads = [1.0], [2.0]
    sgd_step([(values, 0, grads)], lr=0.1)
    assert values[0] == APPROX(0.8)


def test_zero_grads_clears_accumulated_gradients():
    layer = Linear(2, 1, seed=0)
    layer.forward([1.0, 1.0])
    layer.backward([1.0])
    zero_grads(layer.parameters())
    assert layer.weight_grads[0] == APPROX([0.0, 0.0])


def test_sgd_step_does_nothing_when_gradients_are_zero():
    layer = Linear(2, 2, seed=0)
    before = layer.forward([1.0, 1.0])
    zero_grads(layer.parameters())
    sgd_step(layer.parameters(), lr=10.0)
    assert layer.forward([1.0, 1.0]) == APPROX(before)


# ------------------------------------------------------------------- XOR
def test_xor_dataset_has_four_samples():
    assert len(xor_dataset()) == 4


def test_xor_labels_are_the_exclusive_or():
    for x, target in xor_dataset():
        assert target[0] == float(int(x[0]) ^ int(x[1]))


def test_trained_network_solves_xor():
    """Главная проверка урока: собранный фреймворк реально учит сеть."""
    model, _ = train_xor()
    predictions = [model.forward(x)[0] for x, _ in xor_dataset()]
    assert predictions[0] < 0.5
    assert predictions[1] > 0.5
    assert predictions[2] > 0.5
    assert predictions[3] < 0.5


def test_xor_final_loss_is_small():
    _, loss = train_xor()
    assert loss < 0.01


def test_xor_training_is_reproducible():
    _, first = train_xor()
    _, second = train_xor()
    assert first == APPROX(second)


def test_a_single_linear_layer_cannot_solve_xor():
    """Без скрытого слоя с нелинейностью XOR не берётся — отсюда и вся сеть."""
    model = Sequential(Linear(2, 1, seed=0), Sigmoid())
    params = model.parameters()
    for _ in range(4000):
        for x, target in xor_dataset():
            zero_grads(params)
            predicted = model.forward(x)
            model.backward(mse_grad(predicted, target))
            sgd_step(params, 0.3)
    predictions = [model.forward(x)[0] for x, _ in xor_dataset()]
    assert not (predictions[1] > 0.5 and predictions[3] < 0.5)
