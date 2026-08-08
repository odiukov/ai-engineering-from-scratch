"""Тесты к уроку «Backpropagation с нуля». Правь exercise.py."""

import copy
import math

import pytest

from exercise import (
    backward,
    forward,
    init_params,
    loss_for_params,
    numeric_gradient,
    sgd_step,
    sigmoid,
    train_xor,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

XOR_DATA = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


def flat(grads):
    """Все числа словаря градиентов одним плоским списком, в фиксированном порядке."""
    out = []
    for row in grads["w1"]:
        out.extend(row)
    out.extend(grads["b1"])
    out.extend(grads["w2"])
    out.append(grads["b2"])
    return out


def central_difference(params, x, target, h=1e-5):
    """Независимая от exercise.py численная производная — эталон для сверки.

    Строится только на loss_for_params, поэтому проверяет именно backward,
    а не сравнивает две одинаково неверные реализации между собой.
    """
    work = copy.deepcopy(params)

    def d(get, put):
        base = get()
        put(base + h)
        up = loss_for_params(work, x, target)
        put(base - h)
        down = loss_for_params(work, x, target)
        put(base)
        return (up - down) / (2.0 * h)

    out = []
    for i, row in enumerate(work["w1"]):
        for j in range(len(row)):
            out.append(
                d(
                    lambda i=i, j=j: work["w1"][i][j],
                    lambda v, i=i, j=j: work["w1"][i].__setitem__(j, v),
                )
            )
    for i in range(len(work["b1"])):
        out.append(
            d(lambda i=i: work["b1"][i], lambda v, i=i: work["b1"].__setitem__(i, v))
        )
    for i in range(len(work["w2"])):
        out.append(
            d(lambda i=i: work["w2"][i], lambda v, i=i: work["w2"].__setitem__(i, v))
        )
    out.append(d(lambda: work["b2"], lambda v: work.__setitem__("b2", v)))
    return out


# ---------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_survives_huge_inputs():
    assert sigmoid(-1000.0) == APPROX(0.0)
    assert sigmoid(1000.0) == APPROX(1.0)


def test_sigmoid_derivative_identity_holds():
    """s'(z) = s(z)*(1 - s(z)) — на этом равенстве стоит весь backward."""
    h = 1e-6
    numeric = (sigmoid(0.7 + h) - sigmoid(0.7 - h)) / (2 * h)
    s = sigmoid(0.7)
    assert s * (1 - s) == pytest.approx(numeric, abs=1e-7)


# ------------------------------------------------------------ init_params
def test_init_params_has_the_expected_shapes():
    p = init_params(2, 4, seed=0)
    assert len(p["w1"]) == 4
    assert all(len(row) == 2 for row in p["w1"])
    assert len(p["b1"]) == 4
    assert len(p["w2"]) == 4
    assert isinstance(p["b2"], float)


def test_init_params_is_reproducible_for_one_seed():
    assert init_params(2, 4, seed=7) == init_params(2, 4, seed=7)


def test_init_params_differs_between_seeds():
    assert init_params(2, 4, seed=1)["w1"] != init_params(2, 4, seed=2)["w1"]


def test_init_params_starts_biases_at_zero():
    p = init_params(3, 5, seed=0)
    assert p["b1"] == [0.0] * 5
    assert p["b2"] == APPROX(0.0)


def test_init_params_scale_shrinks_with_more_inputs():
    """scale = sqrt(2/n_inputs): чем шире вход, тем мельче веса."""
    wide = max(abs(w) for row in init_params(200, 4, seed=0)["w1"] for w in row)
    narrow = max(abs(w) for row in init_params(2, 4, seed=0)["w1"] for w in row)
    assert wide < narrow


def test_output_weights_use_hidden_width_as_their_fan_in():
    """w2 получает n_hidden входов, поэтому его масштаб не зависит от n_inputs."""
    p = init_params(1, 200, seed=0)
    output_bound = math.sqrt(2.0 / 200)
    assert max(abs(w) for w in p["w2"]) <= output_bound


# ---------------------------------------------------------------- forward
def test_forward_returns_the_full_cache():
    p = {"w1": [[1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    cache = forward(p, [0.0])
    assert set(cache) == {"z1", "a1", "z2", "a2"}


def test_forward_worked_example():
    p = {"w1": [[1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    cache = forward(p, [0.0])
    assert cache["a1"] == pytest.approx([0.5])
    assert cache["a2"] == pytest.approx(sigmoid(0.5))


def test_forward_activations_stay_inside_zero_and_one():
    p = init_params(2, 6, seed=3)
    cache = forward(p, [4.0, -4.0])
    assert all(0.0 < a < 1.0 for a in cache["a1"])
    assert 0.0 < cache["a2"] < 1.0


def test_forward_z1_is_linear_before_the_activation():
    """z1 — это ещё W*x + b, без сигмоиды."""
    p = {"w1": [[2.0, 3.0]], "b1": [-1.0], "w2": [0.0], "b2": 0.0}
    assert forward(p, [1.0, 1.0])["z1"] == pytest.approx([4.0])


# --------------------------------------------------------- loss_for_params
def test_loss_is_zero_on_a_perfect_prediction():
    p = {"w1": [[1.0]], "b1": [0.0], "w2": [0.0], "b2": 0.0}
    assert loss_for_params(p, [0.0], 0.5) == APPROX(0.0)


def test_loss_is_positive_when_prediction_is_off():
    p = {"w1": [[1.0]], "b1": [0.0], "w2": [0.0], "b2": 0.0}
    assert loss_for_params(p, [0.0], 1.0) == APPROX(0.25)


def test_loss_grows_quadratically_with_the_error():
    p = {"w1": [[1.0]], "b1": [0.0], "w2": [0.0], "b2": 0.0}
    assert loss_for_params(p, [0.0], 1.5) == APPROX(4 * loss_for_params(p, [0.0], 1.0))


# --------------------------------------------------------------- backward
def test_backward_has_the_same_shape_as_params():
    p = init_params(2, 4, seed=0)
    g = backward(p, [0.3, -0.7], 1.0)
    assert len(g["w1"]) == len(p["w1"])
    assert all(len(a) == len(b) for a, b in zip(g["w1"], p["w1"]))
    assert len(g["b1"]) == len(p["b1"])
    assert len(g["w2"]) == len(p["w2"])


def test_backward_matches_the_numeric_gradient():
    """Главная проверка урока: аналитика против центральной разности."""
    p = init_params(2, 4, seed=0)
    x, target = [0.6, -0.9], 1.0
    assert flat(backward(p, x, target)) == pytest.approx(
        central_difference(p, x, target), abs=1e-6
    )


def test_backward_matches_numeric_gradient_on_every_xor_sample():
    p = init_params(2, 5, seed=11)
    for x, target in XOR_DATA:
        assert flat(backward(p, x, target)) == pytest.approx(
            central_difference(p, x, target), abs=1e-6
        ), f"градиенты разошлись на примере {x}"


def test_backward_matches_numeric_gradient_after_training():
    """Обученная сеть сидит в насыщении сигмоиды — самое злое место для backward."""
    params, _ = train_xor(seed=0, epochs=300)
    for x, target in XOR_DATA:
        assert flat(backward(params, x, target)) == pytest.approx(
            central_difference(params, x, target), abs=1e-6
        )


def test_backward_matches_numeric_gradient_for_a_wide_hidden_layer():
    p = init_params(3, 12, seed=5)
    x, target = [1.2, -0.4, 0.8], 0.0
    assert flat(backward(p, x, target)) == pytest.approx(
        central_difference(p, x, target), abs=1e-6
    )


def test_backward_gradient_of_bias_equals_gradient_of_z():
    """dL/db2 = dL/dz2, потому что z2 = ... + b2 и производная суммы по b равна 1."""
    p = init_params(2, 3, seed=2)
    g = backward(p, [1.0, 1.0], 1.0)
    a1 = forward(p, [1.0, 1.0])["a1"]
    assert g["w2"] == pytest.approx([g["b2"] * a for a in a1])


def test_backward_gradient_is_zero_for_a_zero_input_weight():
    """Вход, равный нулю, не участвовал в ответе — его вес и не двигается."""
    p = init_params(2, 3, seed=4)
    g = backward(p, [0.0, 1.0], 1.0)
    assert all(row[0] == APPROX(0.0) for row in g["w1"])


def test_backward_gradient_flips_sign_with_the_error():
    """Недооценили цель — градиент в одну сторону, переоценили — в другую."""
    p = init_params(2, 3, seed=6)
    low = backward(p, [0.5, 0.5], 1.0)["b2"]
    high = backward(p, [0.5, 0.5], 0.0)["b2"]
    assert low < 0 < high


def test_backward_does_not_touch_the_params():
    p = init_params(2, 4, seed=0)
    before = copy.deepcopy(p)
    backward(p, [0.3, 0.4], 1.0)
    assert p == before


# -------------------------------------------------------- numeric_gradient
def test_numeric_gradient_matches_backward():
    p = init_params(2, 4, seed=8)
    assert flat(numeric_gradient(p, [0.4, 0.9], 1.0)) == pytest.approx(
        flat(backward(p, [0.4, 0.9], 1.0)), abs=1e-6
    )


def test_numeric_gradient_restores_the_params_exactly():
    """Параметр возвращаем сохранённым значением, а не вторым сложением."""
    p = init_params(2, 4, seed=9)
    before = copy.deepcopy(p)
    numeric_gradient(p, [0.4, 0.9], 1.0)
    assert p == before


def test_numeric_gradient_has_the_same_shape_as_backward():
    p = init_params(3, 4, seed=1)
    assert len(flat(numeric_gradient(p, [1.0, 2.0, 3.0], 0.0))) == len(
        flat(backward(p, [1.0, 2.0, 3.0], 0.0))
    )


# ---------------------------------------------------------------- sgd_step
def test_sgd_step_worked_example():
    p = {"w1": [[1.0]], "b1": [0.0], "w2": [1.0], "b2": 0.0}
    g = {"w1": [[2.0]], "b1": [1.0], "w2": [0.0], "b2": -4.0}
    new = sgd_step(p, g, 0.5)
    assert new["w1"][0] == pytest.approx([0.0])
    assert new["b1"] == pytest.approx([-0.5])
    assert new["w2"] == pytest.approx([1.0])
    assert new["b2"] == APPROX(2.0)


def test_sgd_step_leaves_the_original_params_alone():
    p = init_params(2, 3, seed=0)
    before = copy.deepcopy(p)
    sgd_step(p, backward(p, [1.0, 0.0], 1.0), 0.5)
    assert p == before


def test_sgd_step_lowers_the_loss():
    """Шаг против градиента обязан уменьшить loss на этом примере."""
    p = init_params(2, 4, seed=0)
    x, target = [1.0, 0.0], 1.0
    after = sgd_step(p, backward(p, x, target), 0.5)
    assert loss_for_params(after, x, target) < loss_for_params(p, x, target)


def test_sgd_step_with_zero_learning_rate_changes_nothing():
    p = init_params(2, 3, seed=0)
    assert sgd_step(p, backward(p, [1.0, 1.0], 1.0), 0.0) == p


# --------------------------------------------------------------- train_xor
def test_train_xor_reaches_a_small_loss():
    _, loss = train_xor()
    assert loss < 0.05


def test_train_xor_answers_the_truth_table():
    params, _ = train_xor()
    answers = [1 if forward(params, x)["a2"] >= 0.5 else 0 for x, _ in XOR_DATA]
    assert answers == [0, 1, 1, 0]


def test_train_xor_is_reproducible():
    assert train_xor(seed=1, epochs=200)[1] == APPROX(train_xor(seed=1, epochs=200)[1])


def test_training_longer_does_not_make_it_worse():
    assert train_xor(seed=0, epochs=1000)[1] < train_xor(seed=0, epochs=50)[1]
