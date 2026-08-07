"""Тесты к уроку «Gradient checkpointing и пересчёт активаций». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    backward_checkpointed,
    backward_store_all,
    checkpoint_budget,
    forward_checkpointed,
    forward_store_all,
    layer_backward,
    layer_forward,
    optimal_segment,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# слой из разбора в docstring: вторая координата гасится relu
DEMO_LAYER = ([2.0, 1.0], [0.0, -3.0], [3.0, 1.0], [1.0, 0.0])

EPS = 1e-6  # шаг центральной разности


def flat(nested):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    out = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            out.extend(flat(item))
        else:
            out.append(item)
    return out


def net(n_layers, width, seed, bias_low):
    """Игрушечная сеть. bias_low > 0 держит relu подальше от излома."""
    rng = random.Random(seed)
    return [
        (
            [rng.uniform(0.6, 1.2) for _ in range(width)],
            [rng.uniform(bias_low, 0.6) for _ in range(width)],
            [rng.uniform(0.6, 1.2) for _ in range(width)],
            [rng.uniform(-0.2, 0.2) for _ in range(width)],
        )
        for _ in range(n_layers)
    ]


def smooth_net():
    """Сеть без погашенных relu: на ней численная производная корректна."""
    return net(5, 3, seed=1, bias_low=0.2), [0.5, 0.4, 0.9]


def mixed_net():
    """Сеть, где часть координат погашена relu."""
    return net(6, 4, seed=0, bias_low=-0.5), [0.5, -0.3, 0.9, 0.2]


def bump(params, layer_index, slot, j, delta):
    """Копия params с одним сдвинутым весом. Оригинал не трогаем."""
    changed = [tuple(list(vec) for vec in layer) for layer in params]
    changed[layer_index][slot][j] += delta
    return changed


def numeric_param_grad(params, x, layer_index, slot, j):
    """Центральная разность d(sum(y)) / d(параметр)."""
    up = sum(forward_store_all(x, bump(params, layer_index, slot, j, EPS))[0])
    down = sum(forward_store_all(x, bump(params, layer_index, slot, j, -EPS))[0])
    return (up - down) / (2 * EPS)


# -------------------------------------------------------------- layer_forward
def test_layer_forward_worked_example():
    assert layer_forward([1.0, 1.0], DEMO_LAYER) == APPROX([7.0, 0.0])


def test_relu_leaves_only_the_output_bias():
    """Погашенная координата отдаёт чистый b2, вход на неё не влияет."""
    assert layer_forward([1.0, 99.0], ([1.0, 0.0], [0.0, -1.0], [1.0, 5.0], [0.0, 0.7]))[1] == APPROX(0.7)


def test_layer_forward_does_not_mutate_its_input():
    x = [1.0, 1.0]
    layer_forward(x, DEMO_LAYER)
    assert x == [1.0, 1.0]


# ------------------------------------------------------------- layer_backward
def test_layer_backward_worked_example():
    grad_x, grads = layer_backward([1.0, 1.0], DEMO_LAYER, [1.0, 1.0])
    assert grad_x == APPROX([6.0, 0.0])
    assert flat(grads) == APPROX(flat(([3.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 1.0])))


def test_a_blocked_relu_passes_no_gradient():
    """Всё, что до relu, получает ноль: w1 и b1 этой координаты не учатся."""
    _, (gw1, gb1, _, _) = layer_backward([1.0, 1.0], DEMO_LAYER, [1.0, 1.0])
    assert (gw1[1], gb1[1]) == APPROX((0.0, 0.0))


def test_output_bias_gradient_is_the_incoming_gradient():
    _, (_, _, _, gb2) = layer_backward([1.0, 1.0], DEMO_LAYER, [0.3, -0.7])
    assert gb2 == APPROX([0.3, -0.7])


def test_layer_backward_matches_the_numeric_derivative():
    params, x = smooth_net()
    _, grads = layer_backward(x, params[0], [1.0] * 3)
    single = [params[0]]
    for slot in range(4):
        for j in range(3):
            numeric = numeric_param_grad(single, x, 0, slot, j)
            assert grads[slot][j] == pytest.approx(numeric, abs=1e-6)


# ---------------------------------------------------------- forward_store_all
def test_store_all_keeps_one_activation_per_layer_boundary():
    params, x = mixed_net()
    _, activations = forward_store_all(x, params)
    assert len(activations) == len(params) + 1


def test_first_activation_is_the_input_and_last_is_the_output():
    params, x = mixed_net()
    out, activations = forward_store_all(x, params)
    assert activations[0] == APPROX(x)
    assert activations[-1] == APPROX(out)


def test_store_all_agrees_with_layer_by_layer_chaining():
    params, x = mixed_net()
    manual = x
    for layer in params:
        manual = layer_forward(manual, layer)
    assert forward_store_all(x, params)[0] == APPROX(manual)


# --------------------------------------------------------- backward_store_all
def test_parameter_gradients_match_the_numeric_derivative():
    """Опорная точка урока: аналитический backward сверен с разностью."""
    params, x = smooth_net()
    _, activations = forward_store_all(x, params)
    _, grads = backward_store_all([1.0] * 3, activations, params)
    for layer_index in range(len(params)):
        for slot in range(4):
            for j in range(3):
                numeric = numeric_param_grad(params, x, layer_index, slot, j)
                assert grads[layer_index][slot][j] == pytest.approx(numeric, abs=1e-6)


def test_input_gradient_matches_the_numeric_derivative():
    params, x = smooth_net()
    _, activations = forward_store_all(x, params)
    grad_x, _ = backward_store_all([1.0] * 3, activations, params)
    for j in range(3):
        up, down = list(x), list(x)
        up[j] += EPS
        down[j] -= EPS
        numeric = (sum(forward_store_all(up, params)[0]) - sum(forward_store_all(down, params)[0])) / (2 * EPS)
        assert grad_x[j] == pytest.approx(numeric, abs=1e-6)


def test_gradients_come_back_in_layer_order_not_traversal_order():
    """gb2 равен входящему градиенту у ПОСЛЕДНЕГО слоя. Перевернул список —
    единицы окажутся в grads[0], и тест это увидит."""
    params, x = smooth_net()
    _, activations = forward_store_all(x, params)
    _, grads = backward_store_all([1.0] * 3, activations, params)
    assert len(grads) == len(params)
    assert grads[-1][3] == APPROX([1.0, 1.0, 1.0])
    assert grads[0][3] != pytest.approx([1.0, 1.0, 1.0], abs=1e-9)


# ------------------------------------------------------- forward_checkpointed
def test_checkpointing_does_not_change_the_output():
    params, x = mixed_net()
    plain, _ = forward_store_all(x, params)
    saved_out, _ = forward_checkpointed(x, params, 3)
    assert saved_out == plain


def test_saved_count_is_the_segment_count():
    params, x = mixed_net()
    for segment in range(1, len(params) + 1):
        _, saved = forward_checkpointed(x, params, segment)
        assert len(saved) == math.ceil(len(params) / segment)


def test_checkpointing_stores_less_than_storing_everything():
    """Главный выигрыш: 2 сохранённых входа вместо 7 активаций."""
    params, x = mixed_net()
    _, activations = forward_store_all(x, params)
    _, saved = forward_checkpointed(x, params, 3)
    assert len(saved) < len(activations)


def test_segment_one_saves_every_layer_input():
    params, x = mixed_net()
    _, saved = forward_checkpointed(x, params, 1)
    assert len(saved) == len(params)


def test_a_zero_length_segment_is_rejected():
    params, x = mixed_net()
    with pytest.raises(ValueError):
        forward_checkpointed(x, params, 0)


# ------------------------------------------------------ backward_checkpointed
def test_recomputation_gives_exactly_the_same_gradients():
    """Ради этого всё и затевалось: память меняем, точность — нет."""
    params, x = mixed_net()
    _, activations = forward_store_all(x, params)
    reference_gx, reference = backward_store_all([1.0] * 4, activations, params)
    _, saved = forward_checkpointed(x, params, 3)
    grad_x, grads = backward_checkpointed([1.0] * 4, saved, params, 3)
    assert flat(grads) == flat(reference)
    assert grad_x == reference_gx


def test_every_segment_size_gives_the_same_gradients():
    params, x = mixed_net()
    _, activations = forward_store_all(x, params)
    _, reference = backward_store_all([1.0] * 4, activations, params)
    for segment in range(1, len(params) + 1):
        _, saved = forward_checkpointed(x, params, segment)
        _, grads = backward_checkpointed([1.0] * 4, saved, params, segment)
        assert flat(grads) == flat(reference)


def test_a_segment_that_does_not_divide_the_depth_still_works():
    """6 слоёв по 4 — последний сегмент короче, границы не должны разъехаться."""
    params, x = mixed_net()
    _, activations = forward_store_all(x, params)
    _, reference = backward_store_all([1.0] * 4, activations, params)
    _, saved = forward_checkpointed(x, params, 4)
    _, grads = backward_checkpointed([1.0] * 4, saved, params, 4)
    assert flat(grads) == flat(reference)


def test_recomputed_gradients_also_match_the_numeric_derivative():
    """Совпадать с ошибочным эталоном мало — сверяемся с разностью напрямую."""
    params, x = smooth_net()
    _, saved = forward_checkpointed(x, params, 2)
    _, grads = backward_checkpointed([1.0] * 3, saved, params, 2)
    for layer_index in (0, len(params) - 1):
        for slot in range(4):
            for j in range(3):
                numeric = numeric_param_grad(params, x, layer_index, slot, j)
                assert grads[layer_index][slot][j] == pytest.approx(numeric, abs=1e-6)


# --------------------------------------------------------- checkpoint_budget
def test_without_checkpointing_every_layer_stays_alive():
    assert checkpoint_budget(64, 1000)["floats"] == 64 * 3 * 1000


def test_checkpointing_cuts_the_memory():
    plain = checkpoint_budget(64, 1000)["floats"]
    chunked = checkpoint_budget(64, 1000, 8)["floats"]
    assert chunked < plain / 5


def test_checkpointing_costs_extra_flops():
    """Размен: память вниз, время вверх. Бесплатного варианта тут нет."""
    plain = checkpoint_budget(64, 1000)["flops"]
    chunked = checkpoint_budget(64, 1000, 8)["flops"]
    assert chunked > plain


def test_naive_recompute_costs_a_third():
    assert checkpoint_budget(64, 1000, 8)["overhead"] == pytest.approx(1 / 3)


def test_selective_recompute_costs_five_percent():
    """Korthikanti: пересчитываем только внимание, 15% слоя вместо 100%."""
    budget = checkpoint_budget(64, 1000, 8, recompute_fraction=0.15)
    assert budget["overhead"] == pytest.approx(0.05)


def test_flops_depend_on_what_is_recomputed_not_on_the_segment_size():
    """Память и время крутятся разными рычагами."""
    small = checkpoint_budget(64, 1000, 2)
    big = checkpoint_budget(64, 1000, 32)
    assert small["flops"] == APPROX(big["flops"])
    assert small["floats"] != big["floats"]


def test_checkpoint_budget_rejects_a_zero_segment():
    with pytest.raises(ValueError):
        checkpoint_budget(64, 1000, 0)


# ------------------------------------------------------------ optimal_segment
def test_the_classic_sqrt_rule_appears_at_one_tensor_per_layer():
    assert optimal_segment(64, per_layer=1) == 8 == round(math.sqrt(64))
    assert optimal_segment(100, per_layer=1) == 10


def test_the_optimum_beats_both_extremes():
    best = optimal_segment(64, per_layer=1)
    floats = lambda k: checkpoint_budget(64, 1, k, per_layer=1)["floats"]
    assert floats(best) < floats(1)
    assert floats(best) < floats(64)


def test_fatter_layers_shorten_the_best_segment():
    """Чем больше тензоров живёт внутри слоя, тем короче выгодный сегмент."""
    assert optimal_segment(64, per_layer=3) < optimal_segment(64, per_layer=1)


def test_optimal_segment_stays_inside_the_model():
    assert 1 <= optimal_segment(7, per_layer=1) <= 7
