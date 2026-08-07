"""Тесты к уроку «Диффузионные трансформеры и rectified flow». Правь exercise.py."""

import random

import pytest

from exercise import (
    adaln_zero_block,
    classifier_free_guidance,
    euler_sample,
    flow_matching_loss,
    patchify,
    rectified_flow_path,
    unpatchify,
    velocity_target,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(nested):
    """Развернуть вложенные списки в плоский — pytest.approx не умеет вложенность."""
    out = []
    for item in nested:
        if isinstance(item, list):
            out.extend(flat(item))
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------- patchify
def test_patchify_with_unit_patches_lists_every_pixel():
    assert patchify([[[1, 2], [3, 4]]], 1) == [[1], [2], [3], [4]]


def test_patchify_orders_token_values_channel_major():
    """Внутри токена сначала весь первый канал, потом второй."""
    image = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]
    assert patchify(image, 2) == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_patchify_walks_patches_left_to_right_then_down():
    image = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
    tokens = patchify(image, 2)
    assert len(tokens) == 4
    assert tokens[0] == [1, 2, 5, 6]
    assert tokens[1] == [3, 4, 7, 8]
    assert tokens[2] == [9, 10, 13, 14]


def test_patchify_rejects_size_not_divisible_by_patch():
    with pytest.raises(ValueError):
        patchify([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], 2)


# -------------------------------------------------------------- unpatchify
def test_unpatchify_undoes_patchify():
    image = [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]],
    ]
    assert flat(unpatchify(patchify(image, 2), 2, 2)) == flat(image)


def test_unpatchify_rejects_non_square_token_count():
    """Сетка патчей квадратная: три токена в квадрат не складываются."""
    with pytest.raises(ValueError):
        unpatchify([[1], [2], [3]], 1, 1)


def test_unpatchify_rejects_token_of_wrong_length():
    with pytest.raises(ValueError):
        unpatchify([[1, 2], [3, 4], [5, 6], [7, 8]], 1, 1)


# --------------------------------------------------------- adaln_zero_block
def test_adaln_zero_gate_makes_the_block_identity():
    """Смысл zero-init: свежий блок ничего не меняет, что бы ни выдала ветка."""
    out = adaln_zero_block([1.0, -1.0, 3.0], lambda h: [99.0] * 3, [0.0] * 3, [0.0] * 3, 0.0)
    assert out == APPROX([1.0, -1.0, 3.0])


def test_adaln_zero_normalizes_before_the_branch():
    seen = {}

    def branch(h):
        seen["h"] = list(h)
        return [0.0] * len(h)

    adaln_zero_block([1.0, 2.0, 3.0], branch, [0.0] * 3, [0.0] * 3, 1.0)
    mean = sum(seen["h"]) / 3
    variance = sum((v - mean) ** 2 for v in seen["h"]) / 3
    assert mean == pytest.approx(0.0, abs=1e-6)
    assert variance == pytest.approx(1.0, abs=1e-4)


def test_adaln_scale_and_shift_modulate_the_normalized_input():
    seen = {}

    def branch(h):
        seen["h"] = list(h)
        return [0.0] * len(h)

    adaln_zero_block([1.0, -1.0], branch, [1.0, 1.0], [5.0, 5.0], 1.0)
    # нормированный вход это примерно [1, -1]; (1 + scale) = 2, плюс shift = 5
    assert seen["h"] == pytest.approx([7.0, 3.0], abs=1e-4)


def test_adaln_gate_scales_only_the_branch_not_the_residual():
    out = adaln_zero_block([1.0, -1.0], lambda h: [1.0, 1.0], [0.0] * 2, [0.0] * 2, 0.5)
    assert out == APPROX([1.5, -0.5])


def test_adaln_rejects_scale_of_wrong_length():
    with pytest.raises(ValueError):
        adaln_zero_block([1.0, 2.0], lambda h: h, [0.0], [0.0, 0.0], 1.0)


# --------------------------------------------------- rectified_flow_path
def test_rectified_flow_at_t_zero_is_clean_data():
    assert rectified_flow_path([1.0, -2.0], [9.0, 9.0], 0.0) == APPROX([1.0, -2.0])


def test_rectified_flow_at_t_one_is_pure_noise():
    assert rectified_flow_path([1.0, -2.0], [9.0, 9.0], 1.0) == APPROX([9.0, 9.0])


def test_rectified_flow_midpoint_is_the_average():
    assert rectified_flow_path([0.0, 0.0], [2.0, 4.0], 0.5) == APPROX([1.0, 2.0])


def test_rectified_flow_is_a_straight_line():
    """Равные шаги по t дают равные шаги в пространстве — это и есть прямая."""
    x0, eps = [1.0, 5.0], [-3.0, 2.0]
    points = [rectified_flow_path(x0, eps, k / 4) for k in range(5)]
    deltas = [
        [b - a for a, b in zip(points[i], points[i + 1])] for i in range(4)
    ]
    for delta in deltas[1:]:
        assert delta == APPROX(deltas[0])


# ------------------------------------------------------- velocity_target
def test_velocity_from_origin_is_the_noise_itself():
    assert velocity_target([0.0, 0.0], [2.0, 4.0]) == APPROX([2.0, 4.0])


def test_velocity_points_from_data_to_noise():
    """Знак: eps - x0, а не x0 - eps."""
    assert velocity_target([5.0], [1.0]) == APPROX([-4.0])


def test_velocity_matches_the_numeric_derivative_of_the_path():
    x0, eps = [1.0, -2.0, 0.5], [4.0, 3.0, -1.5]
    h = 1e-5
    for t in (0.1, 0.5, 0.9):
        up = rectified_flow_path(x0, eps, t + h)
        down = rectified_flow_path(x0, eps, t - h)
        numeric = [(a - b) / (2 * h) for a, b in zip(up, down)]
        assert velocity_target(x0, eps) == pytest.approx(numeric, abs=1e-6)


def test_velocity_is_constant_along_the_whole_path():
    """Скорость не зависит от t — потому шагов Эйлера и нужно мало."""
    x0, eps = [2.0, -1.0], [0.0, 5.0]
    v = velocity_target(x0, eps)
    for t in (0.0, 0.25, 0.75, 1.0):
        moved = rectified_flow_path(x0, eps, t + 0.1)
        here = rectified_flow_path(x0, eps, t)
        assert [(a - b) / 0.1 for a, b in zip(moved, here)] == pytest.approx(v, abs=1e-9)


# ----------------------------------------------------- flow_matching_loss
def _perfect_velocity(x0):
    """Идеальная модель: v = (x_t - x0) / t, потому что x_t - x0 = t * v."""

    def velocity_fn(x_t, t):
        return [(xt - a) / t for xt, a in zip(x_t, x0)]

    return velocity_fn


def test_flow_matching_loss_is_zero_for_a_perfect_model():
    x0 = [1.0, -2.0, 0.5]
    loss = flow_matching_loss(_perfect_velocity(x0), [x0, x0, x0], random.Random(7))
    assert loss == pytest.approx(0.0, abs=1e-9)


def test_flow_matching_loss_is_positive_for_a_wrong_model():
    x0 = [1.0, -2.0, 0.5]
    loss = flow_matching_loss(lambda x_t, t: [0.0] * 3, [x0, x0], random.Random(7))
    assert loss > 0.0


def test_flow_matching_loss_divides_by_the_number_of_scalars():
    """Модель, ошибающаяся ровно на 1 в каждой координате, обязана дать MSE = 1."""
    x0 = [1.0, -2.0, 0.5, 4.0]
    exact = _perfect_velocity(x0)
    off_by_one = lambda x_t, t: [v + 1.0 for v in exact(x_t, t)]
    loss = flow_matching_loss(off_by_one, [x0] * 3, random.Random(3))
    assert loss == pytest.approx(1.0, abs=1e-9)


def test_flow_matching_loss_is_reproducible_for_the_same_seed():
    x0 = [1.0, 0.0]
    model = lambda x_t, t: [0.0, 0.0]
    first = flow_matching_loss(model, [x0, x0], random.Random(11))
    second = flow_matching_loss(model, [x0, x0], random.Random(11))
    assert first == APPROX(second)


def test_flow_matching_loss_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        flow_matching_loss(lambda x_t, t: [0.0], [], random.Random(0))


# ------------------------------------------------ classifier_free_guidance
def test_guidance_scale_zero_returns_the_unconditional_prediction():
    assert classifier_free_guidance([1.0, 2.0], [3.0, 9.0], 0.0) == APPROX([1.0, 2.0])


def test_guidance_scale_one_returns_the_conditional_prediction():
    """scale = 1 это не «выключено», а ровно обусловленное предсказание."""
    assert classifier_free_guidance([1.0, 2.0], [3.0, 9.0], 1.0) == APPROX([3.0, 9.0])


def test_guidance_above_one_extrapolates_past_the_conditional():
    assert classifier_free_guidance([1.0], [3.0], 3.5) == APPROX([8.0])


def test_guidance_does_nothing_when_both_predictions_agree():
    assert classifier_free_guidance([2.0, -1.0], [2.0, -1.0], 5.0) == APPROX([2.0, -1.0])


# ------------------------------------------------------------ euler_sample
def test_euler_walks_backwards_along_a_constant_field():
    assert euler_sample(lambda x, t: [1.0], [1.0], 4) == APPROX([0.0])


def test_euler_on_a_straight_path_is_exact_at_any_step_count():
    """Главное свойство rectified flow: 1 шаг и 50 шагов дают один ответ."""
    x0, eps = [1.0, -2.0], [4.0, 6.0]
    v = velocity_target(x0, eps)
    field = lambda x, t: list(v)
    for steps in (1, 2, 5, 50):
        assert euler_sample(field, eps, steps) == pytest.approx(x0, abs=1e-9)


def test_euler_moves_against_the_velocity():
    """Минус, не плюс: скорость смотрит к шуму, сэмплер идёт от него."""
    assert euler_sample(lambda x, t: [1.0], [0.0], 10)[0] < 0.0


def test_euler_feeds_time_from_one_down_to_dt():
    seen = []

    def field(x, t):
        seen.append(t)
        return [0.0]

    euler_sample(field, [0.0], 4)
    assert seen == pytest.approx([1.0, 0.75, 0.5, 0.25], abs=1e-9)


def test_euler_does_not_mutate_the_starting_point():
    start = [1.0, 2.0]
    euler_sample(lambda x, t: [1.0, 1.0], start, 3)
    assert start == [1.0, 2.0]


def test_euler_rejects_zero_steps():
    with pytest.raises(ValueError):
        euler_sample(lambda x, t: [0.0], [1.0], 0)
