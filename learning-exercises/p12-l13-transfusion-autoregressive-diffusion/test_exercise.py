"""Тесты к уроку «Transfusion: авторегрессия по тексту и диффузия по картинке».

Правь exercise.py.
"""

import pytest

from exercise import (
    IMG_CLOSE,
    IMG_OPEN,
    PATCH,
    balanced_weights,
    build_mask,
    find_image_blocks,
    flow_interpolate,
    flow_loss,
    flow_loss_grad,
    flow_target,
    generation_forward_passes,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in matrix for v in row]


# последовательность из упражнения 2 урока: [T, T, <image>, P, P, P, P, </image>, T]
SEQ = [7, 8, IMG_OPEN, PATCH, PATCH, PATCH, PATCH, IMG_CLOSE, 9]


# ------------------------------------------------------- find_image_blocks
def test_find_image_blocks_excludes_the_tags_themselves():
    assert find_image_blocks([5, IMG_OPEN, PATCH, PATCH, IMG_CLOSE, 6]) == [(2, 4)]


def test_find_image_blocks_on_pure_text_is_empty():
    assert find_image_blocks([5, 6, 7]) == []


def test_find_image_blocks_finds_every_block():
    seq = [IMG_OPEN, PATCH, IMG_CLOSE, 1, IMG_OPEN, PATCH, PATCH, IMG_CLOSE]
    assert find_image_blocks(seq) == [(1, 2), (5, 7)]


def test_find_image_blocks_rejects_unclosed_image():
    with pytest.raises(ValueError):
        find_image_blocks([1, IMG_OPEN, PATCH, PATCH])


def test_find_image_blocks_rejects_close_without_open():
    with pytest.raises(ValueError):
        find_image_blocks([1, PATCH, IMG_CLOSE])


# --------------------------------------------------------------- build_mask
def test_build_mask_on_the_lesson_sequence():
    expected = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    assert flat(build_mask(SEQ)) == flat(expected)


def test_pure_text_mask_is_lower_triangular():
    """Без картинок Transfusion вырождается в обычную causal LM."""
    mask = build_mask([1, 2, 3, 4])
    expected = [[1 if j <= i else 0 for j in range(4)] for i in range(4)]
    assert flat(mask) == flat(expected)


def test_image_block_is_bidirectional_and_therefore_symmetric():
    """Внутри одной картинки порядок патчей не значит ничего: M[i][j] == M[j][i]."""
    mask = build_mask(SEQ)
    (start, end), = find_image_blocks(SEQ)
    for i in range(start, end):
        for j in range(start, end):
            assert mask[i][j] == mask[j][i] == 1


def test_patch_is_blind_to_everything_after_its_block():
    """Патч не видит ни свой </image>, ни текст за ним — иначе течёт будущее."""
    mask = build_mask(SEQ)
    (start, end), = find_image_blocks(SEQ)
    for i in range(start, end):
        assert mask[i][end] == 0      # IMG_CLOSE
        assert mask[i][end + 1] == 0  # текст после картинки


def test_text_after_an_image_sees_all_of_its_patches():
    mask = build_mask(SEQ)
    (start, end), = find_image_blocks(SEQ)
    last_text = len(SEQ) - 1
    assert all(mask[last_text][j] == 1 for j in range(start, end))


def test_two_images_do_not_attend_to_each_other():
    """Правило урока покрывает только «тот же блок» — соседняя картинка чужая."""
    seq = [IMG_OPEN, PATCH, IMG_CLOSE, IMG_OPEN, PATCH, IMG_CLOSE]
    mask = build_mask(seq)
    assert mask[4][1] == 0
    assert mask[1][4] == 0


# --------------------------------------------------------- flow_interpolate
def test_flow_interpolate_at_zero_is_the_clean_patch():
    assert flow_interpolate([0.0, 1.0], [1.0, 3.0], 0.0) == APPROX([0.0, 1.0])


def test_flow_interpolate_at_one_is_pure_noise():
    assert flow_interpolate([0.0, 1.0], [1.0, 3.0], 1.0) == APPROX([1.0, 3.0])


def test_flow_interpolate_midpoint_is_the_average():
    assert flow_interpolate([0.0, 1.0], [1.0, 3.0], 0.5) == APPROX([0.5, 2.0])


def test_flow_interpolate_rejects_t_outside_the_segment():
    with pytest.raises(ValueError):
        flow_interpolate([0.0], [1.0], 1.5)


# -------------------------------------------------------------- flow_target
def test_flow_target_points_from_data_to_noise():
    assert flow_target([0.0, 1.0], [1.0, 3.0]) == APPROX([1.0, 2.0])


def test_flow_target_is_zero_when_noise_equals_data():
    assert flow_target([2.0, -1.0], [2.0, -1.0]) == APPROX([0.0, 0.0])


def test_flow_target_flips_sign_when_arguments_swap():
    forward = flow_target([0.0, 1.0], [1.0, 3.0])
    backward = flow_target([1.0, 3.0], [0.0, 1.0])
    assert forward == APPROX([-v for v in backward])


def test_flow_target_is_the_slope_of_the_interpolation():
    """Путь прямой, значит (xt - x0) / t даёт одну и ту же скорость при любом t."""
    x0, eps = [0.3, -0.4], [1.1, 0.9]
    target = flow_target(x0, eps)
    for t in (0.25, 0.5, 0.9):
        xt = flow_interpolate(x0, eps, t)
        slope = [(a - b) / t for a, b in zip(xt, x0)]
        assert slope == APPROX(target)


# ---------------------------------------------------------------- flow_loss
def test_flow_loss_is_zero_on_a_perfect_prediction():
    assert flow_loss([1.0, 2.0], [0.0, 1.0], [1.0, 3.0]) == APPROX(0.0)


def test_flow_loss_worked_example():
    assert flow_loss([0.0, 0.0], [0.0, 1.0], [1.0, 3.0]) == APPROX(2.5)


def test_flow_loss_averages_instead_of_summing():
    """Удвоение размера патча при той же ошибке не должно менять лосс."""
    small = flow_loss([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    big = flow_loss([0.0] * 4, [0.0] * 4, [1.0] * 4)
    assert small == APPROX(big)


def test_flow_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        flow_loss([0.0, 0.0], [0.0], [1.0])


# ----------------------------------------------------------- flow_loss_grad
def test_flow_loss_grad_is_zero_at_the_optimum():
    assert flow_loss_grad([1.0, 2.0], [0.0, 1.0], [1.0, 3.0]) == APPROX([0.0, 0.0])


def test_flow_loss_grad_worked_example():
    assert flow_loss_grad([0.0, 0.0], [0.0, 1.0], [1.0, 3.0]) == APPROX([-1.0, -2.0])


def test_flow_loss_grad_matches_central_difference():
    """Аналитический градиент против численного — двойка и деление на n на месте."""
    pred, x0, eps = [0.4, -0.7, 1.2], [0.1, 0.2, 0.3], [1.0, -0.5, 0.8]
    h = 1e-6
    analytic = flow_loss_grad(pred, x0, eps)
    for k in range(len(pred)):
        up, down = list(pred), list(pred)
        up[k] += h
        down[k] -= h
        numeric = (flow_loss(up, x0, eps) - flow_loss(down, x0, eps)) / (2 * h)
        assert analytic[k] == pytest.approx(numeric, abs=1e-6)


def test_flow_loss_grad_points_uphill():
    """Шаг ПО градиенту обязан увеличить лосс, шаг против — уменьшить."""
    pred, x0, eps = [0.0, 0.0], [0.0, 1.0], [1.0, 3.0]
    grad = flow_loss_grad(pred, x0, eps)
    up = [p + 0.01 * g for p, g in zip(pred, grad)]
    down = [p - 0.01 * g for p, g in zip(pred, grad)]
    assert flow_loss(down, x0, eps) < flow_loss(pred, x0, eps) < flow_loss(up, x0, eps)


# -------------------------------------------------------- balanced_weights
def test_balanced_weights_worked_example():
    assert balanced_weights(2.0, 20.0) == APPROX((1.0, 0.1))


def test_balanced_weights_equalize_the_two_contributions():
    text_loss, image_loss = 0.7, 13.4
    w_text, w_img = balanced_weights(text_loss, image_loss)
    assert w_text * text_loss == APPROX(w_img * image_loss)


def test_balanced_weights_depend_only_on_the_ratio():
    """Оба лосса выросли вдвое — веса те же, балансировать заново не надо."""
    assert balanced_weights(2.0, 20.0) == APPROX(balanced_weights(4.0, 40.0))


def test_balanced_weights_reject_a_nonpositive_loss():
    with pytest.raises(ValueError):
        balanced_weights(1.0, 0.0)


# ------------------------------------------------ generation_forward_passes
def test_generation_forward_passes_worked_example():
    assert generation_forward_passes(50, 256, 20) == 70


def test_diffusion_cost_does_not_grow_with_patch_count():
    """Патчи денойзятся параллельно: 16 их или 4096 — цена одна."""
    assert generation_forward_passes(50, 16, 20) == generation_forward_passes(50, 4096, 20)


def test_autoregressive_mode_pays_per_patch():
    """Chameleon-режим: каждый патч это отдельный проход, отсюда и разрыв."""
    assert generation_forward_passes(50, 256, 0) == 306
    assert generation_forward_passes(50, 256, 0) > generation_forward_passes(50, 256, 20)


def test_generation_forward_passes_rejects_negative_counts():
    with pytest.raises(ValueError):
        generation_forward_passes(50, -1, 20)
