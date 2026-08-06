"""Тесты к уроку «Условные GAN и Pix2Pix». Правь exercise.py."""

import math
import random
import statistics

import pytest

from exercise import (
    best_constant,
    conditioned_input,
    cycle_consistency_loss,
    l1_loss,
    linear_generator,
    one_hot,
    patchgan_score,
    pix2pix_generator_loss,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def noise_batch(n=200, seed=0):
    """Фиксированный батч шума: сравнения по классам должны быть воспроизводимы."""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 1.0)] for _ in range(n)]


def flat(M):
    """Плоский список из матрицы: pytest.approx не умеет вложенные списки."""
    return [v for row in M for v in row]


# --------------------------------------------------------------- one_hot
def test_one_hot_puts_the_one_at_the_class_index():
    assert one_hot(0, 3) == APPROX([1.0, 0.0, 0.0])
    assert one_hot(2, 3) == APPROX([0.0, 0.0, 1.0])


def test_one_hot_sums_to_one():
    assert sum(one_hot(1, 5)) == APPROX(1.0)


def test_negative_class_raises_value_error():
    """Ловушка: список молча принял бы -1 и поставил единицу с конца."""
    with pytest.raises(ValueError):
        one_hot(-1, 3)


def test_class_past_the_end_raises_value_error():
    with pytest.raises(ValueError):
        one_hot(3, 3)


# ------------------------------------------------------ conditioned_input
def test_conditioned_input_appends_the_one_hot():
    assert conditioned_input([0.5], 1, 2) == APPROX([0.5, 0.0, 1.0])


def test_conditioned_input_keeps_the_data_first():
    assert conditioned_input([1.0, 2.0], 0, 2) == APPROX([1.0, 2.0, 1.0, 0.0])


def test_conditioned_input_length_is_data_plus_classes():
    assert len(conditioned_input([1.0, 2.0, 3.0], 1, 4)) == 7


def test_conditioned_input_does_not_mutate_the_data_vector():
    x = [1.0]
    conditioned_input(x, 0, 2)
    assert x == [1.0]


def test_different_classes_give_different_inputs():
    """Если бы условие не доезжало до входа, D нечего было бы проверять."""
    assert conditioned_input([0.5], 0, 2) != conditioned_input([0.5], 1, 2)


# ------------------------------------------------------- linear_generator
def test_linear_generator_reads_the_condition_columns():
    assert linear_generator([1.0], 1, [[0.0, 0.0, 5.0]], [0.0], 2) == APPROX([5.0])


def test_linear_generator_maps_each_class_to_its_own_mode():
    """Класс 0 -> около -2, класс 1 -> около +2, ровно как в code/main.py урока."""
    W, b = [[1.0, -2.0, 2.0]], [0.0]
    zs = noise_batch()
    mean_0 = statistics.fmean(linear_generator(z, 0, W, b, 2)[0] for z in zs)
    mean_1 = statistics.fmean(linear_generator(z, 1, W, b, 2)[0] for z in zs)
    assert mean_0 == pytest.approx(-2.0, abs=0.2)
    assert mean_1 == pytest.approx(2.0, abs=0.2)


def test_dropping_the_condition_destroys_the_input_output_correspondence():
    """Обнулили столбцы условия — и G(z, 0) стал равен G(z, 1). Соответствия нет."""
    zs = noise_batch(n=50)
    conditional = [[1.0, -2.0, 2.0]]
    marginal = [[1.0, 0.0, 0.0]]
    for z in zs:
        assert linear_generator(z, 0, conditional, [0.0], 2) != linear_generator(
            z, 1, conditional, [0.0], 2
        )
        assert linear_generator(z, 0, marginal, [0.0], 2) == linear_generator(
            z, 1, marginal, [0.0], 2
        )


def test_linear_generator_rejects_a_row_of_the_wrong_width():
    with pytest.raises(ValueError):
        linear_generator([1.0], 0, [[1.0, 1.0]], [0.0], 2)


def test_linear_generator_rejects_a_bias_of_the_wrong_length():
    with pytest.raises(ValueError):
        linear_generator([1.0], 0, [[1.0, 1.0, 1.0]], [0.0, 0.0], 2)


# --------------------------------------------------------------- l1_loss
def test_l1_loss_averages_absolute_differences():
    assert l1_loss([1.0, 2.0], [1.0, 4.0]) == APPROX(1.0)


def test_l1_loss_of_a_perfect_match_is_zero():
    assert l1_loss([3.0, -3.0], [3.0, -3.0]) == APPROX(0.0)


def test_l1_loss_is_symmetric():
    assert l1_loss([1.0], [5.0]) == APPROX(l1_loss([5.0], [1.0]))


def test_l1_loss_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        l1_loss([1.0, 2.0], [1.0])


# ---------------------------------------------------------- best_constant
def test_l1_optimum_is_the_median():
    assert best_constant([0.0, 0.0, 0.0, 100.0], "l1") == APPROX(0.0)


def test_l2_optimum_is_the_mean():
    assert best_constant([0.0, 0.0, 0.0, 100.0], "l2") == APPROX(25.0)


def test_l2_invents_an_answer_that_is_not_in_the_data():
    """Вот это и есть «мыло»: среднее двух правдоподобных ответов — третий, нереальный."""
    targets = [0.0, 10.0]
    assert best_constant(targets, "l2") == APPROX(5.0)
    assert 5.0 not in targets


def test_the_median_beats_the_mean_on_the_l1_objective():
    targets = [0.0, 0.0, 0.0, 100.0]
    med = best_constant(targets, "l1")
    avg = best_constant(targets, "l2")
    assert l1_loss(targets, [med] * 4) < l1_loss(targets, [avg] * 4)


def test_median_of_an_odd_sample_is_the_middle_element():
    assert best_constant([5.0, 1.0, 3.0], "l1") == APPROX(3.0)


def test_unknown_norm_raises_value_error():
    with pytest.raises(ValueError):
        best_constant([1.0, 2.0], "linf")


def test_empty_targets_raise_value_error():
    with pytest.raises(ValueError):
        best_constant([], "l1")


# ------------------------------------------------ pix2pix_generator_loss
def test_pix2pix_loss_is_pure_adversarial_on_a_perfect_reconstruction():
    assert pix2pix_generator_loss(0.5, [1.0], [1.0]) == APPROX(math.log(2))


def test_pix2pix_loss_adds_lambda_times_l1():
    assert pix2pix_generator_loss(0.5, [1.0], [0.0]) == APPROX(math.log(2) + 100.0)


def test_lambda_controls_how_much_faithfulness_costs():
    cheap = pix2pix_generator_loss(0.5, [1.0], [0.0], lam=1.0)
    dear = pix2pix_generator_loss(0.5, [1.0], [0.0], lam=100.0)
    assert dear > cheap


def test_lambda_zero_leaves_only_the_adversarial_term():
    """lam = 0 — это обычный cGAN: правдоподобно, но не обязательно то, что просили."""
    assert pix2pix_generator_loss(0.5, [1.0], [99.0], lam=0.0) == APPROX(math.log(2))


def test_fooling_the_discriminator_lowers_the_loss():
    assert pix2pix_generator_loss(0.9, [1.0], [1.0]) < pix2pix_generator_loss(
        0.1, [1.0], [1.0]
    )


def test_pix2pix_loss_survives_a_perfectly_caught_fake():
    value = pix2pix_generator_loss(0.0, [1.0], [1.0])
    assert math.isfinite(value) and value > 20


def test_negative_lambda_raises_value_error():
    with pytest.raises(ValueError):
        pix2pix_generator_loss(0.5, [1.0], [1.0], lam=-1.0)


# ----------------------------------------------------------- patchgan_score
def test_non_overlapping_patches_tile_the_image():
    seen = []
    patchgan_score(
        [[float(r * 4 + c) for c in range(4)] for r in range(4)],
        2,
        2,
        lambda patch: seen.append(flat(patch)) or 0.0,
    )
    assert len(seen) == 4


def test_stride_one_makes_the_patches_overlap():
    seen = []
    patchgan_score(
        [[0.0] * 4 for _ in range(4)],
        2,
        1,
        lambda patch: seen.append(flat(patch)) or 0.0,
    )
    assert len(seen) == 9


def test_mean_of_patch_means_is_the_image_mean_when_patches_tile():
    image = [[float(r * 4 + c) for c in range(4)] for r in range(4)]
    got = patchgan_score(image, 2, 2, lambda p: statistics.fmean(flat(p)))
    assert got == APPROX(statistics.fmean(flat(image)))


def test_one_bad_patch_costs_exactly_its_share():
    """Локальный судья снижает оценку на четверть, а глобальный обнулил бы её целиком."""
    image = [
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    realism = lambda p: min(flat(p))
    assert patchgan_score(image, 2, 2, realism) == APPROX(0.75)
    assert realism(image) == APPROX(0.0)


def test_a_fully_real_image_scores_one():
    image = [[1.0] * 4 for _ in range(4)]
    assert patchgan_score(image, 2, 2, lambda p: min(flat(p))) == APPROX(1.0)


def test_patch_bigger_than_the_image_raises_value_error():
    with pytest.raises(ValueError):
        patchgan_score([[1.0, 1.0], [1.0, 1.0]], 3, 1, lambda p: 0.0)


def test_zero_stride_raises_value_error():
    with pytest.raises(ValueError):
        patchgan_score([[1.0, 1.0], [1.0, 1.0]], 2, 0, lambda p: 0.0)


# ------------------------------------------------- cycle_consistency_loss
def test_cycle_loss_is_zero_for_a_perfect_inverse():
    forward = lambda v: [a * 2.0 for a in v]
    backward = lambda v: [a / 2.0 for a in v]
    assert cycle_consistency_loss([1.0, 2.0, 3.0], forward, backward) == APPROX(0.0)


def test_cycle_loss_measures_the_drift_of_the_round_trip():
    assert cycle_consistency_loss(
        [1.0], lambda v: [a + 1 for a in v], lambda v: v
    ) == APPROX(1.0)


def test_cycle_loss_grows_with_the_drift():
    small = cycle_consistency_loss([1.0], lambda v: [a + 0.1 for a in v], lambda v: v)
    big = cycle_consistency_loss([1.0], lambda v: [a + 5.0 for a in v], lambda v: v)
    assert big > small


def test_identity_pair_scores_zero_while_translating_nothing():
    """Ноль цикл-лосса сам по себе ничего не доказывает — нужен ещё adversarial."""
    assert cycle_consistency_loss([1.0, 2.0], lambda v: v, lambda v: v) == APPROX(0.0)


def test_cycle_loss_needs_the_backward_map_to_undo_the_forward_one():
    """Оба перевода одинаково масштабируют — цикл не замкнулся."""
    doubling = lambda v: [a * 2.0 for a in v]
    assert cycle_consistency_loss([1.0], doubling, doubling) > 0.0
