"""Тесты к уроку «Генерация изображений — GAN». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    bce_with_logits,
    conv_transpose_output_size,
    discriminator_loss,
    generator_loss,
    generator_loss_grad,
    mode_collapse_score,
    power_iteration_sigma,
    sigmoid,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

LOG2 = math.log(2.0)


def numeric_grad(f, x, h=1e-5):
    """Центральная разность — эталон, с которым сверяем аналитический градиент."""
    return (f(x + h) - f(x - h)) / (2 * h)


# ------------------------------------------------------------------ sigmoid
def test_sigmoid_at_zero_is_one_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_is_symmetric_around_one_half():
    """sigmoid(-x) = 1 - sigmoid(x) для любого x."""
    for x in (0.3, 2.0, 7.5):
        assert sigmoid(-x) == pytest.approx(1.0 - sigmoid(x), abs=1e-12)


def test_sigmoid_is_monotonically_increasing():
    values = [sigmoid(x) for x in (-5.0, -1.0, 0.0, 1.0, 5.0)]
    assert values == sorted(values)
    assert len(set(values)) == 5


def test_sigmoid_does_not_overflow_on_large_negative_input():
    """Ловушка: math.exp(1000) — OverflowError. Ветка для x < 0 обязательна."""
    assert sigmoid(-1000.0) == APPROX(0.0)
    assert sigmoid(1000.0) == APPROX(1.0)


# --------------------------------------------------------- bce_with_logits
def test_bce_at_zero_logit_is_log_two_for_both_targets():
    assert bce_with_logits(0.0, 1) == APPROX(LOG2)
    assert bce_with_logits(0.0, 0) == APPROX(LOG2)


def test_bce_is_near_zero_when_the_confident_guess_is_right():
    assert bce_with_logits(100.0, 1) == pytest.approx(0.0, abs=1e-9)
    assert bce_with_logits(-100.0, 0) == pytest.approx(0.0, abs=1e-9)


def test_bce_grows_linearly_when_the_confident_guess_is_wrong():
    """Уверенный промах стоит примерно |logit|, а не бесконечность."""
    assert bce_with_logits(100.0, 0) == pytest.approx(100.0, abs=1e-6)
    assert bce_with_logits(-100.0, 1) == pytest.approx(100.0, abs=1e-6)


def test_bce_matches_the_naive_formula_in_the_safe_range():
    """Там, где наивный путь не переполняется, устойчивая форма даёт то же."""
    for x in (-3.0, -0.5, 0.0, 0.5, 3.0):
        p = sigmoid(x)
        assert bce_with_logits(x, 1) == pytest.approx(-math.log(p), abs=1e-9)
        assert bce_with_logits(x, 0) == pytest.approx(-math.log(1.0 - p), abs=1e-9)


def test_bce_survives_the_logit_that_kills_the_naive_version():
    """sigmoid(-800) = 0.0, log(0) = -inf. Устойчивая форма выдаёт число."""
    value = bce_with_logits(-800.0, 1)
    assert math.isfinite(value)
    assert value == pytest.approx(800.0, abs=1e-6)


# ------------------------------------------------------ discriminator_loss
def test_discriminator_loss_at_equilibrium_is_two_log_two():
    """D отвечает 0.5 везде — ровно точка равновесия из статьи."""
    assert discriminator_loss([0.0, 0.0], [0.0, 0.0]) == APPROX(2 * LOG2)


def test_discriminator_loss_drops_to_zero_when_d_sees_everything():
    assert discriminator_loss([100.0], [-100.0]) == pytest.approx(0.0, abs=1e-9)


def test_discriminator_loss_averages_inside_each_half_then_adds():
    """Две средние складываются, а не усредняются между собой."""
    real = [1.0, 3.0]
    fake = [-2.0, 4.0]
    expected = (bce_with_logits(1.0, 1) + bce_with_logits(3.0, 1)) / 2 + (
        bce_with_logits(-2.0, 0) + bce_with_logits(4.0, 0)
    ) / 2
    assert discriminator_loss(real, fake) == APPROX(expected)


def test_discriminator_loss_ignores_batch_size_when_logits_repeat():
    """Усреднение внутри половин: дублирование батча ничего не меняет."""
    one = discriminator_loss([1.5], [-0.5])
    many = discriminator_loss([1.5] * 7, [-0.5] * 3)
    assert many == APPROX(one)


# ---------------------------------------------------------- generator_loss
def test_generator_loss_non_saturating_at_zero_is_log_two():
    assert generator_loss([0.0]) == APPROX(LOG2)


def test_generator_loss_saturating_is_the_negated_mirror_at_zero():
    assert generator_loss([0.0], non_saturating=False) == APPROX(-LOG2)


def test_generator_loss_falls_when_the_generator_fools_the_discriminator():
    """Обе формы падают по мере успеха G — просто к разным пределам."""
    assert generator_loss([100.0]) < generator_loss([0.0]) < generator_loss([-100.0])
    ns_false = lambda x: generator_loss([x], non_saturating=False)
    assert ns_false(100.0) < ns_false(0.0) < ns_false(-100.0)


def test_generator_loss_saturating_equals_log_one_minus_d():
    """Исходная форма из статьи 2014: mean log(1 - sigmoid(x))."""
    logits = [-1.0, 0.5, 2.0]
    expected = sum(math.log(1.0 - sigmoid(x)) for x in logits) / len(logits)
    assert generator_loss(logits, non_saturating=False) == APPROX(expected)


# ----------------------------------------------------- generator_loss_grad
def test_generator_loss_grad_matches_the_numeric_derivative():
    """Аналитика против центральной разности — обе формы."""
    for x in (-3.0, -0.5, 0.0, 1.0, 4.0):
        analytic = generator_loss_grad(x)
        numeric = numeric_grad(lambda t: generator_loss([t]), x)
        assert analytic == pytest.approx(numeric, abs=1e-6)

        analytic_sat = generator_loss_grad(x, non_saturating=False)
        numeric_sat = numeric_grad(
            lambda t: generator_loss([t], non_saturating=False), x
        )
        assert analytic_sat == pytest.approx(numeric_sat, abs=1e-6)


def test_generator_loss_grad_is_negative_so_the_logit_is_pushed_up():
    """Оба варианта толкают логит фейка вверх: шаг идёт против градиента."""
    assert generator_loss_grad(-2.0) < 0
    assert generator_loss_grad(-2.0, non_saturating=False) < 0


def test_saturating_gradient_vanishes_where_non_saturating_survives():
    """Вся причина переписать лосс: на -10 разница в четыре порядка."""
    ns = abs(generator_loss_grad(-10.0))
    sat = abs(generator_loss_grad(-10.0, non_saturating=False))
    assert ns > 0.99
    assert sat < 1e-4
    assert ns / sat > 1000


def test_generator_loss_grad_goes_to_zero_once_d_is_fooled():
    assert generator_loss_grad(20.0) == pytest.approx(0.0, abs=1e-8)


# --------------------------------------------- conv_transpose_output_size
def test_first_generator_layer_turns_noise_into_four_by_four():
    assert conv_transpose_output_size(1, 4, 1, 0) == 4


def test_dcgan_kernel_four_stride_two_padding_one_doubles_any_size():
    for size in (4, 5, 8, 16, 37):
        assert conv_transpose_output_size(size, 4, 2, 1) == 2 * size


def test_generator_stack_reaches_thirty_two_pixels():
    size = 1
    for kernel, stride, padding in [(4, 1, 0), (4, 2, 1), (4, 2, 1), (4, 2, 1)]:
        size = conv_transpose_output_size(size, kernel, stride, padding)
    assert size == 32


def test_output_padding_adds_exactly_one_pixel_per_unit():
    base = conv_transpose_output_size(8, 3, 2, 1)
    assert conv_transpose_output_size(8, 3, 2, 1, output_padding=1) == base + 1


# ---------------------------------------------------- power_iteration_sigma
def test_power_iteration_finds_the_largest_diagonal_entry():
    sigma = power_iteration_sigma([[3.0, 0.0], [0.0, 1.0]], random.Random(0))
    assert sigma == pytest.approx(3.0, abs=1e-6)


def test_power_iteration_is_reproducible_for_the_same_seed():
    """Случайность идёт только из rng — глобальный random сломал бы тесты."""
    a = power_iteration_sigma([[1.0, 2.0], [3.0, 4.0]], random.Random(11))
    b = power_iteration_sigma([[1.0, 2.0], [3.0, 4.0]], random.Random(11))
    assert a == APPROX(b)


def test_power_iteration_is_homogeneous_in_the_matrix_scale():
    """sigma(cW) = c * sigma(W): норма — однородная функция первой степени."""
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    scaled = [[5.0 * v for v in row] for row in matrix]
    base = power_iteration_sigma(matrix, random.Random(3))
    big = power_iteration_sigma(scaled, random.Random(3))
    assert big == pytest.approx(5.0 * base, abs=1e-6)


def test_power_iteration_bounds_the_layer_gain():
    """sigma — максимальное растяжение: |Wx| <= sigma * |x| для любого x."""
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    sigma = power_iteration_sigma(matrix, random.Random(5))
    rng = random.Random(99)
    for _ in range(20):
        x = [rng.gauss(0, 1), rng.gauss(0, 1)]
        wx = [sum(matrix[i][j] * x[j] for j in range(2)) for i in range(2)]
        assert math.sqrt(sum(v * v for v in wx)) <= sigma * math.sqrt(
            sum(v * v for v in x)
        ) + 1e-6


def test_dividing_by_sigma_makes_the_layer_one_lipschitz():
    """Ровно то, что делает spectral_norm: после деления sigma становится 1."""
    matrix = [[4.0, 1.0], [0.0, 3.0]]
    sigma = power_iteration_sigma(matrix, random.Random(2))
    normed = [[v / sigma for v in row] for row in matrix]
    assert power_iteration_sigma(normed, random.Random(2)) == pytest.approx(
        1.0, abs=1e-5
    )


# ----------------------------------------------------- mode_collapse_score
def test_score_of_two_points_is_the_distance_between_them():
    assert mode_collapse_score([[0.0], [1.0]]) == APPROX(1.0)
    assert mode_collapse_score([[0.0, 0.0], [3.0, 4.0]]) == APPROX(5.0)


def test_identical_samples_score_zero():
    """Полный коллапс: генератор выдаёт одну и ту же картинку."""
    assert mode_collapse_score([[1.0], [1.0], [1.0]]) == APPROX(0.0)


def test_a_single_sample_has_no_pairs_and_scores_zero():
    assert mode_collapse_score([[5.0, 5.0]]) == APPROX(0.0)
    assert mode_collapse_score([]) == APPROX(0.0)


def test_score_averages_over_all_pairs_not_over_samples():
    """У трёх точек три пары: 1 + 2 + 1 делится на 3, а не на 3 сэмпла."""
    assert mode_collapse_score([[0.0], [1.0], [3.0]]) == APPROX((1.0 + 3.0 + 2.0) / 3)


def test_collapsed_batch_scores_far_below_a_diverse_one():
    rng = random.Random(0)
    diverse = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(16)]
    collapsed = [list(diverse[0]) for _ in range(16)]
    assert mode_collapse_score(collapsed) < 0.5 * mode_collapse_score(diverse)


