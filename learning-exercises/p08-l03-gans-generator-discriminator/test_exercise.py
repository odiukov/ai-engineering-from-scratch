"""Тесты к уроку «GAN: генератор против дискриминатора». Правь exercise.py."""

import math

import pytest

from exercise import (
    binary_cross_entropy,
    binary_cross_entropy_grad,
    discriminator_loss,
    generator_loss,
    generator_loss_grad,
    is_mode_collapse,
    optimal_discriminator,
    sigmoid,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------- sigmoid
def test_sigmoid_at_zero_is_a_coin_flip():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_stays_inside_the_unit_interval():
    assert 0.0 < sigmoid(-12.0) < sigmoid(12.0) < 1.0


def test_sigmoid_survives_a_huge_negative_logit():
    """Наивная 1/(1+exp(-x)) падает с OverflowError на x = -800."""
    assert sigmoid(-800.0) == APPROX(0.0)


def test_sigmoid_survives_a_huge_positive_logit():
    assert sigmoid(800.0) == APPROX(1.0)


# --------------------------------------------------- binary_cross_entropy
def test_bce_of_a_coin_flip_is_log_two():
    assert binary_cross_entropy(0.5, 1.0) == APPROX(math.log(2))


def test_bce_rewards_a_confident_correct_answer():
    assert binary_cross_entropy(0.9, 1.0) < binary_cross_entropy(0.6, 1.0)


def test_bce_punishes_a_confident_wrong_answer():
    assert binary_cross_entropy(0.1, 1.0) > binary_cross_entropy(0.4, 1.0)


def test_bce_is_symmetric_between_the_two_targets():
    """BCE(p, 1) на p и BCE(p, 0) на 1-p — одно и то же число."""
    assert binary_cross_entropy(0.3, 1.0) == APPROX(binary_cross_entropy(0.7, 0.0))


def test_bce_clips_instead_of_blowing_up_on_p_equal_zero():
    """Ловушка: log(0) уронит прогон на середине обучения. Зажимаем."""
    value = binary_cross_entropy(0.0, 1.0)
    assert math.isfinite(value) and value > 20


def test_bce_clips_on_p_equal_one_too():
    value = binary_cross_entropy(1.0, 0.0)
    assert math.isfinite(value) and value > 20


# ---------------------------------------------- binary_cross_entropy_grad
def test_bce_grad_worked_examples():
    assert binary_cross_entropy_grad(0.5, 1.0) == APPROX(-2.0)
    assert binary_cross_entropy_grad(0.5, 0.0) == APPROX(2.0)


def test_bce_grad_matches_numeric_gradient_for_a_real_sample():
    h = 1e-6
    p = 0.73
    numeric = (
        binary_cross_entropy(p + h, 1.0) - binary_cross_entropy(p - h, 1.0)
    ) / (2 * h)
    assert binary_cross_entropy_grad(p, 1.0) == pytest.approx(numeric, abs=1e-5)


def test_bce_grad_matches_numeric_gradient_for_a_fake_sample():
    h = 1e-6
    p = 0.21
    numeric = (
        binary_cross_entropy(p + h, 0.0) - binary_cross_entropy(p - h, 0.0)
    ) / (2 * h)
    assert binary_cross_entropy_grad(p, 0.0) == pytest.approx(numeric, abs=1e-5)


def test_bce_grad_sign_points_toward_the_target():
    """Ниже цели — градиент отрицательный, выше цели — положительный."""
    assert binary_cross_entropy_grad(0.2, 1.0) < 0
    assert binary_cross_entropy_grad(0.8, 0.0) > 0


def test_bce_grad_grows_as_p_moves_away_from_the_target():
    """Чем сильнее D ошибся, тем больше по модулю толчок — 1/(p(1-p)) в знаменателе."""
    near = abs(binary_cross_entropy_grad(0.8, 1.0))
    far = abs(binary_cross_entropy_grad(0.05, 1.0))
    assert far > near


# ---------------------------------------------------- discriminator_loss
def test_discriminator_loss_at_equilibrium_is_two_log_two():
    """1.386 в логах — это не поломка, это ровно то самое равновесие."""
    assert discriminator_loss([0.5] * 4, [0.5] * 4) == APPROX(2 * math.log(2))


def test_perfect_discriminator_has_almost_zero_loss():
    assert discriminator_loss([1.0, 1.0], [0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_inverted_discriminator_has_a_huge_loss():
    """D, который называет настоящее подделкой, платит по полной."""
    assert discriminator_loss([0.0], [1.0]) > 40


def test_discriminator_loss_averages_each_batch_separately():
    """Разные размеры пачек не должны перекашивать вклад одной из них."""
    balanced = discriminator_loss([0.5, 0.5], [0.5, 0.5])
    lopsided = discriminator_loss([0.5] * 10, [0.5])
    assert balanced == APPROX(lopsided)


def test_discriminator_loss_on_empty_batch_raises_value_error():
    with pytest.raises(ValueError):
        discriminator_loss([], [0.5])


# -------------------------------------------------------- generator_loss
def test_non_saturating_generator_loss_at_equilibrium_is_log_two():
    assert generator_loss([0.5]) == APPROX(math.log(2))


def test_vanilla_generator_loss_is_the_negative_of_it_at_equilibrium():
    assert generator_loss([0.5], non_saturating=False) == APPROX(-math.log(2))


def test_both_forms_fall_when_the_discriminator_gets_fooled():
    """Обе формы хотят одного: D(G(z)) ближе к единице — лосс меньше."""
    assert generator_loss([0.9]) < generator_loss([0.1])
    assert generator_loss([0.9], non_saturating=False) < generator_loss(
        [0.1], non_saturating=False
    )


def test_generator_loss_averages_over_the_batch():
    assert generator_loss([0.5, 0.5, 0.5]) == APPROX(generator_loss([0.5]))


def test_generator_loss_survives_a_perfectly_caught_fake():
    value = generator_loss([0.0])
    assert math.isfinite(value) and value > 20


def test_generator_loss_on_empty_batch_raises_value_error():
    with pytest.raises(ValueError):
        generator_loss([])


# --------------------------------------------------- generator_loss_grad
def test_generator_loss_grads_agree_at_equilibrium():
    """При p = 0.5 обе формы дают один и тот же градиент — расхождение начинается дальше."""
    assert generator_loss_grad([0.5]) == APPROX([-0.5])
    assert generator_loss_grad([0.5], non_saturating=False) == APPROX([-0.5])


def test_vanilla_generator_gradient_vanishes_when_the_discriminator_is_confident():
    """Ванильная форма глохнет ровно тогда, когда генератору нужнее всего сигнал."""
    p = 1e-4
    assert abs(generator_loss_grad([p], non_saturating=False)[0]) < 1e-3


def test_non_saturating_gradient_stays_strong_when_the_discriminator_is_confident():
    p = 1e-4
    assert abs(generator_loss_grad([p])[0]) > 0.99


def test_non_saturating_gradient_beats_vanilla_by_orders_of_magnitude():
    p = 1e-4
    strong = abs(generator_loss_grad([p])[0])
    weak = abs(generator_loss_grad([p], non_saturating=False)[0])
    assert strong > 1000 * weak


def test_non_saturating_grad_matches_numeric_gradient_over_the_logit():
    h = 1e-6
    logit = -1.3
    up = generator_loss([sigmoid(logit + h)])
    down = generator_loss([sigmoid(logit - h)])
    assert generator_loss_grad([sigmoid(logit)])[0] == pytest.approx(
        (up - down) / (2 * h), abs=1e-6
    )


def test_vanilla_grad_matches_numeric_gradient_over_the_logit():
    h = 1e-6
    logit = -1.3
    up = generator_loss([sigmoid(logit + h)], non_saturating=False)
    down = generator_loss([sigmoid(logit - h)], non_saturating=False)
    assert generator_loss_grad([sigmoid(logit)], non_saturating=False)[
        0
    ] == pytest.approx((up - down) / (2 * h), abs=1e-6)


def test_generator_grad_is_divided_by_the_batch_size():
    one = generator_loss_grad([0.5])[0]
    four = generator_loss_grad([0.5] * 4)[0]
    assert four == APPROX(one / 4)


# ------------------------------------------------ optimal_discriminator
def test_optimal_discriminator_is_a_half_when_the_distributions_match():
    """Главный факт урока: G попал в p_data — и D* стал ровно 0.5 везде."""
    assert optimal_discriminator(0.3, 0.3) == APPROX(0.5)
    assert optimal_discriminator(7.5, 7.5) == APPROX(0.5)


def test_optimal_discriminator_is_certain_where_the_generator_never_goes():
    assert optimal_discriminator(0.4, 0.0) == APPROX(1.0)


def test_optimal_discriminator_is_certain_where_the_data_never_goes():
    assert optimal_discriminator(0.0, 0.4) == APPROX(0.0)


def test_optimal_discriminator_leans_toward_the_heavier_side():
    assert optimal_discriminator(0.9, 0.1) > 0.5
    assert optimal_discriminator(0.1, 0.9) < 0.5


def test_optimal_discriminator_on_a_point_nobody_visits_is_a_half():
    assert optimal_discriminator(0.0, 0.0) == APPROX(0.5)


def test_negative_density_raises_value_error():
    with pytest.raises(ValueError):
        optimal_discriminator(-0.1, 0.5)


def test_equilibrium_discriminator_gives_the_equilibrium_loss():
    """Связка двух функций: D* = 0.5 всюду даёт ровно 2 * log 2."""
    d = optimal_discriminator(0.3, 0.3)
    assert discriminator_loss([d] * 8, [d] * 8) == APPROX(2 * math.log(2))


# ------------------------------------------------------- is_mode_collapse
def test_balanced_samples_are_not_a_collapse():
    assert is_mode_collapse([-2.0, -2.0, -2.0, 2.0]) is False


def test_one_starved_mode_is_a_collapse():
    assert is_mode_collapse([-2.0] * 99 + [2.0]) is True


def test_all_samples_in_one_mode_is_the_worst_case():
    assert is_mode_collapse([2.0] * 50) is True


def test_min_share_is_the_knob():
    """Одна и та же выборка: при мягком порогe это ещё не схлопывание."""
    samples = [-2.0] * 95 + [2.0] * 5
    assert is_mode_collapse(samples, min_share=0.1) is True
    assert is_mode_collapse(samples, min_share=0.02) is False


def test_threshold_moves_the_boundary_between_modes():
    samples = [1.0] * 50 + [3.0] * 50
    assert is_mode_collapse(samples, threshold=2.0) is False
    assert is_mode_collapse(samples, threshold=0.0) is True


def test_mode_collapse_on_empty_samples_raises_value_error():
    with pytest.raises(ValueError):
        is_mode_collapse([])
