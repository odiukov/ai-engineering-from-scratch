"""Тесты к уроку «Оптимизаторы». Правь exercise.py."""

import math

import pytest

from exercise import (
    adam_step,
    adamw_step,
    bias_correct,
    momentum_step,
    noisy_grad,
    run_adam,
    run_momentum,
    run_sgd,
    sgd_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# f(w) = 100*w0^2 + 0.0001*w1^2 — овраг с разницей масштабов в миллион раз
ILL_CONDITIONED = lambda p: [200.0 * p[0], 0.0002 * p[1]]

# f(w) = 100*w0^2 + w1^2 — обычный вытянутый овраг
NARROW_VALLEY = lambda p: [200.0 * p[0], 2.0 * p[1]]


def distance(params):
    """Расстояние до минимума, который у обеих задач стоит в начале координат."""
    return math.sqrt(sum(w * w for w in params))


# ------------------------------------------------------------- sgd_step
def test_sgd_step_worked_example():
    assert sgd_step([1.0, 2.0], [0.5, -1.0], 0.1) == pytest.approx([0.95, 2.1])


def test_sgd_step_stands_still_on_zero_gradient():
    assert sgd_step([1.0, 2.0], [0.0, 0.0], 0.5) == pytest.approx([1.0, 2.0])


def test_sgd_step_moves_against_the_gradient():
    """Градиент показывает вверх, шаг делаем вниз — минус принципиален."""
    assert sgd_step([0.0], [1.0], 0.1)[0] < 0.0


def test_sgd_step_does_not_mutate_the_input():
    params = [1.0, 2.0]
    sgd_step(params, [1.0, 1.0], 0.5)
    assert params == [1.0, 2.0]


def test_sgd_step_scales_with_the_learning_rate():
    small = sgd_step([1.0], [1.0], 0.01)[0]
    big = sgd_step([1.0], [1.0], 0.1)[0]
    assert 1.0 - small == APPROX((1.0 - big) / 10)


# -------------------------------------------------------- momentum_step
def test_momentum_first_step_equals_plain_sgd():
    """Скорость стартует с нуля, так что разгона на первом шаге ещё нет."""
    params, _ = momentum_step([1.0], [1.0], [0.0], 0.1)
    assert params == pytest.approx(sgd_step([1.0], [1.0], 0.1))


def test_momentum_accumulates_the_same_direction():
    params, velocity = momentum_step([0.9], [1.0], [1.0], 0.1)
    assert velocity == pytest.approx([1.9])
    assert params == pytest.approx([0.71])


def test_momentum_cancels_alternating_gradients():
    """Градиент, меняющий знак каждый шаг, — это болтанка поперёк оврага."""
    velocity = [0.0]
    for grad in (1.0, -1.0, 1.0, -1.0, 1.0, -1.0):
        _, velocity = momentum_step([0.0], [grad], velocity, 0.1)
    assert abs(velocity[0]) < 1.0


def test_momentum_amplifies_a_steady_gradient():
    """А одинаковый градиент, наоборот, разгоняется почти до 1/(1 - beta)."""
    velocity = [0.0]
    for _ in range(50):
        _, velocity = momentum_step([0.0], [1.0], velocity, 0.1)
    assert velocity[0] == pytest.approx(10.0, rel=0.01)


def test_momentum_with_zero_beta_is_plain_sgd():
    params, _ = momentum_step([1.0], [2.0], [5.0], 0.1, beta=0.0)
    assert params == pytest.approx(sgd_step([1.0], [2.0], 0.1))


# --------------------------------------------------------- bias_correct
def test_bias_correction_restores_the_first_step():
    """m1 = 0.1*g при beta1 = 0.9 — поправка возвращает настоящий градиент."""
    assert bias_correct(0.1, 0.9, 1) == APPROX(1.0)


def test_bias_correction_fades_away_over_time():
    assert bias_correct(0.1, 0.9, 100) == pytest.approx(0.1, abs=1e-4)


def test_bias_correction_is_stronger_for_beta_close_to_one():
    """У второго момента beta2 = 0.999, и разогревается он в десять раз дольше."""
    assert bias_correct(1.0, 0.999, 1) > bias_correct(1.0, 0.9, 1)


def test_bias_correction_never_shrinks_the_moment():
    assert all(bias_correct(1.0, 0.9, t) >= 1.0 for t in range(1, 20))


# ------------------------------------------------------------ adam_step
def test_adam_first_step_is_nearly_lr_when_gradient_dominates_epsilon():
    """Точный модуль шага: lr*|g|/(|g|+eps), почти lr при |g| >> eps."""
    params, _, _ = adam_step([1.0], [1.0], [0.0], [0.0], 1, lr=0.1)
    assert params == pytest.approx([1.0 - 0.1 / (1.0 + 1e-8)], abs=1e-12)


def test_adam_step_size_is_nearly_magnitude_invariant_above_epsilon():
    """Когда оба |g| >> eps, разница первых шагов практически исчезает."""
    small, _, _ = adam_step([1.0], [1.0], [0.0], [0.0], 1, lr=0.1)
    huge, _, _ = adam_step([1.0], [1000.0], [0.0], [0.0], 1, lr=0.1)
    assert small == pytest.approx(huge, abs=1e-6)


def test_adam_epsilon_reduces_the_first_step_for_a_tiny_gradient():
    """При |g| = eps первый шаг равен половине learning rate, не всему lr."""
    params, _, _ = adam_step(
        [1.0], [1e-3], [0.0], [0.0], 1, lr=0.1, eps=1e-3
    )
    assert params == pytest.approx([0.95], abs=1e-12)


def test_adam_returns_updated_moments():
    _, m, v = adam_step([1.0], [2.0], [0.0], [0.0], 1, lr=0.1)
    assert m == pytest.approx([0.2])
    assert v == pytest.approx([0.004])


def test_adam_follows_the_sign_of_the_gradient():
    up, _, _ = adam_step([0.0], [-1.0], [0.0], [0.0], 1, lr=0.1)
    down, _, _ = adam_step([0.0], [1.0], [0.0], [0.0], 1, lr=0.1)
    assert up[0] > 0.0 > down[0]


def test_adam_stands_still_on_zero_gradient():
    params, _, _ = adam_step([1.0], [0.0], [0.0], [0.0], 1, lr=0.1)
    assert params == pytest.approx([1.0])


def test_adam_does_not_mutate_its_state():
    params, m, v = [1.0], [0.0], [0.0]
    adam_step(params, [1.0], m, v, 1, lr=0.1)
    assert (params, m, v) == ([1.0], [0.0], [0.0])


# ----------------------------------------------------------- adamw_step
def test_adamw_without_decay_is_plain_adam():
    a, _, _ = adamw_step([1.0], [1.0], [0.0], [0.0], 1, lr=0.1, weight_decay=0.0)
    b, _, _ = adam_step([1.0], [1.0], [0.0], [0.0], 1, lr=0.1)
    assert a == pytest.approx(b)


def test_adamw_shrinks_weights_even_without_a_gradient():
    """Развязка целиком: Adam стоит на месте, а вес всё равно ужимается."""
    params, _, _ = adamw_step([1.0], [0.0], [0.0], [0.0], 1, lr=0.1, weight_decay=0.5)
    assert params == pytest.approx([0.95])


def test_adamw_decay_is_proportional_to_the_weight():
    params, _, _ = adamw_step(
        [1.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 1, lr=0.1, weight_decay=0.5
    )
    assert params == pytest.approx([0.95, 3.8])


def test_adamw_decay_ignores_the_second_moment():
    """У L2-через-градиент штраф делился бы на sqrt(v) и был бы разным."""
    quiet, _, _ = adamw_step([1.0], [0.0], [0.0], [0.0], 5, lr=0.1, weight_decay=0.5)
    noisy, _, _ = adamw_step([1.0], [0.0], [0.0], [100.0], 5, lr=0.1, weight_decay=0.5)
    assert quiet == pytest.approx(noisy)


# ------------------------------------------------------------- прогоны
def test_run_sgd_matches_a_single_step():
    assert run_sgd(lambda p: [2 * p[0]], [1.0], 0.1, 1) == pytest.approx([0.8])


def test_run_momentum_matches_sgd_on_the_first_step():
    assert run_momentum(lambda p: [2 * p[0]], [1.0], 0.1, 1) == pytest.approx(
        run_sgd(lambda p: [2 * p[0]], [1.0], 0.1, 1)
    )


def test_run_adam_first_step_is_the_learning_rate():
    assert run_adam(lambda p: [2 * p[0]], [1.0], 0.1, 1) == pytest.approx([0.9], abs=1e-6)


def test_zero_steps_return_the_starting_point():
    assert run_adam(ILL_CONDITIONED, [1.0, 1.0], 0.005, 0) == pytest.approx([1.0, 1.0])


def test_runners_do_not_mutate_the_starting_point():
    start = [1.0, 1.0]
    run_sgd(NARROW_VALLEY, start, 0.001, 5)
    run_momentum(NARROW_VALLEY, start, 0.001, 5)
    run_adam(NARROW_VALLEY, start, 0.001, 5)
    assert start == [1.0, 1.0]


def test_adam_converges_faster_than_plain_sgd():
    """Одна задача, один старт, один lr, одно число шагов — Adam ближе к минимуму.

    Овраг с разницей масштабов в миллион раз: SGD обязан выбрать lr по
    самой крутой координате, иначе разойдётся, и по пологой не двигается
    вовсе. Adam делит каждую координату на её собственный второй момент.
    """
    start, lr, steps = [1.0, 1.0], 0.005, 500
    adam_left = distance(run_adam(ILL_CONDITIONED, start, lr, steps))
    sgd_left = distance(run_sgd(ILL_CONDITIONED, start, lr, steps))
    assert adam_left < sgd_left / 10


def test_sgd_alone_does_not_diverge_at_this_learning_rate():
    """Проверка честности сравнения: lr подобран так, что SGD устойчив."""
    assert distance(run_sgd(ILL_CONDITIONED, [1.0, 1.0], 0.005, 500)) < math.sqrt(2)


def test_momentum_beats_plain_sgd_in_a_narrow_valley():
    start, lr, steps = [1.0, 1.0], 0.005, 100
    assert distance(run_momentum(NARROW_VALLEY, start, lr, steps)) < distance(
        run_sgd(NARROW_VALLEY, start, lr, steps)
    )


def test_more_steps_get_closer_to_the_minimum():
    start = [1.0, 1.0]
    assert distance(run_adam(ILL_CONDITIONED, start, 0.005, 500)) < distance(
        run_adam(ILL_CONDITIONED, start, 0.005, 100)
    )


# ------------------------------------------------------------ noisy_grad
def test_noisy_grad_with_zero_sigma_is_the_clean_gradient():
    assert noisy_grad(lambda p: [1.0, -2.0], [0.0, 0.0], 0.0, seed=0) == pytest.approx(
        [1.0, -2.0]
    )


def test_noisy_grad_is_reproducible_for_one_seed():
    args = (lambda p: [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.5)
    assert noisy_grad(*args, seed=3) == noisy_grad(*args, seed=3)


def test_noisy_grad_differs_between_seeds():
    args = (lambda p: [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.5)
    assert noisy_grad(*args, seed=1) != noisy_grad(*args, seed=2)


def test_noisy_grad_keeps_the_length_of_the_gradient():
    assert len(noisy_grad(lambda p: [0.0] * 7, [0.0] * 7, 1.0, seed=0)) == 7


def test_noisy_grad_is_unbiased_on_average():
    """Шум мини-батча портит отдельный шаг, но не сдвигает направление в среднем."""
    total = 0.0
    for seed in range(400):
        total += noisy_grad(lambda p: [5.0], [0.0], 1.0, seed=seed)[0]
    assert total / 400 == pytest.approx(5.0, abs=0.2)
