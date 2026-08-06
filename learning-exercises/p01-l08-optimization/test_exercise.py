"""Тесты к уроку «Оптимизация». Правь exercise.py."""

import math

import pytest

from exercise import (
    adam_step,
    cosine_annealing,
    exponential_decay,
    minimize_adam,
    minimize_momentum,
    rosenbrock,
    rosenbrock_gradient,
    sgd_momentum_step,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------- rosenbrock
def test_rosenbrock_is_zero_at_its_minimum():
    assert rosenbrock([1.0, 1.0]) == APPROX(0.0)


def test_rosenbrock_at_the_origin():
    assert rosenbrock([0.0, 0.0]) == APPROX(1.0)


def test_rosenbrock_at_the_classic_starting_point():
    assert rosenbrock([-1.0, 1.0]) == APPROX(4.0)


def test_rosenbrock_is_never_negative():
    """Сумма двух квадратов — отрицательной быть не может."""
    points = [[-2.0, 3.0], [0.5, 0.1], [1.5, 2.5], [0.0, -1.0]]
    assert all(rosenbrock(p) >= 0 for p in points)


def test_rosenbrock_valley_runs_along_the_parabola_y_equals_x_squared():
    """На линии y = x^2 второе слагаемое зануляется — это и есть дно оврага."""
    on_the_floor = rosenbrock([1.1, 1.21])
    off_the_floor = rosenbrock([1.1, 1.0])
    assert on_the_floor < 0.02
    assert off_the_floor > 100 * on_the_floor


# ------------------------------------------------------ rosenbrock_gradient
def test_rosenbrock_gradient_vanishes_at_the_minimum():
    assert rosenbrock_gradient([1.0, 1.0]) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_rosenbrock_gradient_at_the_origin():
    assert rosenbrock_gradient([0.0, 0.0]) == pytest.approx([-2.0, 0.0], abs=1e-9)


def test_rosenbrock_gradient_uses_four_hundred_not_two_hundred():
    """В точке (1, 2) разрыв y - x^2 равен 1, и df/dx обязан быть -400."""
    assert rosenbrock_gradient([1.0, 2.0]) == pytest.approx([-400.0, 200.0], abs=1e-9)


def test_rosenbrock_gradient_matches_the_numeric_derivative():
    h = 1e-6
    x, y = 0.3, -0.7
    num_x = (rosenbrock([x + h, y]) - rosenbrock([x - h, y])) / (2 * h)
    num_y = (rosenbrock([x, y + h]) - rosenbrock([x, y - h])) / (2 * h)
    assert rosenbrock_gradient([x, y]) == pytest.approx([num_x, num_y], abs=1e-4)


def test_rosenbrock_gradient_dy_sign_follows_the_gap():
    """y выше параболы — df/dy положителен, ниже — отрицателен."""
    assert rosenbrock_gradient([1.0, 2.0])[1] > 0
    assert rosenbrock_gradient([1.0, 0.0])[1] < 0


# ---------------------------------------------------------- sgd_momentum_step
def test_sgd_momentum_first_step_from_zero_velocity():
    assert sgd_momentum_step([1.0], [2.0], [0.0], 0.1) == (
        pytest.approx([0.8], abs=1e-9),
        pytest.approx([2.0], abs=1e-9),
    )


def test_sgd_momentum_second_step_accumulates_speed():
    params, velocity = sgd_momentum_step([0.8], [2.0], [2.0], 0.1)
    assert velocity == pytest.approx([3.8], abs=1e-9)
    assert params == pytest.approx([0.42], abs=1e-9)


def test_sgd_momentum_with_zero_momentum_is_plain_gradient_descent():
    params, _ = sgd_momentum_step([3.0, 4.0], [6.0, 8.0], [0.0, 0.0], 0.1, momentum=0.0)
    assert params == pytest.approx([2.4, 3.2], abs=1e-9)


def test_sgd_momentum_moves_against_the_gradient():
    params, _ = sgd_momentum_step([0.0], [1.0], [0.0], 0.1)
    assert params[0] < 0


def test_sgd_momentum_velocity_saturates_at_g_over_one_minus_beta():
    """Постоянный градиент 1.0 при momentum 0.9 разгоняет скорость до 10."""
    velocity = [0.0]
    for _ in range(200):
        _, velocity = sgd_momentum_step([0.0], [1.0], velocity, 0.1, 0.9)
    assert velocity[0] == pytest.approx(10.0, abs=1e-6)


def test_sgd_momentum_does_not_mutate_its_inputs():
    params, grads, velocity = [1.0], [2.0], [3.0]
    sgd_momentum_step(params, grads, velocity, 0.1)
    assert (params, grads, velocity) == ([1.0], [2.0], [3.0])


# ------------------------------------------------------------------ adam_step
def test_adam_first_step_length_is_the_learning_rate():
    """Bias correction: без него первый шаг был бы в 10 раз короче."""
    params, _, _ = adam_step([1.0], [2.0], [0.0], [0.0], 1, lr=0.1)
    assert params[0] == pytest.approx(0.9, abs=1e-3)


def test_adam_first_step_ignores_the_scale_of_a_huge_gradient():
    params, _, _ = adam_step([1.0], [1e6], [0.0], [0.0], 1, lr=0.1)
    assert params[0] == pytest.approx(0.9, abs=1e-3)


def test_adam_first_step_ignores_the_scale_of_a_tiny_gradient():
    """Именно это и значит «свой learning rate у каждого веса»."""
    params, _, _ = adam_step([1.0], [1e-6], [0.0], [0.0], 1, lr=0.1)
    assert params[0] == pytest.approx(0.9, abs=1e-2)


def test_adam_updates_both_moments():
    _, m, v = adam_step([1.0], [2.0], [0.0], [0.0], 1)
    assert m == pytest.approx([0.2], abs=1e-9)
    assert v == pytest.approx([0.004], abs=1e-9)


def test_adam_with_zero_gradient_stays_put():
    params, _, _ = adam_step([5.0], [0.0], [0.0], [0.0], 1, lr=0.1)
    assert params[0] == pytest.approx(5.0, abs=1e-9)


def test_adam_does_not_mutate_its_inputs():
    params, grads, m, v = [1.0], [2.0], [0.0], [0.0]
    adam_step(params, grads, m, v, 1)
    assert (params, grads, m, v) == ([1.0], [2.0], [0.0], [0.0])


# ----------------------------------------------------------- minimize_momentum
def test_minimize_momentum_finds_the_bottom_of_a_parabola():
    result = minimize_momentum(lambda p: [2 * p[0]], [10.0], 0.05)
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_minimize_momentum_beats_plain_gradient_descent_on_rosenbrock():
    """Тот же lr и то же число шагов — момент проходит овраг, спуск застревает."""
    with_momentum = minimize_momentum(rosenbrock_gradient, [-1.0, 1.0], 1e-4, 0.9, 5000)
    without = minimize_momentum(rosenbrock_gradient, [-1.0, 1.0], 1e-4, 0.0, 5000)
    assert rosenbrock(with_momentum) < rosenbrock(without)


def test_minimize_momentum_with_zero_steps_returns_the_start():
    assert minimize_momentum(lambda p: [2 * p[0]], [7.0], 0.1, steps=0) == [7.0]


def test_minimize_momentum_does_not_mutate_start():
    start = [10.0]
    minimize_momentum(lambda p: [2 * p[0]], start, 0.05)
    assert start == [10.0]


# --------------------------------------------------------------- minimize_adam
def test_minimize_adam_finds_the_bottom_of_a_parabola():
    result = minimize_adam(lambda p: [2 * p[0]], [10.0], 0.1, 500)
    assert result[0] == pytest.approx(0.0, abs=1e-4)


def test_minimize_adam_reaches_the_rosenbrock_minimum():
    result = minimize_adam(rosenbrock_gradient, [-1.0, 1.0], 0.01, 5000)
    assert result == pytest.approx([1.0, 1.0], abs=1e-3)


def test_minimize_adam_handles_wildly_different_gradient_scales():
    """Градиенты по осям отличаются в миллион раз — один общий lr тут бессилен."""
    grad = lambda p: [2000 * p[0], 0.002 * p[1]]
    adam = minimize_adam(grad, [1.0, 1.0], 0.01, 3000)
    momentum = minimize_momentum(grad, [1.0, 1.0], 1e-4, 0.9, 3000)
    assert abs(adam[1]) < 1e-3
    assert abs(momentum[1]) > 0.9


def test_minimize_adam_does_not_mutate_start():
    start = [10.0]
    minimize_adam(lambda p: [2 * p[0]], start, 0.1, 100)
    assert start == [10.0]


# ----------------------------------------------------------- exponential_decay
def test_exponential_decay_at_step_zero_returns_the_initial_rate():
    """Ловушка: на нулевом шаге lr обязан быть исходным, а не уже урезанным."""
    assert exponential_decay(0.1, 0) == APPROX(0.1)


def test_exponential_decay_after_a_thousand_steps():
    assert exponential_decay(0.1, 1000) == pytest.approx(0.0367695, abs=1e-6)


def test_exponential_decay_with_an_explicit_factor():
    assert exponential_decay(0.1, 10, 0.5) == APPROX(0.1 / 1024)


def test_exponential_decay_is_monotonically_decreasing():
    values = [exponential_decay(0.1, s) for s in range(0, 500, 50)]
    assert all(a > b for a, b in zip(values, values[1:]))


# ----------------------------------------------------------- cosine_annealing
def test_cosine_annealing_starts_at_lr_max():
    assert cosine_annealing(0.1, 0.0, 0, 1000) == APPROX(0.1)


def test_cosine_annealing_ends_at_lr_min():
    assert cosine_annealing(0.1, 0.001, 1000, 1000) == APPROX(0.001)


def test_cosine_annealing_halfway_is_the_average_of_the_bounds():
    assert cosine_annealing(0.1, 0.02, 500, 1000) == APPROX(0.06)


def test_cosine_annealing_never_leaves_the_bounds():
    assert all(
        0.02 <= cosine_annealing(0.1, 0.02, s, 1000) <= 0.1 for s in range(0, 1001, 25)
    )


def test_cosine_annealing_is_monotonically_decreasing():
    values = [cosine_annealing(0.1, 0.0, s, 1000) for s in range(0, 1001, 50)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_cosine_annealing_falls_slowly_at_the_start():
    """Через четверть пути потеряно меньше 15% — этим косинус и ценен."""
    assert cosine_annealing(0.1, 0.0, 250, 1000) == pytest.approx(
        0.05 * (1 + math.cos(math.pi / 4)), abs=1e-9
    )
    assert cosine_annealing(0.1, 0.0, 250, 1000) > 0.085
