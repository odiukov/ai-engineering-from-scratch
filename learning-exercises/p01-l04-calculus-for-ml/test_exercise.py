"""Тесты к уроку «Производные и градиенты». Правь exercise.py."""

import math

import pytest

from exercise import derivative, descent_step, gradient, is_close_to_minimum, minimize

APPROX = lambda x: pytest.approx(x, abs=1e-4)


# ------------------------------------------------------------- derivative
def test_derivative_of_square():
    assert derivative(lambda t: t * t, 3) == APPROX(6.0)


def test_derivative_of_constant_is_zero():
    assert derivative(lambda t: 5, 42) == APPROX(0.0)


def test_derivative_of_line_is_its_slope():
    assert derivative(lambda t: 3 * t + 7, 100) == APPROX(3.0)


def test_derivative_of_sin_is_cos():
    assert derivative(math.sin, 1.0) == APPROX(math.cos(1.0))


def test_derivative_is_negative_on_a_falling_slope():
    """Знак несёт смысл: функция убывает — производная отрицательна."""
    assert derivative(lambda t: -2 * t, 0) < 0


def test_derivative_uses_central_difference():
    """Односторонняя разность на кубе даёт заметную ошибку, центральная — нет.

    Для t^3 в точке 1 точный ответ 3. Односторонняя ошибётся примерно на h,
    центральная — на h^2. Этот допуск проходит только центральная.
    """
    assert derivative(lambda t: t ** 3, 1.0) == pytest.approx(3.0, abs=1e-8)


# --------------------------------------------------------------- gradient
def test_gradient_of_paraboloid():
    assert gradient(lambda p: p[0] ** 2 + p[1] ** 2, [3, 4]) == APPROX([6.0, 8.0])


def test_gradient_ignores_unused_variables():
    """Функция не зависит от y — частная производная по y равна нулю."""
    assert gradient(lambda p: p[0] ** 2, [2.0, 99.0]) == APPROX([4.0, 0.0])


def test_gradient_works_in_three_dimensions():
    g = gradient(lambda p: p[0] + 2 * p[1] + 3 * p[2], [0.0, 0.0, 0.0])
    assert g == APPROX([1.0, 2.0, 3.0])


def test_gradient_does_not_mutate_the_input_point():
    """Ловушка: если править сам point, посчитается ерунда."""
    point = [3.0, 4.0]
    gradient(lambda p: p[0] ** 2 + p[1] ** 2, point)
    assert point == [3.0, 4.0]


# ----------------------------------------------------------- descent_step
def test_descent_step_moves_against_the_gradient():
    assert descent_step([3.0, 4.0], [6.0, 8.0], 0.1) == APPROX([2.4, 3.2])


def test_descent_step_with_zero_gradient_stays_put():
    assert descent_step([1.0, 2.0], [0.0, 0.0], 0.5) == APPROX([1.0, 2.0])


def test_descent_step_sign_is_minus_not_plus():
    """Плюс вместо минуса — самая частая ошибка: модель будет расти, не падать."""
    moved = descent_step([0.0], [1.0], 0.1)
    assert moved[0] < 0


# --------------------------------------------------------------- minimize
def test_minimize_finds_the_bottom_of_a_parabola():
    assert minimize(lambda p: (p[0] - 5) ** 2, [0.0])[0] == pytest.approx(5.0, abs=1e-2)


def test_minimize_in_two_dimensions():
    result = minimize(lambda p: (p[0] - 1) ** 2 + (p[1] + 2) ** 2, [0.0, 0.0])
    assert result == pytest.approx([1.0, -2.0], abs=1e-2)


def test_minimize_does_not_mutate_start():
    start = [0.0]
    minimize(lambda p: p[0] ** 2, start)
    assert start == [0.0]


def test_minimize_decreases_the_loss():
    f = lambda p: (p[0] - 3) ** 2 + 1
    assert f(minimize(f, [10.0])) < f([10.0])


# ------------------------------------------------------ is_close_to_minimum
def test_at_the_minimum():
    assert is_close_to_minimum(lambda p: p[0] ** 2, [0.0]) is True


def test_away_from_the_minimum():
    assert is_close_to_minimum(lambda p: p[0] ** 2, [5.0]) is False


def test_flat_maximum_also_reads_as_zero_gradient():
    """Нулевой градиент не доказывает минимум — в максимуме он тоже нулевой."""
    assert is_close_to_minimum(lambda p: -(p[0] ** 2), [0.0]) is True
