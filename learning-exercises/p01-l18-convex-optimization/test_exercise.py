"""Тесты к уроку «Выпуклая оптимизация». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    check_convexity,
    convex_combination,
    eigenvalues_2x2,
    hessian_2x2,
    kkt_violations,
    newton_minimize,
    newton_step,
    segment_violation,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-4)


def flat(M):
    """pytest.approx не сравнивает вложенные списки — разворачиваем матрицу."""
    return [value for row in M for value in row]


# ------------------------------------------------------- convex_combination
def test_convex_combination_midpoint_is_the_average():
    assert convex_combination([0.0, 0.0], [10.0, 20.0], 0.5) == APPROX([5.0, 10.0])


def test_convex_combination_t_one_returns_the_first_point():
    """Ловушка порядка: t весит x, поэтому t=1 это x, а t=0 это y."""
    assert convex_combination([1.0], [3.0], 1.0) == APPROX([1.0])
    assert convex_combination([1.0], [3.0], 0.0) == APPROX([3.0])


def test_convex_combination_stays_between_the_endpoints():
    """Смысл выпуклой комбинации: точка не может вылезти за отрезок."""
    x, y = [-2.0, 7.0], [4.0, 1.0]
    for t in (0.1, 0.37, 0.9):
        for value, a, b in zip(convex_combination(x, y, t), x, y):
            assert min(a, b) - 1e-12 <= value <= max(a, b) + 1e-12


def test_convex_combination_does_not_mutate_its_inputs():
    """Ловушка: результат — новый список, аргументы остаются как были."""
    x, y = [1.0, 2.0], [3.0, 4.0]
    convex_combination(x, y, 0.5)
    assert x == [1.0, 2.0]
    assert y == [3.0, 4.0]


# --------------------------------------------------------- segment_violation
def test_segment_violation_is_negative_for_a_convex_function():
    """У параболы график лежит ПОД хордой, значит нарушение отрицательно."""
    assert segment_violation(lambda p: p[0] ** 2, [0.0], [2.0], 0.5) == APPROX(-1.0)


def test_segment_violation_is_zero_for_a_linear_function():
    """Линейная функция выпукла и вогнута сразу: хорда лежит ровно на графике."""
    assert segment_violation(lambda p: 3 * p[0] + 1, [0.0], [2.0], 0.5) == APPROX(0.0)


def test_segment_violation_is_positive_for_a_non_convex_function():
    """Ловушка знака: положительный результат = график вылез НАД хордой."""
    assert segment_violation(lambda p: p[0] ** 3, [-2.0], [0.0], 0.5) == APPROX(3.0)


# ------------------------------------------------------------ check_convexity
def test_check_convexity_accepts_the_parabola():
    assert check_convexity(lambda p: p[0] ** 2, 1) == (True, 0)


def test_check_convexity_accepts_absolute_value_despite_the_kink():
    """|x| не дифференцируема в нуле, но выпукла — определение это видит."""
    assert check_convexity(lambda p: abs(p[0]), 1) == (True, 0)


def test_check_convexity_rejects_the_sine():
    ok, violations = check_convexity(lambda p: math.sin(p[0]), 1)
    assert ok is False
    assert violations > 0


def test_check_convexity_rejects_the_saddle_xy():
    """f(x,y) = x*y — седло: гессиан [[0,1],[1,0]] имеет разные знаки."""
    ok, violations = check_convexity(lambda p: p[0] * p[1], 2)
    assert ok is False
    assert violations > 0


def test_check_convexity_is_reproducible_and_ignores_global_random():
    """Ловушка: свой random.Random(seed), а не глобальный random.

    Между двумя вызовами дёргаем глобальный поток случайности. Если функция
    берёт числа оттуда, второй ответ разойдётся с первым.
    """
    f = lambda p: math.sin(p[0])
    random.seed(1)
    first = check_convexity(f, 1)
    for _ in range(50):
        random.random()
    second = check_convexity(f, 1)
    assert first == second


# ----------------------------------------------------------------- hessian_2x2
def test_hessian_of_a_quadratic_form():
    f = lambda p: p[0] ** 2 + 3 * p[0] * p[1] + p[1] ** 2
    assert flat(hessian_2x2(f, [0.0, 0.0])) == ROUGH([2.0, 3.0, 3.0, 2.0])


def test_hessian_of_a_separable_quadratic_is_diagonal():
    """Нет слагаемого с x*y — смешанные производные нулевые."""
    f = lambda p: 5 * p[0] ** 2 + p[1] ** 2
    assert flat(hessian_2x2(f, [1.0, 1.0])) == ROUGH([10.0, 0.0, 0.0, 2.0])


def test_hessian_is_symmetric():
    f = lambda p: math.exp(p[0]) + p[0] * p[1] ** 2
    H = hessian_2x2(f, [0.3, -0.7])
    assert H[0][1] == ROUGH(H[1][0])


def test_hessian_does_not_mutate_the_point():
    """Ловушка: сдвиги на +-h делаются на копиях, сам point не трогаем."""
    point = [1.0, 2.0]
    hessian_2x2(lambda p: p[0] ** 2 + p[1] ** 2, point)
    assert point == [1.0, 2.0]


# -------------------------------------------------------------- eigenvalues_2x2
def test_eigenvalues_of_a_diagonal_matrix_are_its_diagonal():
    assert eigenvalues_2x2([[2.0, 0.0], [0.0, 3.0]]) == APPROX([2.0, 3.0])


def test_eigenvalues_are_sorted_ascending():
    """Ловушка порядка: наименьшее собственное число всегда первое."""
    assert eigenvalues_2x2([[10.0, 0.0], [0.0, 2.0]]) == APPROX([2.0, 10.0])


def test_convex_bowl_has_non_negative_eigenvalues():
    """Гессиан выпуклой чаши положительно полуопределён."""
    f = lambda p: 5 * p[0] ** 2 + p[1] ** 2
    values = eigenvalues_2x2(hessian_2x2(f, [1.0, 1.0]))
    assert values[0] > 0
    assert values[1] / values[0] == pytest.approx(5.0, rel=1e-4)


def test_saddle_has_eigenvalues_of_mixed_signs():
    """Разные знаки — седло, определение выпуклости обязано его отвергнуть."""
    values = eigenvalues_2x2([[2.0, 3.0], [3.0, 2.0]])
    assert values == APPROX([-1.0, 5.0])
    assert values[0] < 0 < values[1]


def test_eigenvalues_reject_a_non_symmetric_matrix():
    """Ловушка: при отрицательном дискриминанте sqrt считать нельзя."""
    with pytest.raises(ValueError):
        eigenvalues_2x2([[0.0, 1.0], [-1.0, 0.0]])


# ------------------------------------------------------------------ newton_step
def test_newton_step_lands_on_the_minimum_of_a_quadratic():
    """Для квадратичной функции квадратичное приближение точное — шаг один."""
    assert newton_step([1.0, 1.0], [10.0, 2.0], [[10.0, 0.0], [0.0, 2.0]]) == APPROX(
        [0.0, 0.0]
    )


def test_newton_step_with_zero_gradient_stays_put():
    assert newton_step([4.0, -3.0], [0.0, 0.0], [[2.0, 0.0], [0.0, 2.0]]) == APPROX(
        [4.0, -3.0]
    )


def test_newton_step_is_invariant_to_rescaling_the_objective():
    """Умножили f на 7 — градиент и гессиан выросли в 7 раз, шаг тот же.

    Ловушка: это работает только если делить на гессиан. Умножение на H
    вместо H^(-1) увеличило бы шаг в 49 раз.
    """
    grad, H = [3.0, -4.0], [[2.0, 1.0], [1.0, 5.0]]
    plain = newton_step([1.0, 1.0], grad, H)
    scaled = newton_step(
        [1.0, 1.0],
        [7 * g for g in grad],
        [[7 * v for v in row] for row in H],
    )
    assert scaled == APPROX(plain)


def test_newton_step_rejects_a_singular_hessian():
    """Ловушка: нулевой определитель — это не деление на ноль, а ValueError."""
    with pytest.raises(ValueError):
        newton_step([1.0, 1.0], [1.0, 1.0], [[1.0, 1.0], [1.0, 1.0]])


# -------------------------------------------------------------- newton_minimize
def test_newton_minimize_solves_a_quadratic_in_one_step():
    result = newton_minimize(
        lambda p: [10 * p[0], 2 * p[1]],
        lambda p: [[10.0, 0.0], [0.0, 2.0]],
        [10.0, 10.0],
        steps=1,
    )
    assert result == APPROX([0.0, 0.0])


def test_newton_minimize_finds_a_shifted_minimum():
    result = newton_minimize(
        lambda p: [2 * (p[0] - 3), 2 * (p[1] + 1)],
        lambda p: [[2.0, 0.0], [0.0, 2.0]],
        [0.0, 0.0],
    )
    assert result == APPROX([3.0, -1.0])


def test_newton_minimize_beats_gradient_descent_in_an_elongated_valley():
    """Ради этого урок и написан: число обусловленности 50 душит спуск.

    Ньютону хватает одного шага, градиентному спуску мало и двухсот.
    """
    f = lambda p: 50 * p[0] ** 2 + p[1] ** 2
    grad = lambda p: [100 * p[0], 2 * p[1]]
    hess = lambda p: [[100.0, 0.0], [0.0, 2.0]]

    fast = newton_minimize(grad, hess, [10.0, 10.0], steps=1)

    slow = [10.0, 10.0]
    for _ in range(200):
        g = grad(slow)
        slow = [slow[0] - 0.005 * g[0], slow[1] - 0.005 * g[1]]

    assert f(fast) < f(slow)
    assert f(slow) > 1.0


def test_newton_minimize_returns_an_already_optimal_start_untouched():
    """Ловушка: критерий остановки проверяется ДО шага, а не после."""
    result = newton_minimize(
        lambda p: [0.0, 0.0],
        lambda p: [[0.0, 0.0], [0.0, 0.0]],
        [2.0, 5.0],
    )
    assert result == APPROX([2.0, 5.0])


def test_newton_minimize_does_not_mutate_the_start():
    start = [10.0, 10.0]
    newton_minimize(
        lambda p: [10 * p[0], 2 * p[1]],
        lambda p: [[10.0, 0.0], [0.0, 2.0]],
        start,
    )
    assert start == [10.0, 10.0]


# --------------------------------------------------------------- kkt_violations
GRAD_SQ = lambda p: [2 * p[0], 2 * p[1]]
G_LINE = lambda p: 1 - p[0] - p[1]
GG_LINE = lambda p: [-1.0, -1.0]


def test_kkt_is_clean_at_the_constrained_optimum():
    """min x^2+y^2 при x+y >= 1: решение (0.5, 0.5), lambda = 1."""
    v = kkt_violations(GRAD_SQ, [(G_LINE, GG_LINE)], [0.5, 0.5], [1.0])
    assert v["stationarity"] == APPROX(0.0)
    assert v["primal"] == APPROX(0.0)
    assert v["dual"] == APPROX(0.0)
    assert v["slackness"] == APPROX(0.0)


def test_kkt_flags_an_infeasible_point():
    """Точка (0,0) нарушает x+y >= 1 ровно на единицу."""
    v = kkt_violations(GRAD_SQ, [(G_LINE, GG_LINE)], [0.0, 0.0], [0.0])
    assert v["primal"] == APPROX(1.0)
    assert v["stationarity"] == APPROX(0.0)


def test_kkt_flags_a_negative_multiplier():
    """Отрицательная lambda запрещена: ограничение начинает тянуть не в ту сторону."""
    v = kkt_violations(GRAD_SQ, [(G_LINE, GG_LINE)], [0.5, 0.5], [-1.0])
    assert v["dual"] == APPROX(1.0)
    assert v["stationarity"] == APPROX(2.0)


def test_kkt_flags_broken_complementary_slackness():
    """Ловушка: нежёсткость меряется модулем |lambda*g|.

    Ограничение x+y <= 5 в точке (0,0) неактивно, g = -5, значит его
    множитель обязан быть нулём. Взяли 2 — нарушение равно 10.
    """
    g = lambda p: p[0] + p[1] - 5
    gg = lambda p: [1.0, 1.0]
    v = kkt_violations(GRAD_SQ, [(g, gg)], [0.0, 0.0], [2.0])
    assert v["slackness"] == APPROX(10.0)
    assert v["primal"] == APPROX(0.0)


def test_kkt_without_constraints_is_just_a_zero_gradient():
    """Без ограничений KKT вырождается в «градиент равен нулю»."""
    assert kkt_violations(GRAD_SQ, [], [3.0, 0.0], [])["stationarity"] == APPROX(6.0)
    assert kkt_violations(GRAD_SQ, [], [0.0, 0.0], [])["stationarity"] == APPROX(0.0)
