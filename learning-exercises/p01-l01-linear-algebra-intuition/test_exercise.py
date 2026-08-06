"""Тесты к уроку 01. Не редактируй этот файл — правь exercise.py."""

import math

import pytest

from exercise import (
    angle_between,
    cosine_similarity,
    dot,
    is_invertible_2x2,
    magnitude,
    matvec,
    most_similar_pair,
    project,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
# у углов допуск мягче: acos рядом с ±1 теряет точность по природе арифметики
ANGLE = lambda x: pytest.approx(x, abs=1e-5)


# --------------------------------------------------------------- magnitude
def test_magnitude_egyptian_triangle():
    assert magnitude([3, 4]) == APPROX(5.0)


def test_magnitude_works_in_any_dimension():
    assert magnitude([1, 1, 1]) == APPROX(math.sqrt(3))
    assert magnitude([1, 1, 1, 1, 1]) == APPROX(math.sqrt(5))


def test_magnitude_ignores_sign():
    """Длина не бывает отрицательной: [-3, -4] так же далеко, как [3, 4]."""
    assert magnitude([-3, -4]) == APPROX(5.0)


def test_magnitude_of_zero_vector():
    assert magnitude([0, 0, 0]) == APPROX(0.0)


# --------------------------------------------------------------------- dot
def test_dot_basic():
    assert dot([1, 2, 3], [4, 5, 6]) == APPROX(32)


def test_dot_perpendicular_is_zero():
    assert dot([2, 3], [3, -2]) == APPROX(0)


def test_dot_opposite_is_negative():
    assert dot([2, 0], [-3, 0]) == APPROX(-6)


def test_dot_is_symmetric():
    """a·b и b·a — одно и то же."""
    a, b = [1, -2, 3], [4, 5, -6]
    assert dot(a, b) == APPROX(dot(b, a))


# ------------------------------------------------------- cosine_similarity
def test_cosine_same_direction_is_one():
    assert cosine_similarity([1, 0], [1, 0]) == APPROX(1.0)


def test_cosine_perpendicular_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity([1, 0], [-1, 0]) == APPROX(-1.0)


def test_cosine_ignores_length():
    """Главное свойство: удлинил вектор — косинус не изменился."""
    short = cosine_similarity([1, 2, 3], [4, 5, 6])
    long = cosine_similarity([100, 200, 300], [4, 5, 6])
    assert short == APPROX(long)


# ----------------------------------------------------------- angle_between
def test_angle_perpendicular():
    assert angle_between([1, 0], [0, 1]) == ANGLE(90.0)


def test_angle_diagonal():
    assert angle_between([1, 0], [1, 1]) == ANGLE(45.0)


def test_angle_same_vector_is_zero():
    """Ловушка на округление: косинус выйдет 1.0000000002, acos от такого падает."""
    assert angle_between([1, 2, 3], [1, 2, 3]) == ANGLE(0.0)


def test_angle_opposite_is_180():
    assert angle_between([1, 1], [-1, -1]) == ANGLE(180.0)


# ----------------------------------------------------------------- project
def test_project_drops_the_y_component():
    assert project([3, 4], [1, 0]) == APPROX([3.0, 0.0])


def test_project_perpendicular_collapses_to_zero():
    assert project([0, 5], [1, 0]) == APPROX([0.0, 0.0])


def test_project_onto_longer_vector_gives_same_answer():
    """Длина onto не влияет — важно только его направление."""
    assert project([3, 4], [7, 0]) == APPROX([3.0, 0.0])


def test_projection_residual_is_perpendicular():
    """Ключевое свойство: то, что тень потеряла, всегда перпендикулярно."""
    a, b = [3, 4], [1, 1]
    p = project(a, b)
    residual = [a[i] - p[i] for i in range(len(a))]
    assert dot(residual, b) == APPROX(0.0)


# ------------------------------------------------------------------ matvec
def test_matvec_rotation_90():
    assert matvec([[0, -1], [1, 0]], [3, 1]) == APPROX([-1, 3])


def test_matvec_scaling():
    assert matvec([[2, 0], [0, 3]], [1, 1]) == APPROX([2, 3])


def test_matvec_identity_changes_nothing():
    assert matvec([[1, 0], [0, 1]], [7, -2]) == APPROX([7, -2])


def test_matvec_non_square_reduces_dimensions():
    """Слой нейросети: 3 числа на входе, 2 на выходе."""
    M = [[1, 0, 0], [0, 1, 1]]
    assert matvec(M, [5, 2, 3]) == APPROX([5, 5])


def test_matvec_rank_one_collides():
    """Разные точки, одинаковый результат — вот почему обратно не восстановить."""
    M = [[1, 2], [2, 4]]
    assert matvec(M, [2, 0]) == APPROX(matvec(M, [0, 1]))


# -------------------------------------------------------- is_invertible_2x2
def test_invertible_rotation():
    assert is_invertible_2x2([[0, -1], [1, 0]]) is True


def test_not_invertible_rank_one():
    assert is_invertible_2x2([[1, 2], [2, 4]]) is False


def test_not_invertible_zero_matrix():
    assert is_invertible_2x2([[0, 0], [0, 0]]) is False


def test_invertible_handles_floats_near_zero():
    """Определитель = 1e-15, то есть по сути ноль. Не сравнивай с нулём напрямую."""
    assert is_invertible_2x2([[1.0, 2.0], [2.0, 4.0 + 1e-15]]) is False


# -------------------------------------------------------- most_similar_pair
def test_most_similar_pair_obvious():
    assert most_similar_pair([[1, 0], [0, 1], [0.9, 0.1]]) == (0, 2)


def test_most_similar_pair_ignores_length():
    """Вектор 2 — это вектор 0, растянутый в 10 раз. Направление то же."""
    assert most_similar_pair([[1, 1], [1, -1], [10, 10]]) == (0, 2)


def test_most_similar_pair_returns_sorted_indices():
    i, j = most_similar_pair([[0, 1], [1, 0], [0.99, 0.01], [-1, 0]])
    assert i < j, "верни (i, j) так, чтобы i было меньше j"
    assert (i, j) == (1, 2)
