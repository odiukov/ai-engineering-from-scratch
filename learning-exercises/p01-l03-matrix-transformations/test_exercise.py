"""Тесты к уроку «Матричные преобразования». Правь exercise.py."""

import math

import pytest

from exercise import (
    apply,
    compose,
    determinant_2x2,
    eigenvalues_2x2,
    is_eigenvector,
    rotation_matrix,
    scaling_matrix,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — сравниваем матрицы плоскими."""
    return [x for row in M for x in row]


# ------------------------------------------------------- rotation_matrix
def test_rotation_zero_is_identity():
    assert flat(rotation_matrix(0)) == APPROX([1.0, 0.0, 0.0, 1.0])


def test_rotation_90():
    assert flat(rotation_matrix(90)) == APPROX([0.0, -1.0, 1.0, 0.0])


def test_rotation_takes_degrees_not_radians():
    """Ловушка: math.cos ждёт радианы, а на вход даны градусы."""
    assert flat(rotation_matrix(180)) == APPROX([-1.0, 0.0, 0.0, -1.0])


def test_rotation_preserves_length():
    """Поворот не растягивает: длина вектора не меняется."""
    v = apply(rotation_matrix(37), [3, 4])
    assert math.hypot(*v) == APPROX(5.0)


# -------------------------------------------------------- scaling_matrix
def test_scaling_shape():
    assert scaling_matrix(2, 3) == [[2, 0], [0, 3]]


def test_scaling_applied():
    assert apply(scaling_matrix(2, 3), [1, 1]) == APPROX([2, 3])


# ------------------------------------------------------------------ apply
def test_apply_rotation():
    assert apply([[0, -1], [1, 0]], [3, 1]) == APPROX([-1, 3])


def test_apply_identity():
    assert apply([[1, 0], [0, 1]], [7, -2]) == APPROX([7, -2])


# ---------------------------------------------------------------- compose
def test_compose_matches_sequential_application():
    A, B, v = rotation_matrix(90), scaling_matrix(2, 3), [1, 1]
    assert apply(compose(A, B), v) == APPROX(apply(A, apply(B, v)))


def test_compose_order_matters():
    """Повернуть-потом-растянуть не то же, что растянуть-потом-повернуть."""
    A, B = rotation_matrix(90), scaling_matrix(2, 1)
    assert apply(compose(A, B), [1, 0]) != APPROX(apply(compose(B, A), [1, 0]))


# --------------------------------------------------------- determinant_2x2
def test_determinant_identity_is_one():
    assert determinant_2x2([[1, 0], [0, 1]]) == APPROX(1)


def test_determinant_scaling_multiplies_area():
    assert determinant_2x2([[2, 0], [0, 3]]) == APPROX(6)


def test_determinant_rank_one_is_zero():
    assert determinant_2x2([[1, 2], [2, 4]]) == APPROX(0)


def test_determinant_rotation_is_one():
    """Поворот не меняет площадь."""
    assert determinant_2x2(rotation_matrix(37)) == APPROX(1)


def test_determinant_can_be_negative():
    """Отражение переворачивает ориентацию — определитель отрицательный."""
    assert determinant_2x2([[0, 1], [1, 0]]) == APPROX(-1)


# --------------------------------------------------------- eigenvalues_2x2
def test_eigenvalues_of_diagonal_are_the_diagonal():
    assert eigenvalues_2x2([[2, 0], [0, 3]]) == APPROX((2.0, 3.0))


def test_eigenvalues_sorted_ascending():
    """Даже если на диагонали числа стоят по убыванию."""
    assert eigenvalues_2x2([[5, 0], [0, 1]]) == APPROX((1.0, 5.0))


def test_eigenvalues_of_rank_one_include_zero():
    """Схлопывание всегда даёт нулевое собственное значение."""
    assert eigenvalues_2x2([[1, 2], [2, 4]]) == APPROX((0.0, 5.0))


def test_eigenvalue_sum_equals_trace():
    M = [[3, 1], [2, 4]]
    lo, hi = eigenvalues_2x2(M)
    assert lo + hi == APPROX(M[0][0] + M[1][1])


def test_eigenvalue_product_equals_determinant():
    M = [[3, 1], [2, 4]]
    lo, hi = eigenvalues_2x2(M)
    assert lo * hi == APPROX(determinant_2x2(M))


# ----------------------------------------------------------- is_eigenvector
def test_eigenvector_of_diagonal():
    assert is_eigenvector([[2, 0], [0, 3]], [1, 0]) is True


def test_non_eigenvector_gets_rotated_off_its_line():
    assert is_eigenvector([[2, 0], [0, 3]], [1, 1]) is False


def test_scaled_eigenvector_is_still_an_eigenvector():
    """Собственный вектор задаёт направление, длина роли не играет."""
    assert is_eigenvector([[2, 0], [0, 3]], [7, 0]) is True


def test_zero_vector_is_not_an_eigenvector():
    """Соглашение: нулевой вектор исключён, иначе им был бы любой M."""
    assert is_eigenvector([[2, 0], [0, 3]], [0, 0]) is False


def test_eigenvector_with_leading_zero_component():
    """Ловушка: нельзя делить на первую компоненту не глядя — она может быть 0."""
    assert is_eigenvector([[2, 0], [0, 3]], [0, 5]) is True
