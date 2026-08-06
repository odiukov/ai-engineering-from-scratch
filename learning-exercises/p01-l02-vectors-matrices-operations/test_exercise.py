"""Тесты к уроку «Векторы, матрицы и операции». Правь exercise.py."""

import pytest

from exercise import hadamard, identity, is_symmetric, matmul, trace, transpose


# ------------------------------------------------------------- transpose
def test_transpose_rectangular():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_transpose_twice_returns_original():
    M = [[1, 2, 3], [4, 5, 6]]
    assert transpose(transpose(M)) == M


def test_transpose_returns_lists_not_tuples():
    """zip отдаёт кортежи — не забудь превратить их в списки."""
    result = transpose([[1, 2], [3, 4]])
    assert all(isinstance(row, list) for row in result)


# ---------------------------------------------------------------- matmul
def test_matmul_by_identity_changes_nothing():
    A = [[1, 2], [3, 4]]
    assert matmul(A, identity(2)) == A


def test_matmul_row_by_column():
    assert matmul([[1, 2]], [[3], [4]]) == [[11]]


def test_matmul_shapes_2x3_by_3x2():
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 0], [0, 1], [1, 1]]
    assert matmul(A, B) == [[4, 5], [10, 11]]


def test_matmul_is_not_commutative():
    """Ключевой факт: порядок важен. Повернуть-потом-растянуть не равно
    растянуть-потом-повернуть."""
    rot = [[0, -1], [1, 0]]
    scale = [[2, 0], [0, 1]]
    assert matmul(rot, scale) != matmul(scale, rot)


# -------------------------------------------------------------- identity
def test_identity_shape_and_content():
    assert identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_identity_of_one():
    assert identity(1) == [[1]]


# ----------------------------------------------------------------- trace
def test_trace_basic():
    assert trace([[1, 2], [3, 4]]) == 5


def test_trace_of_identity_is_size():
    assert trace(identity(5)) == 5


# ---------------------------------------------------------- is_symmetric
def test_symmetric_true():
    assert is_symmetric([[1, 2], [2, 1]]) is True


def test_symmetric_false():
    assert is_symmetric([[1, 2], [3, 1]]) is False


def test_identity_is_symmetric():
    assert is_symmetric(identity(4)) is True


# -------------------------------------------------------------- hadamard
def test_hadamard_scales_elementwise():
    assert hadamard([[1, 2], [3, 4]], [[10, 10], [10, 10]]) == [[10, 20], [30, 40]]


def test_hadamard_differs_from_matmul():
    """Разные операции. Путать их — классическая ошибка."""
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    assert hadamard(A, B) != matmul(A, B)


def test_hadamard_with_zero_mask_zeroes_everything():
    """Так работает маска: где ноль — там всё стирается."""
    assert hadamard([[1, 2], [3, 4]], [[1, 0], [0, 1]]) == [[1, 0], [0, 4]]
