"""Тесты к уроку «Сингулярное разложение». Правь exercise.py."""

import math

import pytest

from exercise import (
    condition_number,
    frobenius_norm,
    outer,
    power_iteration,
    pseudoinverse,
    reconstruct,
    svd,
    top_singular_triple,
)

APPROX = lambda x: pytest.approx(x, abs=1e-6)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу."""
    return [x for row in M for x in row]


def matvec(M, v):
    return [sum(a * b for a, b in zip(row, v)) for row in M]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------- outer
def test_outer_builds_the_expected_matrix():
    assert flat(outer([1, 2], [10, 20, 30])) == APPROX([10, 20, 30, 20, 40, 60])


def test_outer_shape_is_len_u_by_len_v():
    M = outer([1, 2, 3], [1, 1])
    assert len(M) == 3 and len(M[0]) == 2


def test_outer_is_not_symmetric_in_its_arguments():
    """Порядок аргументов меняет форму: outer(u, v) транспонирована к outer(v, u)."""
    direct = outer([1, 2], [3, 4, 5])
    swapped = outer([3, 4, 5], [1, 2])
    assert (len(direct), len(direct[0])) == (2, 3)
    assert (len(swapped), len(swapped[0])) == (3, 2)


def test_outer_result_has_rank_one_so_rows_are_proportional():
    """У матрицы ранга 1 вторая строка кратна первой."""
    M = outer([1.0, 4.0], [2.0, 5.0, 7.0])
    ratios = [b / a for a, b in zip(M[0], M[1])]
    assert ratios == APPROX([4.0, 4.0, 4.0])


# ---------------------------------------------------------- frobenius_norm
def test_frobenius_norm_of_a_single_row_is_vector_length():
    assert frobenius_norm([[3, 4]]) == APPROX(5.0)


def test_frobenius_norm_of_identity_is_sqrt_of_size():
    assert frobenius_norm([[1, 0], [0, 1]]) == APPROX(math.sqrt(2))


def test_frobenius_norm_of_zero_matrix_is_zero():
    assert frobenius_norm([[0, 0], [0, 0]]) == APPROX(0.0)


# ---------------------------------------------------------- power_iteration
def test_power_iteration_finds_the_stretched_axis():
    assert power_iteration([[3, 0], [0, 1]]) == APPROX([1.0, 0.0])


def test_power_iteration_returns_a_unit_vector():
    v = power_iteration([[1, 2], [3, 4]])
    assert math.sqrt(sum(x * x for x in v)) == APPROX(1.0)


def test_power_iteration_ignores_a_crushed_direction():
    """Матрица [[0,2],[0,0]] убивает первую координату — вектор смотрит вдоль второй."""
    assert power_iteration([[0, 2], [0, 0]]) == APPROX([0.0, 1.0])


def test_power_iteration_on_zero_matrix_returns_zeros():
    assert power_iteration([[0, 0], [0, 0]]) == APPROX([0.0, 0.0])


# ------------------------------------------------------ top_singular_triple
def test_top_singular_value_of_a_diagonal_matrix_is_its_largest_entry():
    sigma, _, _ = top_singular_triple([[3, 0], [0, 1]])
    assert sigma == APPROX(3.0)


def test_top_triple_satisfies_the_defining_relation():
    """Смысл тройки: A v = sigma * u. Это и есть определение."""
    A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    sigma, u, v = top_singular_triple(A)
    assert matvec(A, v) == APPROX([sigma * x for x in u])


def test_top_triple_vectors_are_unit_length():
    _, u, v = top_singular_triple([[1.0, 2.0], [3.0, 4.0]])
    assert math.sqrt(sum(x * x for x in u)) == APPROX(1.0)
    assert math.sqrt(sum(x * x for x in v)) == APPROX(1.0)


def test_top_singular_value_of_a_zero_matrix_is_zero():
    sigma, u, _ = top_singular_triple([[0, 0], [0, 0]])
    assert sigma == APPROX(0.0)
    assert u == APPROX([0.0, 0.0])


# --------------------------------------------------------------------- svd
def test_svd_of_a_diagonal_matrix_recovers_its_entries():
    assert [s for s, _, _ in svd([[4, 0, 0], [0, 3, 0], [0, 0, 2]])] == APPROX(
        [4.0, 3.0, 2.0]
    )


def test_svd_values_come_out_sorted_descending():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    sigmas = [s for s, _, _ in svd(A)]
    assert sigmas == sorted(sigmas, reverse=True)


def test_svd_of_a_rank_one_matrix_returns_exactly_one_triple():
    """Ранг 1 — значит ненулевое сингулярное число ровно одно."""
    assert len(svd(outer([1, 2, 3], [1, 1]))) == 1


def test_svd_right_vectors_are_orthogonal_to_each_other():
    A = [[1.0, 2.0], [3.0, 4.0]]
    (_, _, v1), (_, _, v2) = svd(A)
    assert dot(v1, v2) == APPROX(0.0)


def test_svd_k_limits_the_number_of_triples():
    assert len(svd([[4, 0, 0], [0, 3, 0], [0, 0, 2]], 2)) == 2


def test_svd_does_not_mutate_the_input_matrix():
    """Ловушка: дефляция вычитает слои — если делать это на месте, A пропадёт."""
    A = [[1.0, 2.0], [3.0, 4.0]]
    svd(A)
    assert flat(A) == [1.0, 2.0, 3.0, 4.0]


def test_svd_keeps_a_tiny_but_well_conditioned_matrix():
    """Ранг зависит от относительного масштаба, а не от абсолютного cutoff."""
    triples = svd([[1e-12]])
    assert len(triples) == 1
    assert triples[0][0] == pytest.approx(1e-12, rel=1e-9, abs=0.0)


# --------------------------------------------------------------- reconstruct
def test_full_reconstruction_returns_the_original_matrix():
    A = [[1.0, 2.0], [3.0, 4.0]]
    assert flat(reconstruct(svd(A))) == APPROX(flat(A))


def test_truncated_reconstruction_keeps_the_shape():
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert flat(reconstruct(svd(A), 1)) == pytest.approx(flat(A), abs=10.0)
    assert len(reconstruct(svd(A), 1)[0]) == 3


def test_rank_one_error_equals_the_discarded_singular_value():
    """Эккарт — Янг: ошибка усечения по Фробениусу — корень из суммы
    квадратов выброшенных sigma, не больше и не меньше."""
    A = [[1.0, 2.0], [3.0, 4.0]]
    triples = svd(A)
    approx = reconstruct(triples, 1)
    diff = [[A[i][j] - approx[i][j] for j in range(2)] for i in range(2)]
    assert frobenius_norm(diff) == APPROX(triples[1][0])


def test_more_terms_never_make_the_approximation_worse():
    A = [[1.0, 2.0, 0.0], [3.0, 4.0, 1.0], [0.0, 1.0, 5.0]]
    triples = svd(A)
    errors = []
    for k in range(1, 4):
        approx = reconstruct(triples, k)
        diff = [[A[i][j] - approx[i][j] for j in range(3)] for i in range(3)]
        errors.append(frobenius_norm(diff))
    assert errors == sorted(errors, reverse=True)


def test_reconstruct_with_zero_terms_is_the_zero_matrix():
    assert flat(reconstruct(svd([[1.0, 2.0], [3.0, 4.0]]), 0)) == APPROX([0, 0, 0, 0])


# ---------------------------------------------------------- condition_number
def test_condition_number_of_identity_is_one():
    assert condition_number([[1, 0], [0, 1]]) == APPROX(1.0)


def test_condition_number_of_a_diagonal_matrix_is_the_ratio():
    assert condition_number([[2, 0], [0, 1]]) == APPROX(2.0)


def test_condition_number_of_a_singular_matrix_is_infinite():
    """Ловушка: svd вернёт всего одну тройку — считать по ней нельзя."""
    assert condition_number([[1, 1], [1, 1]]) == float("inf")


# ------------------------------------------------------------- pseudoinverse
def test_pseudoinverse_of_a_diagonal_matrix_inverts_its_entries():
    assert flat(pseudoinverse([[2, 0], [0, 4]])) == APPROX([0.5, 0.0, 0.0, 0.25])


def test_pseudoinverse_of_an_invertible_matrix_is_the_inverse():
    A = [[1.0, 2.0], [3.0, 4.0]]
    P = pseudoinverse(A)
    product = [[sum(A[i][t] * P[t][j] for t in range(2)) for j in range(2)] for i in range(2)]
    assert flat(product) == APPROX([1.0, 0.0, 0.0, 1.0])


def test_pseudoinverse_shape_is_transposed():
    """У A три строки и два столбца — у A+ два и три."""
    P = pseudoinverse([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    assert len(P) == 2 and len(P[0]) == 3


def test_pseudoinverse_solves_an_overdetermined_system_by_least_squares():
    """Три уравнения, два неизвестных, точного решения нет — есть наилучшее."""
    A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    b = [3.0, 5.0, 6.0]
    x = matvec(pseudoinverse(A), b)
    assert x == pytest.approx([1.5, 5.0 / 3.0], abs=1e-5)


def test_least_squares_solution_beats_a_nearby_guess():
    """Смысл наименьших квадратов: любой сдвиг увеличивает невязку."""
    A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    b = [3.0, 5.0, 6.0]
    x = matvec(pseudoinverse(A), b)
    residual = lambda p: sum((r - t) ** 2 for r, t in zip(matvec(A, p), b))
    assert residual(x) < residual([x[0] + 0.1, x[1]])
    assert residual(x) < residual([x[0], x[1] - 0.1])
