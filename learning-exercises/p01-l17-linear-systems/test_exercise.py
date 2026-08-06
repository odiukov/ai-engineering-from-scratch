"""Тесты к уроку «Линейные системы». Правь exercise.py."""

import pytest

from exercise import (
    back_substitution,
    cholesky,
    forward_substitution,
    gaussian_elimination,
    least_squares,
    normal_equations,
    ridge_regression,
    solve_cholesky,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в плоский."""
    return [v for row in M for v in row]


def matvec(A, x):
    """A @ x — нужна тестам, чтобы подставить решение обратно в систему."""
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


# ------------------------------------------------------- back_substitution
def test_back_substitution_solves_a_two_by_two():
    assert back_substitution([[2, 1], [0, 3]], [5, 9]) == APPROX([1.0, 3.0])


def test_back_substitution_on_identity_returns_the_right_hand_side():
    assert back_substitution([[1, 0], [0, 1]], [4, 7]) == APPROX([4.0, 7.0])


def test_back_substitution_ignores_the_lower_triangle():
    """Ниже диагонали по договорённости нули: мусор там не должен влиять."""
    clean = back_substitution([[2, 1], [0, 3]], [5, 9])
    dirty = back_substitution([[2, 1], [99, 3]], [5, 9])
    assert dirty == APPROX(clean)


def test_back_substitution_rejects_zero_on_the_diagonal():
    """Ловушка: ноль на диагонали — это деление на ноль, а не «почти решение»."""
    with pytest.raises(ValueError):
        back_substitution([[1, 2], [0, 0]], [3, 4])


# ---------------------------------------------------- forward_substitution
def test_forward_substitution_solves_a_two_by_two():
    assert forward_substitution([[2, 0], [1, 3]], [4, 11]) == APPROX([2.0, 3.0])


def test_forward_substitution_on_identity_returns_the_right_hand_side():
    assert forward_substitution([[1, 0], [0, 1]], [4, 7]) == APPROX([4.0, 7.0])


def test_forward_substitution_ignores_the_upper_triangle():
    """Выше диагонали нули: если функция туда смотрит, ответ поедет."""
    clean = forward_substitution([[2, 0], [1, 3]], [4, 11])
    dirty = forward_substitution([[2, 99], [1, 3]], [4, 11])
    assert dirty == APPROX(clean)


def test_forward_substitution_rejects_zero_on_the_diagonal():
    with pytest.raises(ValueError):
        forward_substitution([[0, 0], [2, 1]], [3, 4])


# ---------------------------------------------------- gaussian_elimination
def test_gaussian_elimination_solves_the_lesson_example():
    A = [[2, 1, 1], [4, 3, 3], [2, 3, 1]]
    assert gaussian_elimination(A, [8, 20, 12]) == APPROX([2.0, 2.0, 2.0])


def test_gaussian_elimination_survives_zero_first_pivot():
    """Ловушка: A[0][0] == 0. Без выбора pivot тут деление на ноль."""
    assert gaussian_elimination([[0, 1], [1, 0]], [2, 3]) == APPROX([3.0, 2.0])


def test_gaussian_elimination_stays_accurate_on_a_tiny_pivot():
    """Ловушка: pivot 1e-20 не ноль, деления не будет — будет множитель 1e20.

    Точное решение тут [1.0, 1.0]. Без перестановки строк первая координата
    выходит нулём: множитель съедает все значащие цифры.
    """
    A = [[1e-20, 1.0], [1.0, 1.0]]
    assert gaussian_elimination(A, [1.0, 2.0]) == pytest.approx([1.0, 1.0], abs=1e-9)


def test_gaussian_elimination_does_not_mutate_its_inputs():
    """Ловушка: исключать надо в копии, чужую матрицу портить нельзя."""
    A = [[2, 1, 1], [4, 3, 3], [2, 3, 1]]
    b = [8, 20, 12]
    gaussian_elimination(A, b)
    assert A == [[2, 1, 1], [4, 3, 3], [2, 3, 1]]
    assert b == [8, 20, 12]


def test_gaussian_elimination_rejects_a_singular_matrix():
    """Вторая строка — удвоенная первая: единственного решения нет."""
    with pytest.raises(ValueError):
        gaussian_elimination([[1, 2], [2, 4]], [3, 6])


# ---------------------------------------------------------------- cholesky
def test_cholesky_matches_the_lesson_example():
    assert flat(cholesky([[4, 2], [2, 5]])) == APPROX([2.0, 0.0, 1.0, 2.0])


def test_cholesky_product_reconstructs_the_matrix():
    """Смысл разложения: L L^T обязан вернуть исходную A."""
    A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
    L = cholesky(A)
    n = len(A)
    LLt = [[sum(L[i][k] * L[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
    assert flat(LLt) == pytest.approx(flat(A), abs=1e-9)


def test_cholesky_is_lower_triangular():
    """Над диагональю должны стоять явные нули, а не что попало."""
    L = cholesky([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
    assert flat([[L[i][j] for j in range(i + 1, 3)] for i in range(3)]) == APPROX([0.0, 0.0, 0.0])


def test_cholesky_rejects_a_non_symmetric_matrix():
    """Ловушка: на несимметричной формулы дадут L, у которого L L^T != A."""
    with pytest.raises(ValueError):
        cholesky([[4, 2], [3, 5]])


def test_cholesky_rejects_a_matrix_that_is_not_positive_definite():
    """Под корнем окажется неположительное число — это ValueError, а не NaN."""
    with pytest.raises(ValueError):
        cholesky([[1, 2], [2, 1]])


# ---------------------------------------------------------- solve_cholesky
def test_solve_cholesky_on_a_diagonal_matrix():
    assert solve_cholesky([[4, 0], [0, 9]], [8, 18]) == APPROX([2.0, 2.0])


def test_solve_cholesky_matches_gaussian_elimination():
    """Один и тот же ответ двумя дорогами — значит, транспонирование на месте."""
    A = [[4, 2], [2, 5]]
    assert solve_cholesky(A, [10, 14]) == pytest.approx(gaussian_elimination(A, [10, 14]), abs=1e-9)


def test_solve_cholesky_solution_reproduces_b():
    """Подставляем решение обратно: A x обязан совпасть с b."""
    A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
    b = [1.0, -2.0, 3.0]
    assert matvec(A, solve_cholesky(A, b)) == pytest.approx(b, abs=1e-9)


def test_solve_cholesky_rejects_a_non_positive_definite_matrix():
    with pytest.raises(ValueError):
        solve_cholesky([[1, 2], [2, 1]], [1, 1])


# -------------------------------------------------------- normal_equations
def test_normal_equations_match_the_lesson_example():
    AtA, Atb = normal_equations([[1, 1], [1, 2], [1, 3], [1, 4]], [3, 5, 6, 8])
    assert flat(AtA) == APPROX([4.0, 10.0, 10.0, 30.0])
    assert Atb == APPROX([22.0, 63.0])


def test_normal_equations_produce_a_symmetric_matrix():
    """A^T A симметрична для любой A — это свойство, а не совпадение."""
    AtA, _ = normal_equations([[1, 2, 3], [4, 5, 6], [7, 8, 10], [0, 1, 0]], [1, 2, 3, 4])
    transposed = [[AtA[j][i] for j in range(3)] for i in range(3)]
    assert flat(transposed) == pytest.approx(flat(AtA), abs=1e-9)


def test_normal_equations_shrink_a_tall_matrix_to_n_by_n():
    """Ловушка индексов: суммируем по строкам, размер берём по столбцам."""
    AtA, Atb = normal_equations([[1, 0], [0, 1], [1, 1], [2, 0]], [1, 1, 1, 1])
    assert len(AtA) == 2 and all(len(row) == 2 for row in AtA)
    assert len(Atb) == 2
    assert flat(AtA) == APPROX([6.0, 1.0, 1.0, 2.0])


def test_normal_equations_do_not_mutate_the_input():
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    b = [3, 5, 6, 8]
    normal_equations(A, b)
    assert A == [[1, 1], [1, 2], [1, 3], [1, 4]]
    assert b == [3, 5, 6, 8]


# ------------------------------------------------------------ least_squares
def test_least_squares_fits_the_lesson_line():
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    assert least_squares(A, [3, 5, 6, 8]) == pytest.approx([1.5, 1.6], abs=1e-9)


def test_least_squares_is_exact_when_the_system_is_consistent():
    """Если b лежит в пространстве столбцов, невязка ровно нулевая."""
    A = [[1, 0], [0, 1], [1, 1]]
    x_true = [2.0, -3.0]
    b = matvec(A, x_true)
    assert least_squares(A, b) == pytest.approx(x_true, abs=1e-9)


def test_least_squares_residual_is_orthogonal_to_the_columns():
    """Определение минимума: A^T (A x - b) равно нулю."""
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    b = [3, 5, 6, 8]
    x = least_squares(A, b)
    residual = [ax - bi for ax, bi in zip(matvec(A, x), b)]
    projections = [sum(A[k][i] * residual[k] for k in range(len(A))) for i in range(2)]
    assert projections == pytest.approx([0.0, 0.0], abs=1e-9)


def test_least_squares_fails_on_collinear_columns():
    """Столбец — копия другого: A^T A вырождена, решения нет."""
    with pytest.raises(ValueError):
        least_squares([[1, 1], [2, 2], [3, 3]], [1, 2, 3])


# --------------------------------------------------------- ridge_regression
def test_ridge_with_zero_lambda_matches_least_squares():
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    b = [3, 5, 6, 8]
    assert ridge_regression(A, b, 0.0) == pytest.approx(least_squares(A, b), abs=1e-9)


def test_ridge_solves_where_least_squares_is_singular():
    """Ради этого ridge и существует: lam*I вытаскивает вырожденную задачу."""
    assert ridge_regression([[1, 1], [1, 1]], [2, 2], 1.0) == APPROX([0.8, 0.8])


def test_ridge_shrinks_the_weights_as_lambda_grows():
    """Чем больше lam, тем ближе веса к нулю."""
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    b = [3, 5, 6, 8]
    norms = [sum(v * v for v in ridge_regression(A, b, lam)) for lam in (0.0, 1.0, 10.0)]
    assert norms[0] > norms[1] > norms[2]


def test_ridge_does_not_mutate_the_input():
    """Ловушка: lam прибавляется к копии A^T A, а не к матрице вызывающего."""
    A = [[1, 1], [1, 2], [1, 3], [1, 4]]
    b = [3, 5, 6, 8]
    ridge_regression(A, b, 5.0)
    assert A == [[1, 1], [1, 2], [1, 3], [1, 4]]
    assert b == [3, 5, 6, 8]
