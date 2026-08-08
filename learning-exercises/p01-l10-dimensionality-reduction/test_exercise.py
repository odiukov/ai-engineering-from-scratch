"""Тесты к уроку «Снижение размерности». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    center,
    column_means,
    covariance_matrix,
    explained_variance_ratio,
    power_iteration,
    project,
    reconstruction_error,
    top_components,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в плоскую."""
    return [value for row in M for value in row]


def _elongated_cloud(n=300):
    """Облако, вытянутое вдоль направления (1, 1)."""
    rng = random.Random(7)
    points = []
    for _ in range(n):
        long_axis = rng.gauss(0, 5)
        short_axis = rng.gauss(0, 0.2)
        points.append(
            [long_axis / math.sqrt(2) - short_axis / math.sqrt(2),
             long_axis / math.sqrt(2) + short_axis / math.sqrt(2)]
        )
    return points


def _rank_two_cloud(n=200):
    """Три признака, но третий — линейная комбинация первых двух."""
    rng = random.Random(11)
    points = []
    for _ in range(n):
        a, b = rng.gauss(0, 3), rng.gauss(0, 1)
        points.append([a + 2 * b, b, a - b])
    return points


# ------------------------------------------------------------ column_means
def test_column_means_basic():
    assert column_means([[1, 10], [3, 20]]) == APPROX([2.0, 15.0])


def test_column_means_of_a_single_row_is_that_row():
    assert column_means([[4.0, -1.0, 7.0]]) == APPROX([4.0, -1.0, 7.0])


def test_column_means_average_down_the_columns_not_across_the_rows():
    """Ловушка: строка — это объект, столбец — признак. Путать нельзя."""
    assert column_means([[1, 2, 3], [3, 4, 5]]) == APPROX([2.0, 3.0, 4.0])


# ------------------------------------------------------------------ center
def test_center_basic():
    assert flat(center([[1, 10], [3, 20]])) == APPROX([-1.0, -5.0, 1.0, 5.0])


def test_center_makes_every_column_mean_zero():
    assert column_means(center(_rank_two_cloud())) == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-9
    )


def test_center_does_not_mutate_the_input():
    X = [[1.0, 10.0], [3.0, 20.0]]
    center(X)
    assert flat(X) == APPROX([1.0, 10.0, 3.0, 20.0])


def test_center_keeps_the_shape():
    X = [[1, 2, 3], [4, 5, 6]]
    C = center(X)
    assert len(C) == 2 and all(len(row) == 3 for row in C)


# ------------------------------------------------------- covariance_matrix
def test_covariance_matrix_of_perfectly_correlated_features():
    assert flat(covariance_matrix([[1, 1], [2, 2], [3, 3]])) == APPROX(
        [1.0, 1.0, 1.0, 1.0]
    )


def test_covariance_matrix_uses_n_minus_one():
    """Ловушка: делитель n дал бы 1.0, несмещённая оценка даёт 2.0."""
    assert covariance_matrix([[1.0], [3.0]])[0][0] == APPROX(2.0)


def test_covariance_matrix_is_symmetric():
    C = covariance_matrix(_rank_two_cloud())
    assert flat(C) == pytest.approx(flat([list(col) for col in zip(*C)]), abs=1e-9)


def test_covariance_matrix_of_anticorrelated_features_is_negative():
    assert covariance_matrix([[1, 3], [2, 2], [3, 1]])[0][1] < 0


def test_covariance_matrix_of_orthogonal_features_has_zero_off_diagonal():
    C = covariance_matrix([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    assert C[0][1] == APPROX(0.0)


def test_covariance_diagonal_holds_the_plain_variances():
    X = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    C = covariance_matrix(X)
    assert C[0][0] == APPROX(5 / 3)
    assert C[1][1] == APPROX(500 / 3)


# --------------------------------------------------------- power_iteration
def test_power_iteration_finds_the_largest_eigenvalue():
    value, _ = power_iteration([[2, 0], [0, 3]])
    assert value == pytest.approx(3.0, abs=1e-6)


def test_power_iteration_finds_it_whichever_axis_it_sits_on():
    value, vector = power_iteration([[5, 0], [0, 1]])
    assert value == pytest.approx(5.0, abs=1e-6)
    assert [abs(x) for x in vector] == pytest.approx([1.0, 0.0], abs=1e-6)


def test_power_iteration_returns_a_unit_vector():
    _, vector = power_iteration([[3, 1], [1, 3]])
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0, abs=1e-9)


def test_power_iteration_vector_actually_satisfies_the_eigen_equation():
    """Проверяем определение: A @ v должно равняться lambda * v."""
    A = [[3, 1], [1, 3]]
    value, v = power_iteration(A)
    Av = [sum(A[i][j] * v[j] for j in range(2)) for i in range(2)]
    assert Av == pytest.approx([value * x for x in v], abs=1e-6)


def test_power_iteration_on_a_zero_matrix_does_not_divide_by_zero():
    """Ловушка: нормировать нулевой вектор нельзя — так падает дефляция."""
    value, vector = power_iteration([[0.0, 0.0], [0.0, 0.0]])
    assert value == APPROX(0.0)
    assert vector == APPROX([0.0, 0.0])


def test_power_iteration_finds_the_eigenvalue_with_largest_magnitude():
    """Отрицательная доминанта больше по модулю, хотя меньше алгебраически."""
    value, vector = power_iteration([[-5.0, 0.0], [0.0, 2.0]])
    assert value == pytest.approx(-5.0, abs=1e-6)
    assert [abs(x) for x in vector] == pytest.approx([1.0, 0.0], abs=1e-6)


# ---------------------------------------------------------- top_components
def test_top_components_returns_eigenvalues_in_descending_order():
    values = [value for value, _ in top_components([[2, 0], [0, 3]], 2)]
    assert values == pytest.approx([3.0, 2.0], abs=1e-6)


def test_top_components_finds_the_rotated_pair():
    values = [value for value, _ in top_components([[3, 1], [1, 3]], 2)]
    assert values == pytest.approx([4.0, 2.0], abs=1e-6)


def test_top_components_are_orthogonal_to_each_other():
    pairs = top_components([[3, 1], [1, 3]], 2)
    v1, v2 = pairs[0][1], pairs[1][1]
    assert sum(a * b for a, b in zip(v1, v2)) == pytest.approx(0.0, abs=1e-6)


def test_top_components_does_not_mutate_the_matrix():
    """Дефляция обязана работать на копии."""
    A = [[3.0, 1.0], [1.0, 3.0]]
    top_components(A, 2)
    assert flat(A) == APPROX([3.0, 1.0, 1.0, 3.0])


def test_top_components_reports_zero_for_a_direction_without_variance():
    """Данные плоские: третьей компоненте забирать уже нечего."""
    values = [v for v, _ in top_components(covariance_matrix(_rank_two_cloud()), 3)]
    assert values[2] == pytest.approx(0.0, abs=1e-6)


def test_top_component_of_an_elongated_cloud_points_along_the_cloud():
    """Знак собственного вектора произволен, поэтому сравниваем модули."""
    _, vector = top_components(covariance_matrix(_elongated_cloud()), 1)[0]
    assert [abs(x) for x in vector] == pytest.approx(
        [1 / math.sqrt(2), 1 / math.sqrt(2)], abs=1e-2
    )


# -------------------------------------------------- explained_variance_ratio
def test_explained_variance_ratio_splits_a_diagonal_matrix():
    assert explained_variance_ratio([[4, 0], [0, 1]], 2) == pytest.approx(
        [0.8, 0.2], abs=1e-6
    )


def test_explained_variance_ratio_of_one_component_is_the_largest_share():
    assert explained_variance_ratio([[4, 0], [0, 1]], 1) == pytest.approx(
        [0.8], abs=1e-6
    )


def test_explained_variance_ratios_sum_to_one_over_all_components():
    """Полная дисперсия — это след матрицы, он же сумма всех собственных чисел."""
    ratios = explained_variance_ratio(covariance_matrix(_rank_two_cloud()), 3)
    assert sum(ratios) == pytest.approx(1.0, abs=1e-6)


def test_explained_variance_ratios_are_non_increasing():
    ratios = explained_variance_ratio(covariance_matrix(_rank_two_cloud()), 3)
    assert all(a >= b - 1e-9 for a, b in zip(ratios, ratios[1:]))


def test_explained_variance_ratio_flags_the_flat_direction():
    ratios = explained_variance_ratio(covariance_matrix(_rank_two_cloud()), 3)
    assert ratios[2] == pytest.approx(0.0, abs=1e-6)
    assert sum(ratios[:2]) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- project
def test_project_onto_the_first_axis_returns_the_centered_coordinate():
    assert flat(project([[1, 0], [-1, 0]], [[1.0, 0.0]])) == APPROX([1.0, -1.0])


def test_project_produces_one_number_per_component():
    scores = project(_rank_two_cloud(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert len(scores) == 200 and all(len(row) == 2 for row in scores)


def test_project_output_is_centered():
    """Проекции центрированных данных сами центрированы — среднее ноль."""
    X = _elongated_cloud()
    components = [v for _, v in top_components(covariance_matrix(X), 2)]
    assert column_means(project(X, components)) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_project_subtracts_the_mean_before_the_dot_product():
    """Ловушка: без центрирования координаты уедут на постоянный сдвиг."""
    assert flat(project([[10.0, 0.0], [12.0, 0.0]], [[1.0, 0.0]])) == APPROX(
        [-1.0, 1.0]
    )


def test_first_projection_carries_more_spread_than_the_second():
    X = _elongated_cloud()
    components = [v for _, v in top_components(covariance_matrix(X), 2)]
    scores = project(X, components)
    spread = covariance_matrix(scores)
    assert spread[0][0] > spread[1][1]


# ------------------------------------------------------ reconstruction_error
def test_reconstruction_error_is_zero_when_nothing_is_dropped():
    X = _elongated_cloud()
    assert reconstruction_error(X, 2) == pytest.approx(0.0, abs=1e-12)


def test_reconstruction_error_is_zero_for_data_that_lies_on_one_line():
    assert reconstruction_error([[1, 0], [-1, 0], [2, 0], [-2, 0]], 1) == pytest.approx(
        0.0, abs=1e-12
    )


def test_reconstruction_error_shrinks_as_components_are_added():
    X = _rank_two_cloud()
    assert reconstruction_error(X, 1) > reconstruction_error(X, 2)


def test_reconstruction_error_is_never_negative():
    assert reconstruction_error(_rank_two_cloud(), 1) >= 0


def test_dropping_a_direction_without_variance_costs_nothing():
    """Данные лежат в плоскости: третья компонента ничего не добавляет."""
    X = _rank_two_cloud()
    assert reconstruction_error(X, 2) == pytest.approx(
        reconstruction_error(X, 3), abs=1e-9
    )


def test_reconstruction_error_of_an_elongated_cloud_is_small_at_one_component():
    """Облако почти одномерное — на одну компоненту сжимается почти без потерь."""
    assert reconstruction_error(_elongated_cloud(), 1) < 0.1
