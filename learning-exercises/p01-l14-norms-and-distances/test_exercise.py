"""Тесты к уроку «Нормы и расстояния». Правь exercise.py."""

import math

import pytest

from exercise import (
    cosine_similarity,
    distance,
    dot,
    edit_distance,
    jaccard_similarity,
    lp_norm,
    mahalanobis,
    nearest_neighbor,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
INF = float("inf")


# --------------------------------------------------------------------- dot
def test_dot_of_two_vectors():
    assert dot([1, 2, 3], [4, 5, 6]) == APPROX(32)


def test_dot_of_perpendicular_vectors_is_zero():
    assert dot([1, 0], [0, 1]) == APPROX(0)


def test_dot_is_commutative():
    assert dot([1, 2], [3, 4]) == APPROX(dot([3, 4], [1, 2]))


def test_dot_grows_with_magnitude_while_cosine_does_not():
    """В этом вся разница dot и косинуса: dot видит длину, косинус нет."""
    assert dot([3, 0], [1, 0]) > dot([1, 0], [1, 0])
    assert cosine_similarity([3, 0], [1, 0]) == APPROX(cosine_similarity([1, 0], [1, 0]))


# ----------------------------------------------------------------- lp_norm
def test_l2_norm_is_the_default():
    assert lp_norm([3, 4]) == APPROX(5.0)


def test_l1_norm_sums_absolute_values():
    assert lp_norm([1, -2, 3], 1) == APPROX(6.0)


def test_l_infinity_norm_is_the_largest_absolute_component():
    assert lp_norm([1, -2, 3], INF) == APPROX(3.0)


def test_lp_norm_between_one_and_infinity():
    assert lp_norm([1, 1], 3) == APPROX(2 ** (1 / 3))


def test_norm_of_the_zero_vector_is_zero_for_every_p():
    for p in (1, 2, 3, INF):
        assert lp_norm([0, 0, 0], p) == APPROX(0.0)


def test_norm_is_absolutely_homogeneous():
    """||c*v|| = |c| * ||v|| — одно из трёх определяющих свойств нормы."""
    v = [1.0, -2.0, 3.0]
    for p in (1, 2, INF):
        assert lp_norm([-3 * x for x in v], p) == APPROX(3 * lp_norm(v, p))


def test_norm_ignores_the_sign_of_components():
    """Без модуля нечётные p дали бы отрицательный вклад."""
    assert lp_norm([1, -2, 3], 3) == APPROX(lp_norm([1, 2, 3], 3))


# ---------------------------------------------------------------- distance
def test_l2_distance_between_two_points():
    assert distance([1, 2, 3], [4, 0, 6]) == APPROX(math.sqrt(22))


def test_l1_distance_walks_along_the_axes():
    assert distance([1, 1], [4, 5], 1) == APPROX(7.0)


def test_l_infinity_distance_looks_at_the_worst_axis_only():
    assert distance([1, 1], [4, 5], INF) == APPROX(4.0)


def test_distance_to_itself_is_zero():
    assert distance([1.5, -2.0], [1.5, -2.0]) == APPROX(0.0)


def test_distance_is_symmetric():
    assert distance([1, 2, 3], [4, 0, 6]) == APPROX(distance([4, 0, 6], [1, 2, 3]))


def test_l_infinity_never_exceeds_l2_which_never_exceeds_l1():
    """Порядок гарантирован для любой пары точек: чем больше p, тем сильнее
    доминирует самая крупная координата."""
    a, b = [1, 2, 3, -4], [4, 0, 6, 1]
    assert distance(a, b, INF) <= distance(a, b, 2) <= distance(a, b, 1)


# ------------------------------------------------------- cosine_similarity
def test_cosine_of_a_forty_five_degree_angle():
    assert cosine_similarity([1, 0], [1, 1]) == APPROX(1 / math.sqrt(2))


def test_cosine_of_identical_directions_is_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == APPROX(1.0)


def test_cosine_of_opposite_directions_is_minus_one():
    assert cosine_similarity([1, 0], [-1, 0]) == APPROX(-1.0)


def test_cosine_of_perpendicular_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_does_not_change_when_a_vector_is_stretched():
    """Документ вдвое длиннее — тот же смысл. Ради этого косинус и берут."""
    a, b = [1.0, 2.0, 3.0], [4.0, 1.0, 0.5]
    stretched = [100 * x for x in b]
    assert cosine_similarity(a, stretched) == APPROX(cosine_similarity(a, b))


def test_cosine_with_a_zero_vector_does_not_divide_by_zero():
    assert cosine_similarity([0, 0], [1, 1]) == APPROX(0.0)


def test_high_cosine_can_coexist_with_a_large_l2_distance():
    """Два вектора одного направления, но разной длины: угол нулевой,
    а евклидово расстояние огромное."""
    a, b = [1.0, 1.0], [30.0, 30.0]
    assert cosine_similarity(a, b) > 0.99
    assert distance(a, b) > 10


# ------------------------------------------------------------- mahalanobis
def test_mahalanobis_with_identity_reduces_to_euclidean():
    """Единичная обратная ковариация означает «признаки независимы и
    единичной дисперсии» — это ровно L2."""
    x, y = [0.0, 0.0], [3.0, 4.0]
    assert mahalanobis(x, y, [[1, 0], [0, 1]]) == APPROX(distance(x, y))


def test_mahalanobis_shrinks_an_axis_with_large_variance():
    """Обратная дисперсия 0.25 по первой оси = разброс там вчетверо шире,
    значит те же 4 единицы стоят вдвое дешевле."""
    assert mahalanobis([0, 0], [4, 0], [[0.25, 0], [0, 1]]) == APPROX(2.0)


def test_mahalanobis_to_itself_is_zero():
    assert mahalanobis([1.0, 2.0], [1.0, 2.0], [[2, 0.5], [0.5, 3]]) == APPROX(0.0)


def test_mahalanobis_is_symmetric():
    S = [[2.0, 0.5], [0.5, 3.0]]
    assert mahalanobis([1.0, 2.0], [4.0, 0.0], S) == APPROX(
        mahalanobis([4.0, 0.0], [1.0, 2.0], S)
    )


def test_correlation_makes_two_equally_distant_points_unequal():
    """Обе точки на расстоянии 2 по L2, но корреляция признаков делает
    одну из них обычной, а другую выбросом."""
    S = [[2.0, -1.8], [-1.8, 2.0]]  # обратная ковариация, признаки связаны
    along = mahalanobis([0.0, 0.0], [math.sqrt(2), math.sqrt(2)], S)
    across = mahalanobis([0.0, 0.0], [math.sqrt(2), -math.sqrt(2)], S)
    assert across > along


# -------------------------------------------------------- jaccard_similarity
def test_jaccard_counts_intersection_over_union():
    assert jaccard_similarity({1, 2, 3}, {1, 3, 4, 5}) == APPROX(0.4)


def test_jaccard_of_identical_sets_is_one():
    assert jaccard_similarity({"cat", "dog"}, {"dog", "cat"}) == APPROX(1.0)


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard_similarity({1, 2}, {3, 4}) == APPROX(0.0)


def test_jaccard_of_two_empty_sets_does_not_divide_by_zero():
    assert jaccard_similarity(set(), set()) == APPROX(1.0)


def test_jaccard_ignores_repetitions_because_it_works_on_sets():
    """Жаккар считает наличие, а не частоту — это его граница применимости."""
    assert jaccard_similarity([1, 1, 1, 2], [1, 2]) == APPROX(1.0)


# ----------------------------------------------------------- edit_distance
def test_kitten_to_sitting_takes_three_edits():
    assert edit_distance("kitten", "sitting") == 3


def test_edit_distance_of_equal_strings_is_zero():
    assert edit_distance("abc", "abc") == 0


def test_edit_distance_from_an_empty_string_is_its_length():
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3


def test_a_single_substitution_costs_one():
    assert edit_distance("cat", "bat") == 1


def test_a_single_insertion_costs_one():
    assert edit_distance("cat", "cart") == 1


def test_edit_distance_is_symmetric():
    assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")


def test_edit_distance_never_exceeds_the_longer_length():
    assert edit_distance("abcd", "wxyz") == 4


# --------------------------------------------------------- nearest_neighbor
def test_nearest_neighbor_under_l2():
    assert nearest_neighbor([1, 0], [[10, 0], [1, 1]], distance) == 1


def test_the_metric_decides_the_answer_not_the_data():
    """Тот же запрос, те же точки, разные метрики — разные соседи.
    Это и есть главный вывод урока."""
    query, points = [1, 0], [[10, 0], [1, 1]]
    cosine = lambda a, b: 1 - cosine_similarity(a, b)
    assert nearest_neighbor(query, points, distance) == 1
    assert nearest_neighbor(query, points, cosine) == 0


def test_l1_and_l_infinity_can_disagree_too():
    query = [0, 0]
    points = [[3, 3], [5, 0]]
    assert nearest_neighbor(query, points, lambda a, b: distance(a, b, 1)) == 1
    assert nearest_neighbor(query, points, lambda a, b: distance(a, b, INF)) == 0


def test_the_query_itself_wins_when_it_is_in_the_dataset():
    assert nearest_neighbor([2, 2], [[0, 0], [2, 2], [9, 9]], distance) == 1


def test_ties_go_to_the_smaller_index():
    assert nearest_neighbor([0, 0], [[1, 0], [0, 1], [-1, 0]], distance) == 0
