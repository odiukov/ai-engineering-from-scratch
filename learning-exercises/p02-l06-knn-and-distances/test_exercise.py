"""Тесты к уроку «Метод k ближайших соседей и расстояния». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    build_kdtree,
    cosine_distance,
    k_nearest,
    kdtree_nearest,
    knn_classify,
    knn_regress,
    l1_distance,
    l2_distance,
    minkowski_distance,
    standardize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


# ------------------------------------------------------------- l2_distance
def test_l2_on_the_three_four_five_triangle():
    assert l2_distance([0.0, 0.0], [3.0, 4.0]) == APPROX(5.0)


def test_l2_from_a_point_to_itself_is_zero():
    assert l2_distance([1.0, -7.0, 3.0], [1.0, -7.0, 3.0]) == APPROX(0.0)


def test_l2_is_symmetric():
    a, b = [1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]
    assert l2_distance(a, b) == APPROX(l2_distance(b, a))


def test_l2_distances_converge_in_high_dimensions():
    """Проклятие размерности: в 200 измерениях все точки почти равноудалены.

    Отношение максимального расстояния к минимальному в 2D разлетается, а
    в 200D жмётся к единице — слово "ближайший" теряет смысл.
    """
    rng = random.Random(0)
    low_q = [rng.random() for _ in range(2)]
    low = [[rng.random() for _ in range(2)] for _ in range(200)]
    low_d = [l2_distance(p, low_q) for p in low]

    high_q = [rng.random() for _ in range(200)]
    high = [[rng.random() for _ in range(200)] for _ in range(200)]
    high_d = [l2_distance(p, high_q) for p in high]

    assert max(low_d) / min(low_d) > 5
    assert max(high_d) / min(high_d) < 1.5


# ------------------------------------------------------------- l1_distance
def test_l1_sums_absolute_differences():
    assert l1_distance([0.0, 0.0], [3.0, 4.0]) == APPROX(7.0)


def test_l1_handles_negative_coordinates():
    assert l1_distance([1.0], [-2.0]) == APPROX(3.0)


def test_l1_and_l2_agree_in_one_dimension():
    assert l1_distance([5.0], [1.0]) == APPROX(l2_distance([5.0], [1.0]))


def test_l1_is_gentler_to_a_single_outlier_than_l2():
    """L2 возводит выброс в квадрат, L1 — нет. Отсюда и устойчивость."""
    near = [1.0, 1.0, 1.0]
    outlier = [0.0, 0.0, 10.0]
    assert l1_distance([0.0] * 3, outlier) / l1_distance([0.0] * 3, near) == APPROX(
        10 / 3
    )
    assert l2_distance([0.0] * 3, outlier) / l2_distance([0.0] * 3, near) > 5


# --------------------------------------------------------- cosine_distance
def test_cosine_ignores_the_length_of_the_vectors():
    """Главное свойство: важен только угол. Растянули вектор — ничего не изменилось."""
    assert cosine_distance([1.0, 0.0], [5.0, 0.0]) == APPROX(0.0)
    assert cosine_distance([1.0, 2.0], [100.0, 200.0]) == APPROX(0.0)


def test_cosine_of_perpendicular_vectors_is_one():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == APPROX(1.0)


def test_cosine_of_opposite_vectors_is_two():
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == APPROX(2.0)


def test_cosine_with_a_zero_vector_is_one_not_a_crash():
    """Ловушка: нулевая норма в знаменателе."""
    assert cosine_distance([0.0, 0.0], [1.0, 2.0]) == APPROX(1.0)


def test_cosine_disagrees_with_l2_about_who_is_closer():
    """Один и тот же запрос, разные метрики — разные соседи. Метрика решает."""
    query = [1.0, 1.0]
    long_same_direction = [10.0, 10.0]
    short_other_direction = [1.4, 0.0]
    assert cosine_distance(query, long_same_direction) < cosine_distance(
        query, short_other_direction
    )
    assert l2_distance(query, long_same_direction) > l2_distance(
        query, short_other_direction
    )


# ------------------------------------------------------ minkowski_distance
def test_minkowski_with_p_one_is_l1():
    a, b = [0.0, 0.0], [3.0, 4.0]
    assert minkowski_distance(a, b, p=1) == APPROX(l1_distance(a, b))


def test_minkowski_with_p_two_is_l2():
    a, b = [0.0, 0.0], [3.0, 4.0]
    assert minkowski_distance(a, b, p=2) == APPROX(l2_distance(a, b))


def test_minkowski_with_infinite_p_is_the_largest_coordinate_gap():
    """Ловушка: в бесконечную степень не возводят — это отдельная ветка."""
    assert minkowski_distance([0.0, 0.0], [3.0, 4.0], p=float("inf")) == APPROX(4.0)


def test_minkowski_creeps_toward_chebyshev_as_p_grows():
    a, b = [0.0, 0.0], [3.0, 4.0]
    assert minkowski_distance(a, b, p=50) == pytest.approx(4.0, abs=1e-5)


def test_minkowski_shrinks_as_p_grows():
    a, b = [0.0, 0.0], [3.0, 4.0]
    values = [minkowski_distance(a, b, p=p) for p in (1, 2, 4, 10)]
    assert all(later < earlier for earlier, later in zip(values, values[1:]))


# ----------------------------------------------------------------- k_nearest
def test_k_nearest_returns_distance_index_pairs_in_order():
    assert k_nearest([[0.0], [10.0], [1.0]], [0.0], 2) == [
        (APPROX(0.0), 0),
        (APPROX(1.0), 2),
    ]


def test_k_nearest_returns_exactly_k_neighbours():
    X = [[float(i)] for i in range(10)]
    assert len(k_nearest(X, [0.0], 4)) == 4


def test_k_nearest_returns_everything_when_k_is_too_large():
    X = [[float(i)] for i in range(3)]
    assert len(k_nearest(X, [0.0], 99)) == 3


def test_k_nearest_is_sorted_by_distance():
    X = [[float(i)] for i in (7, 2, 9, 1, 5)]
    distances = [d for d, _ in k_nearest(X, [0.0], 5)]
    assert distances == sorted(distances)


def test_k_nearest_respects_the_distance_function():
    """С косинусом ближайший другой: длина перестаёт иметь значение."""
    X = [[10.0, 10.0], [1.4, 0.0]]
    assert k_nearest(X, [1.0, 1.0], 1, l2_distance)[0][1] == 1
    assert k_nearest(X, [1.0, 1.0], 1, cosine_distance)[0][1] == 0


# -------------------------------------------------------------- knn_classify
def test_knn_classify_takes_the_majority_of_the_neighbours():
    assert knn_classify([[0.0], [1.0], [10.0]], [0, 0, 1], [0.5], k=3) == 0


def test_knn_with_k_one_copies_the_nearest_label():
    X = [[0.0], [5.0], [6.0]]
    assert knn_classify(X, [0, 1, 1], [5.2], k=1) == 1


def test_a_large_k_smooths_the_boundary_toward_the_majority_class():
    """K = N превращает KNN в "всегда самый частый класс" — максимум смещения."""
    X = [[0.0], [5.0], [6.0], [7.0]]
    y = [0, 1, 1, 1]
    assert knn_classify(X, y, [0.1], k=1) == 0
    assert knn_classify(X, y, [0.1], k=4) == 1


def test_knn_classify_breaks_a_tie_toward_the_smaller_label():
    assert knn_classify([[0.0], [1.0]], [1, 0], [0.5], k=2) == 0


def test_weighting_lets_one_very_close_neighbour_outvote_two_distant_ones():
    X = [[0.0], [5.0], [6.0]]
    y = [0, 1, 1]
    assert knn_classify(X, y, [0.1], k=3, weighted=False) == 1
    assert knn_classify(X, y, [0.1], k=3, weighted=True) == 0


def test_weighted_knn_survives_a_query_sitting_on_a_training_point():
    """Ловушка: расстояние 0 без epsilon в знаменателе роняет функцию."""
    assert knn_classify([[1.0], [2.0]], [7, 9], [1.0], k=2, weighted=True) == 7


# --------------------------------------------------------------- knn_regress
def test_knn_regress_averages_the_neighbours():
    assert knn_regress([[0.0], [2.0]], [10.0, 20.0], [1.0], k=2) == APPROX(15.0)


def test_knn_regress_with_k_one_copies_the_nearest_value():
    assert knn_regress([[0.0], [5.0]], [1.0, 100.0], [4.9], k=1) == APPROX(100.0)


def test_knn_regress_cannot_extrapolate():
    """Обучали на значениях до 20 — 1000 модель не предскажет никогда."""
    X = [[0.0], [1.0], [2.0]]
    y = [0.0, 10.0, 20.0]
    for query in ([100.0], [-100.0], [0.5]):
        assert min(y) <= knn_regress(X, y, query, k=2) <= max(y)


def test_weighted_regression_leans_toward_the_closer_neighbour():
    X = [[0.0], [10.0]]
    y = [0.0, 100.0]
    plain = knn_regress(X, y, [1.0], k=2, weighted=False)
    weighted = knn_regress(X, y, [1.0], k=2, weighted=True)
    assert plain == APPROX(50.0)
    assert weighted < 20.0


def test_weighted_regression_survives_a_zero_distance():
    assert knn_regress([[1.0], [9.0]], [5.0, 500.0], [1.0], k=2, weighted=True) == (
        pytest.approx(5.0, abs=1e-6)
    )


# --------------------------------------------------------------- standardize
def test_standardize_centres_and_rescales_one_column():
    scaled, means, stds = standardize([[0.0], [2.0]])
    assert flat(scaled) == APPROX([-1.0, 1.0])
    assert means == APPROX([1.0])
    assert stds == APPROX([1.0])


def test_standardized_columns_have_zero_mean_and_unit_std():
    X = [[1.0, 100.0], [2.0, 300.0], [3.0, 200.0], [10.0, 700.0]]
    scaled, _, _ = standardize(X)
    for j in range(2):
        column = [row[j] for row in scaled]
        mean = sum(column) / len(column)
        std = math.sqrt(sum((v - mean) ** 2 for v in column) / len(column))
        assert mean == pytest.approx(0.0, abs=1e-9)
        assert std == pytest.approx(1.0, abs=1e-9)


def test_standardize_turns_a_constant_column_into_zeros():
    """Ловушка: std = 0 — ни NaN, ни исключения быть не должно."""
    scaled, _, stds = standardize([[7.0], [7.0], [7.0]])
    assert flat(scaled) == APPROX([0.0, 0.0, 0.0])
    assert stds == APPROX([0.0])


def test_scaling_stops_the_big_column_from_deciding_everything():
    """Без стандартизации второй признак съедает первый целиком.

    Точки [0, 0] и [1, 0] отличаются по первому признаку максимально, но
    в сыром виде их расстояния до запроса совпадают до пятого знака.
    После стандартизации разница становится больше единицы.
    """
    X = [[0.0, 0.0], [1.0, 500.0], [1.0, 0.0], [0.0, 500.0]]
    query = [0.0, 250.0]

    raw = [l2_distance(x, query) for x in X]
    assert raw[0] == pytest.approx(raw[2], rel=1e-4)

    scaled, means, stds = standardize(X)
    scaled_query = [(query[j] - means[j]) / stds[j] for j in range(2)]
    fixed = [l2_distance(x, scaled_query) for x in scaled]
    assert abs(fixed[0] - fixed[2]) > 1.0


# ------------------------------------------------------------- KD-дерево
def _grid(n=8):
    """Регулярная сетка n x n — 64 точки при n=8."""
    return [[float(i), float(j)] for i in range(n) for j in range(n)]


def test_kdtree_root_splits_on_the_first_axis():
    tree = build_kdtree([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert tree["axis"] == 0


def test_kdtree_alternates_axes_by_depth():
    tree = build_kdtree([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert tree["left"]["axis"] == 1


def test_kdtree_of_a_single_point_has_no_children():
    tree = build_kdtree([[1.0, 1.0]])
    assert tree["left"] is None and tree["right"] is None


def test_kdtree_of_nothing_is_none():
    assert build_kdtree([]) is None


def test_kdtree_keeps_every_point():
    points = _grid(4)
    seen = []

    def walk(node):
        if node is None:
            return
        seen.append(node["index"])
        walk(node["left"])
        walk(node["right"])

    walk(build_kdtree(points))
    assert sorted(seen) == list(range(len(points)))


def test_kdtree_indices_point_back_at_the_original_list():
    """Индексы должны пережить рекурсивную сортировку, иначе метку не достать."""
    points = _grid(4)
    tree = build_kdtree(points)

    def walk(node):
        if node is None:
            return
        assert points[node["index"]] == node["point"]
        walk(node["left"])
        walk(node["right"])

    walk(tree)


def test_kdtree_finds_the_same_point_as_brute_force():
    points = _grid()
    tree = build_kdtree(points)
    for query in ([0.2, 0.3], [7.4, 7.1], [3.5, 4.5], [-5.0, 20.0]):
        _, dist, _ = kdtree_nearest(tree, query)
        brute = min(range(len(points)), key=lambda i: l2_distance(points[i], query))
        assert dist == pytest.approx(l2_distance(points[brute], query))


def test_kdtree_visits_fewer_nodes_than_a_linear_scan():
    """Ради этого дерево и строится: половины отсекаются целиком."""
    points = _grid()
    _, _, visited = kdtree_nearest(build_kdtree(points), [0.2, 0.3])
    assert visited < len(points)


def test_kdtree_returns_exact_hit_at_zero_distance():
    points = _grid(4)
    idx, dist, _ = kdtree_nearest(build_kdtree(points), points[5])
    assert dist == pytest.approx(0.0)
    assert points[idx] == points[5]


def test_kdtree_must_search_both_halves_near_the_boundary():
    """Точка ровно на плоскости раздела: отсечь вторую ветку нельзя."""
    points = [[0.0, 0.0], [10.0, 0.0]]
    _, dist, _ = kdtree_nearest(build_kdtree(points), [5.0, 0.0])
    assert dist == pytest.approx(5.0)


def test_kdtree_works_in_three_dimensions():
    points = [[float(i), float(j), float(k)]
              for i in range(3) for j in range(3) for k in range(3)]
    tree = build_kdtree(points)
    idx, _, _ = kdtree_nearest(tree, [0.1, 0.1, 0.1])
    assert points[idx] == [0.0, 0.0, 0.0]
