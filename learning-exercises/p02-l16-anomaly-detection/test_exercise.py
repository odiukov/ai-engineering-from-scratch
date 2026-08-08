"""Тесты к уроку «Поиск аномалий». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    build_isolation_tree,
    expected_path_length,
    iqr_bounds,
    iqr_flags,
    isolation_scores,
    path_length,
    percentile,
    zscore_flags,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# плотное облако вокруг нуля плюс одна точка далеко в стороне
CLUSTER = [[0.1 * ((i * 7) % 11 - 5), 0.1 * ((i * 5) % 13 - 6)] for i in range(40)]
WITH_OUTLIER = CLUSTER + [[50.0, 50.0]]


# ------------------------------------------------------------ zscore_flags
def test_zscore_flags_the_point_far_from_the_mean():
    assert zscore_flags([[0.0], [0.0], [0.0], [10.0]], 1.5) == [
        False,
        False,
        False,
        True,
    ]


def test_zscore_flags_nothing_when_everything_is_close():
    assert zscore_flags([[1.0], [2.0], [3.0], [2.0]], 3.0) == [False] * 4


def test_zscore_survives_a_constant_column():
    """Ловушка: std = 0 и деление на ноль на ровном месте."""
    assert zscore_flags([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]], 3.0) == [False] * 3


def test_zscore_flags_a_row_if_any_single_feature_is_extreme():
    rows = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 9.0]]
    assert zscore_flags(rows, 1.5)[3] is True


def test_higher_threshold_never_flags_more_rows():
    strict = zscore_flags(WITH_OUTLIER, 4.0)
    loose = zscore_flags(WITH_OUTLIER, 2.0)
    assert sum(strict) <= sum(loose)


def test_zscore_is_blinded_by_a_group_of_outliers():
    """Слабость метода в чистом виде: аномалии сами раздувают std и прячутся.

    Двадцать нулей и три сотни — сотни очевидно аномальны, но они настолько
    сдвинули среднее и разброс, что до порога в три сигмы не дотягивают.
    """
    rows = [[0.0]] * 20 + [[100.0]] * 3
    assert zscore_flags(rows, 3.0) == [False] * 23


# --------------------------------------------------------------- percentile
def test_percentile_interpolates_between_neighbours():
    assert percentile([1, 2, 3, 4], 25) == APPROX(1.75)


def test_percentile_zero_and_hundred_are_min_and_max():
    assert percentile([4, 1, 3, 2], 0) == APPROX(1.0)
    assert percentile([4, 1, 3, 2], 100) == APPROX(4.0)


def test_percentile_of_odd_length_hits_an_exact_element():
    assert percentile([1, 2, 3], 50) == APPROX(2.0)


def test_percentile_does_not_reorder_the_input():
    """Ловушка: sort() на месте переставит данные у вызывающего."""
    values = [4, 1, 3, 2]
    percentile(values, 50)
    assert values == [4, 1, 3, 2]


def test_percentile_ignores_input_order():
    assert percentile([4, 1, 3, 2], 25) == APPROX(percentile([1, 2, 3, 4], 25))


# --------------------------------------------------------------- iqr_bounds
def test_iqr_bounds_of_a_short_range():
    lower, upper = iqr_bounds([1, 2, 3, 4])
    assert lower == APPROX(-0.5)
    assert upper == APPROX(5.5)


def test_iqr_bounds_are_robust_to_an_extreme_value():
    """Перцентили не двигаются от того, что один выброс стал огромным.

    Именно этим IQR отличается от z-score, у которого от такого выброса
    среднее и std уезжают сразу.
    """
    normal = iqr_bounds([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    spiked = iqr_bounds([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000])
    assert normal == APPROX(list(spiked))


def test_bigger_factor_widens_the_bounds():
    narrow = iqr_bounds([1, 2, 3, 4], 1.5)
    wide = iqr_bounds([1, 2, 3, 4], 3.0)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


def test_iqr_bounds_survive_a_zero_spread():
    """Вырожденные усы Tukey честно схлопываются в центральное значение."""
    lower, upper = iqr_bounds([5.0] * 10)
    assert lower == APPROX(5.0)
    assert upper == APPROX(5.0)


# ---------------------------------------------------------------- iqr_flags
def test_iqr_flags_the_far_value():
    flags = iqr_flags([[1.0], [2.0], [3.0], [4.0], [100.0]])
    assert flags == [False, False, False, False, True]


def test_iqr_catches_what_zscore_misses():
    """Тот же вход, что и в тесте про ослеплённый z-score, — IQR его берёт."""
    rows = [[0.0]] * 20 + [[100.0]] * 3
    assert zscore_flags(rows, 3.0) == [False] * 23
    assert iqr_flags(rows)[-3:] == [True, True, True]


def test_zero_iqr_flags_do_not_depend_on_measurement_scale():
    """Подстановка IQR=1 скрыла бы малые отклонения, но не те же большие."""
    small = [[0.0]] * 20 + [[0.5]] * 3
    large = [[0.0]] * 20 + [[500.0]] * 3
    assert iqr_flags(small) == iqr_flags(large)
    assert iqr_flags(small)[-3:] == [True, True, True]


def test_iqr_flags_are_per_column():
    """Границы считаются по каждому признаку отдельно, а не по всей матрице."""
    rows = [[0.0, 1000.0]] * 10 + [[0.0, 1001.0], [9.0, 1000.0]]
    assert iqr_flags(rows)[-1] is True


def test_iqr_misses_a_combination_that_is_normal_feature_by_feature():
    """Слабость метода: по каждому признаку точка обычна, вместе — невозможна.

    Все точки лежат на диагонали x == y, кроме одной, у которой координаты
    поменяны местами. Диапазоны обоих столбцов она не нарушает.
    """
    rows = [[float(i), float(i)] for i in range(20)] + [[0.0, 19.0]]
    assert iqr_flags(rows)[-1] is False


# ------------------------------------------------------ expected_path_length
def test_expected_path_length_of_one_point_is_zero():
    assert expected_path_length(1) == APPROX(0.0)


def test_expected_path_length_of_two_points_is_one():
    """Ловушка: приближение через логарифм здесь врёт, точный ответ 1.0."""
    assert expected_path_length(2) == APPROX(1.0)


def test_expected_path_length_grows_with_sample_size():
    assert expected_path_length(8) < expected_path_length(64) < expected_path_length(512)


def test_expected_path_length_of_256_matches_the_paper():
    assert expected_path_length(256) == pytest.approx(10.24, abs=1e-2)


# ------------------------------------- build_isolation_tree / path_length
def test_a_tree_of_identical_points_is_a_leaf():
    """Резать нечего: все значения признака совпали."""
    tree = build_isolation_tree([[1.0], [1.0], [1.0]], 5, random.Random(0))
    assert tree == {"size": 3}


def test_zero_depth_gives_a_leaf():
    tree = build_isolation_tree([[1.0], [2.0], [3.0]], 0, random.Random(0))
    assert tree == {"size": 3}


def test_path_length_of_a_leaf_is_its_expected_path_length():
    assert path_length({"size": 8}, [0.0]) == APPROX(expected_path_length(8))


def test_path_length_follows_the_branch_the_point_belongs_to():
    tree = {
        "feature": 0,
        "threshold": 5.0,
        "left": {"size": 1},
        "right": {"size": 8},
    }
    assert path_length(tree, [1.0]) == APPROX(1.0)
    assert path_length(tree, [9.0]) == APPROX(1.0 + expected_path_length(8))


def test_a_split_tree_isolates_the_lonely_point_faster():
    """Точка в пустоте отрезается меньшим числом разрезов, чем точка в толпе."""
    rows = [[float(i) / 100] for i in range(30)] + [[100.0]]
    tree = build_isolation_tree(rows, 6, random.Random(1))
    lonely = path_length(tree, [100.0])
    crowd = sum(path_length(tree, [float(i) / 100]) for i in range(30)) / 30
    assert lonely < crowd


# ----------------------------------------------------------- isolation_scores
def test_isolation_scores_return_one_number_per_row():
    assert len(isolation_scores(WITH_OUTLIER, n_trees=20, max_samples=16)) == 41


def test_isolation_scores_live_strictly_between_zero_and_one():
    scores = isolation_scores(WITH_OUTLIER, n_trees=20, max_samples=16)
    assert all(0.0 < s < 1.0 for s in scores)


def test_the_outlier_gets_the_highest_score():
    """Ради этого всё и затевалось: далёкая точка изолируется первой."""
    scores = isolation_scores(WITH_OUTLIER, n_trees=40, max_samples=16, seed=7)
    assert scores.index(max(scores)) == len(WITH_OUTLIER) - 1


def test_isolation_scores_are_reproducible_for_a_fixed_seed():
    """Без воспроизводимости детектор невозможно отлаживать."""
    a = isolation_scores(WITH_OUTLIER, n_trees=20, max_samples=16, seed=3)
    b = isolation_scores(WITH_OUTLIER, n_trees=20, max_samples=16, seed=3)
    assert a == APPROX(b)


def test_isolation_score_of_the_outlier_beats_the_cluster_average():
    scores = isolation_scores(WITH_OUTLIER, n_trees=40, max_samples=16, seed=11)
    cluster_mean = sum(scores[:-1]) / (len(scores) - 1)
    assert scores[-1] > cluster_mean


def test_isolation_scores_handle_a_sample_larger_than_the_data():
    """max_samples больше числа строк — подвыборка просто равна всем данным."""
    scores = isolation_scores(CLUSTER[:5], n_trees=10, max_samples=100)
    assert len(scores) == 5
    assert all(math.isfinite(s) for s in scores)
