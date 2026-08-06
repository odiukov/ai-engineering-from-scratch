"""Тесты к уроку «Решающие деревья». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    best_split,
    bootstrap_sample,
    build_regression_tree,
    build_tree,
    entropy,
    feature_importance,
    fit_random_forest,
    forest_predict,
    gini_impurity,
    information_gain,
    split_dataset,
    tree_predict,
    variance_reduction,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# один признак режет выборку пополам, второй — чистый шум
STEP_X = [[1.0, 9.0], [2.0, 3.0], [5.0, 9.0], [6.0, 3.0]]
STEP_Y = [0, 0, 1, 1]

# полоса: класс 1 сидит в середине. Одним порогом такое не режется,
# значит ни линейная, ни логистическая регрессия этого не выучат
BAND_X = [[0.0], [1.0], [2.0], [3.0]]
BAND_Y = [0, 1, 1, 0]


def depth_of(node):
    """Глубина дерева: у листа 0, у развилки на единицу больше глубокого ребёнка."""
    if node["leaf"]:
        return 0
    return 1 + max(depth_of(node["left"]), depth_of(node["right"]))


# ------------------------------------------------------------ gini_impurity
def test_gini_of_a_pure_node_is_zero():
    assert gini_impurity([1, 1, 1]) == APPROX(0.0)


def test_gini_of_a_fifty_fifty_node_is_one_half():
    assert gini_impurity([0, 1]) == APPROX(0.5)


def test_gini_on_a_hand_checked_example():
    assert gini_impurity([0] * 6 + [1] * 4) == APPROX(0.48)


def test_gini_does_not_depend_on_the_order_of_labels():
    assert gini_impurity([0, 0, 1, 1]) == APPROX(gini_impurity([1, 0, 1, 0]))


def test_gini_of_an_empty_node_is_zero_not_a_crash():
    """Ловушка: делить на len([]) нельзя."""
    assert gini_impurity([]) == APPROX(0.0)


def test_gini_grows_with_the_number_of_equally_sized_classes():
    assert gini_impurity([0, 1]) < gini_impurity([0, 1, 2]) < gini_impurity([0, 1, 2, 3])


# ------------------------------------------------------------------ entropy
def test_entropy_of_a_pure_node_is_zero():
    assert entropy([7, 7, 7]) == APPROX(0.0)


def test_entropy_of_a_fifty_fifty_node_is_one_bit():
    assert entropy([0, 1]) == APPROX(1.0)


def test_entropy_of_four_equal_classes_is_two_bits():
    """k равновероятных классов — ровно log2(k) бит."""
    assert entropy([0, 1, 2, 3]) == APPROX(2.0)


def test_entropy_on_a_hand_checked_example():
    expected = -(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4))
    assert entropy([0] * 6 + [1] * 4) == APPROX(expected)


def test_entropy_of_an_empty_node_is_zero_not_a_crash():
    """Ловушка: log2(0) бросает ValueError."""
    assert entropy([]) == APPROX(0.0)


# --------------------------------------------------------- information_gain
def test_information_gain_of_a_perfect_split_equals_the_parent_impurity():
    assert information_gain([0, 0, 1, 1], [0, 0], [1, 1]) == APPROX(0.5)


def test_information_gain_of_a_useless_split_is_zero():
    """Дети такие же грязные, как родитель — разбиение ничего не дало."""
    assert information_gain([0, 0, 1, 1], [0, 1], [0, 1]) == APPROX(0.0)


def test_information_gain_with_an_empty_child_is_zero():
    """Ловушка: разбиения не произошло, а формула без проверки даст выигрыш."""
    assert information_gain([0, 1], [], [0, 1]) == APPROX(0.0)


def test_information_gain_weights_children_by_their_size():
    """Простое среднее по детям переоценило бы отрез одного объекта.

    Родитель [0,0,0,1]: слева чистый [0], справа [0,0,1].
    Взвешенно: 0.25*0 + 0.75*0.444 = 0.333, выигрыш 0.375 - 0.333 = 0.042.
    Невзвешенно получилось бы 0.375 - 0.222 = 0.153 — втрое больше.
    """
    gain = information_gain([0, 0, 0, 1], [0], [0, 0, 1])
    assert gain == pytest.approx(0.375 - 0.75 * gini_impurity([0, 0, 1]), abs=1e-12)


def test_information_gain_works_with_entropy_too():
    assert information_gain([0, 0, 1, 1], [0, 0], [1, 1], "entropy") == APPROX(1.0)


# ------------------------------------------------------------ split_dataset
def test_split_dataset_sends_small_values_left():
    assert split_dataset([[1.0], [3.0]], [0, 1], 0, 2.0) == (
        [[1.0]],
        [0],
        [[3.0]],
        [1],
    )


def test_split_dataset_sends_the_threshold_itself_left():
    """Ловушка: условие <=, а не <. Порог принадлежит левой ветке."""
    _, left_y, _, right_y = split_dataset([[2.0], [2.5]], [0, 1], 0, 2.0)
    assert left_y == [0] and right_y == [1]


def test_split_dataset_keeps_objects_glued_to_their_labels():
    left_X, left_y, right_X, right_y = split_dataset(STEP_X, STEP_Y, 0, 3.0)
    assert left_X == [[1.0, 9.0], [2.0, 3.0]] and left_y == [0, 0]
    assert right_X == [[5.0, 9.0], [6.0, 3.0]] and right_y == [1, 1]


def test_split_dataset_loses_nothing():
    _, left_y, _, right_y = split_dataset(STEP_X, STEP_Y, 1, 5.0)
    assert len(left_y) + len(right_y) == len(STEP_Y)


# --------------------------------------------------------------- best_split
def test_best_split_finds_the_midpoint_between_the_classes():
    assert best_split([[1.0], [2.0], [5.0], [6.0]], [0, 0, 1, 1]) == (
        0,
        APPROX(3.5),
        APPROX(0.5),
    )


def test_best_split_picks_the_informative_feature_over_the_noisy_one():
    """Признак 1 в STEP_X ничего не объясняет — выбор обязан пасть на признак 0."""
    feature, _, _ = best_split(STEP_X, STEP_Y)
    assert feature == 0


def test_best_split_thresholds_are_midpoints_not_data_values():
    """Ловушка: порог, равный значению из данных, может оставить ребёнка пустым."""
    _, threshold, _ = best_split([[1.0], [2.0], [5.0], [6.0]], [0, 0, 1, 1])
    assert threshold not in (1.0, 2.0, 5.0, 6.0)


def test_best_split_gives_up_when_all_labels_are_the_same():
    assert best_split([[1.0], [2.0], [3.0]], [1, 1, 1]) == (None, None, 0.0)


def test_best_split_gives_up_when_all_objects_are_identical():
    """Разных значений нет — и порогов-кандидатов тоже нет."""
    assert best_split([[4.0], [4.0]], [0, 1]) == (None, None, 0.0)


# --------------------------------------------------------------- build_tree
def test_build_tree_on_a_pure_sample_is_a_single_leaf():
    assert build_tree([[1.0], [2.0]], [5, 5]) == {"leaf": True, "value": 5}


def test_build_tree_produces_the_documented_node_shape():
    tree = build_tree([[1.0], [2.0]], [0, 1])
    assert tree["leaf"] is False
    assert tree["feature"] == 0
    assert tree["threshold"] == APPROX(1.5)
    assert tree["left"] == {"leaf": True, "value": 0}
    assert tree["right"] == {"leaf": True, "value": 1}


def test_build_tree_learns_the_training_data_perfectly():
    tree = build_tree(STEP_X, STEP_Y)
    assert tree_predict(tree, STEP_X) == STEP_Y


def test_build_tree_handles_a_band_that_a_straight_line_cannot():
    """Два последовательных разреза вырезают середину — в этом сила дерева."""
    tree = build_tree(BAND_X, BAND_Y)
    assert tree_predict(tree, BAND_X) == BAND_Y
    assert depth_of(tree) >= 2


def test_max_depth_one_produces_a_stump():
    """Ловушка предобрезки: с max_depth=1 оба ребёнка обязаны быть листьями."""
    tree = build_tree(BAND_X, BAND_Y, max_depth=1)
    assert depth_of(tree) == 1
    assert tree["left"]["leaf"] and tree["right"]["leaf"]


def test_max_depth_zero_produces_a_bare_leaf():
    tree = build_tree(STEP_X, STEP_Y, max_depth=0)
    assert tree["leaf"] is True


def test_min_samples_split_stops_the_growth():
    tree = build_tree(STEP_X, STEP_Y, min_samples_split=10)
    assert tree["leaf"] is True


def test_leaf_label_is_the_majority_class():
    tree = build_tree([[1.0], [1.0], [1.0], [1.0]], [0, 0, 0, 1], max_depth=0)
    assert tree["value"] == 0


def test_an_unlimited_tree_memorises_noise():
    """Без обрезки дерево дорастает до чистых листьев — это не обобщение."""
    X = [[float(i)] for i in range(8)]
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    assert tree_predict(build_tree(X, y), X) == y
    assert depth_of(build_tree(X, y, max_depth=1)) == 1


# ------------------------------------------------------------- tree_predict
def test_tree_predict_on_a_bare_leaf_answers_the_same_thing_always():
    assert tree_predict({"leaf": True, "value": 7}, [[0.0], [1.0]]) == [7, 7]


def test_tree_predict_returns_one_label_per_object():
    tree = build_tree(STEP_X, STEP_Y)
    assert len(tree_predict(tree, STEP_X * 3)) == 12


def test_tree_predict_sends_the_threshold_value_left():
    """То же правило, что в split_dataset: разойдутся — дерево соврёт на границе."""
    tree = {
        "leaf": False,
        "feature": 0,
        "threshold": 2.0,
        "left": {"leaf": True, "value": "L"},
        "right": {"leaf": True, "value": "R"},
    }
    assert tree_predict(tree, [[2.0], [2.001]]) == ["L", "R"]


# --------------------------------------------------------- bootstrap_sample
def test_bootstrap_keeps_the_sample_size():
    X = [[float(i)] for i in range(10)]
    X_boot, y_boot = bootstrap_sample(X, list(range(10)), seed=0)
    assert len(X_boot) == 10 and len(y_boot) == 10


def test_bootstrap_keeps_objects_glued_to_their_labels():
    """Ловушка: индекс берётся один на оба списка, а не два независимых."""
    X = [[float(i)] for i in range(20)]
    y = [i * 100 for i in range(20)]
    X_boot, y_boot = bootstrap_sample(X, y, seed=4)
    assert all(label == obj[0] * 100 for obj, label in zip(X_boot, y_boot))


def test_bootstrap_is_reproducible_with_the_same_seed():
    X, y = [[float(i)] for i in range(20)], list(range(20))
    assert bootstrap_sample(X, y, seed=5) == bootstrap_sample(X, y, seed=5)


def test_bootstrap_differs_with_another_seed():
    X, y = [[float(i)] for i in range(20)], list(range(20))
    assert bootstrap_sample(X, y, seed=1) != bootstrap_sample(X, y, seed=2)


def test_bootstrap_draws_with_replacement_so_duplicates_appear():
    _, y_boot = bootstrap_sample([[float(i)] for i in range(30)], list(range(30)), 0)
    assert len(set(y_boot)) < 30


def test_bootstrap_covers_roughly_sixty_three_percent_of_the_originals():
    """Классика бэггинга: ~63% объектов попадают в выборку, остальные out-of-bag."""
    X, y = [[float(i)] for i in range(500)], list(range(500))
    _, y_boot = bootstrap_sample(X, y, seed=0)
    assert 0.58 < len(set(y_boot)) / 500 < 0.68


def test_bootstrap_invents_nothing_new():
    X, y = [[float(i)] for i in range(10)], list(range(10))
    _, y_boot = bootstrap_sample(X, y, seed=9)
    assert set(y_boot) <= set(y)


# ------------------------------------------------------------ случайный лес
def _noisy_dataset(n=60, seed=7):
    """Два информативных признака и два чисто шумовых."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        a = rng.uniform(0, 10)
        b = rng.uniform(0, 10)
        X.append([a, b, rng.uniform(0, 10), rng.uniform(0, 10)])
        y.append(1 if a + b > 10 else 0)
    return X, y


def test_forest_grows_the_requested_number_of_trees():
    X, y = _noisy_dataset()
    assert len(fit_random_forest(X, y, n_trees=7, max_depth=3, seed=0)) == 7


def test_forest_is_reproducible_for_a_given_seed():
    X, y = _noisy_dataset()
    a = fit_random_forest(X, y, n_trees=5, max_depth=3, seed=42)
    b = fit_random_forest(X, y, n_trees=5, max_depth=3, seed=42)
    assert a == b


def test_forest_trees_differ_from_each_other():
    """Если деревья одинаковые, голосование бессмысленно."""
    X, y = _noisy_dataset()
    forest = fit_random_forest(X, y, n_trees=5, max_depth=3, max_features=2, seed=1)
    assert any(t != forest[0] for t in forest[1:])


def test_forest_predict_returns_one_label_per_row():
    X, y = _noisy_dataset()
    forest = fit_random_forest(X, y, n_trees=5, max_depth=3, seed=0)
    assert len(forest_predict(forest, X)) == len(X)


def test_forest_predict_is_a_majority_vote():
    """Два дерева из трёх говорят 1 — ответ 1, даже если третье уверено в 0."""
    ones = {"leaf": True, "value": 1}
    zeros = {"leaf": True, "value": 0}
    assert forest_predict([ones, ones, zeros], [[0.0]]) == [1]


def test_forest_beats_a_single_overfitted_tree_on_held_out_data():
    """Ради этого лес и существует: усреднение гасит разброс."""
    X, y = _noisy_dataset(n=80, seed=3)
    train_X, train_y = X[:50], y[:50]
    test_X, test_y = X[50:], y[50:]

    single = tree_predict(build_tree(train_X, train_y), test_X)
    forest = fit_random_forest(train_X, train_y, n_trees=25, max_features=2, seed=0)
    voted = forest_predict(forest, test_X)

    acc = lambda p: sum(a == b for a, b in zip(p, test_y)) / len(test_y)
    assert acc(voted) >= acc(single)


def test_max_features_actually_restricts_the_choice():
    """С max_features=1 развилка не может выбирать признак свободно."""
    X, y = _noisy_dataset()
    rng = random.Random(0)
    restricted = build_tree(X, y, max_depth=1, max_features=1, rng=rng)
    unrestricted = build_tree(X, y, max_depth=1)
    assert restricted["feature"] != unrestricted["feature"] or restricted != unrestricted


# ------------------------------------------------------- важность признаков
def test_importance_sums_to_one():
    X, y = _noisy_dataset()
    tree = build_tree(X, y, max_depth=4)
    assert sum(feature_importance(tree, X, y)) == pytest.approx(1.0)


def test_importance_has_one_entry_per_feature():
    X, y = _noisy_dataset()
    tree = build_tree(X, y, max_depth=3)
    assert len(feature_importance(tree, X, y)) == len(X[0])


def test_unused_feature_gets_zero_importance():
    X = [[1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    y = [0, 0, 1, 1]
    assert feature_importance(build_tree(X, y), X, y) == pytest.approx([1.0, 0.0])


def test_informative_features_outrank_noise():
    """Первые два признака определяют метку, последние два — шум."""
    X, y = _noisy_dataset(n=100, seed=11)
    imp = feature_importance(build_tree(X, y, max_depth=4), X, y)
    assert min(imp[0], imp[1]) > max(imp[2], imp[3])


def test_importance_of_a_single_leaf_is_all_zeros():
    """Дерево без развилок ничего не объясняет — нормировать нечего."""
    X, y = [[1.0, 2.0]], [0]
    assert feature_importance(build_tree(X, y), X, y) == [0.0, 0.0]


def test_root_split_weighs_more_than_a_deep_one():
    """Развилка у корня решает судьбу всей выборки, глубокая — горстки."""
    X = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]]
    y = [0, 0, 0, 0, 1, 1, 1, 0]
    shallow = feature_importance(build_tree(X, y, max_depth=1), X, y)
    assert shallow == pytest.approx([1.0])


# --------------------------------------------------- дерево для регрессии
def test_variance_reduction_on_a_clean_split():
    assert variance_reduction([1.0, 1.0, 5.0, 5.0], [1.0, 1.0], [5.0, 5.0]) == pytest.approx(4.0)


def test_variance_reduction_is_zero_when_nothing_improves():
    assert variance_reduction([1.0, 5.0], [1.0, 5.0], []) == pytest.approx(0.0)


def test_variance_reduction_of_identical_values_is_zero():
    assert variance_reduction([3.0] * 4, [3.0, 3.0], [3.0, 3.0]) == pytest.approx(0.0)


def test_regression_leaf_holds_the_mean_not_the_mode():
    """Главное отличие от классификации: лист усредняет, а не голосует."""
    tree = build_regression_tree([[0.0], [0.0], [0.0]], [1.0, 2.0, 9.0])
    assert tree == {"leaf": True, "value": pytest.approx(4.0)}


def test_regression_tree_splits_two_levels():
    tree = build_regression_tree([[1.0], [2.0]], [10.0, 20.0])
    assert tree["feature"] == 0
    assert tree["threshold"] == pytest.approx(1.5)
    assert tree["left"]["value"] == pytest.approx(10.0)
    assert tree["right"]["value"] == pytest.approx(20.0)


def test_regression_tree_reuses_tree_predict():
    """tree_predict не знает, метка в листе или число — формат узлов один."""
    X = [[1.0], [2.0], [8.0], [9.0]]
    y = [1.0, 1.0, 100.0, 100.0]
    tree = build_regression_tree(X, y)
    assert tree_predict(tree, X) == pytest.approx(y)


def test_regression_tree_cannot_extrapolate():
    """Справа от обучающих данных дерево навсегда застревает в крайнем листе."""
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [10.0, 20.0, 30.0, 40.0]
    tree = build_regression_tree(X, y)
    assert tree_predict(tree, [[100.0], [1000.0]]) == pytest.approx([40.0, 40.0])


def test_regression_tree_respects_max_depth():
    X = [[float(i)] for i in range(8)]
    y = [float(i) for i in range(8)]
    tree = build_regression_tree(X, y, max_depth=1)
    assert tree["left"]["leaf"] and tree["right"]["leaf"]
