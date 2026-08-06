"""Тесты к уроку «Несбалансированные данные». Правь exercise.py."""

import pytest

from exercise import (
    best_threshold,
    class_weights,
    confusion_counts,
    k_nearest,
    matthews_corrcoef,
    precision_recall_f1,
    random_oversample,
    smote,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# 99 отрицательных и 1 положительный — классика «точность 99%»
SKEWED_TRUE = [1] + [0] * 99
ALWAYS_NEGATIVE = [0] * 100


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in rows for v in row]


# ------------------------------------------------------- confusion_counts
def test_confusion_counts_on_a_mixed_prediction():
    assert confusion_counts([1, 0, 1, 0], [1, 0, 0, 0]) == (1, 2, 0, 1)


def test_confusion_counts_of_a_perfect_prediction_has_no_errors():
    tp, tn, fp, fn = confusion_counts([1, 0, 1], [1, 0, 1])
    assert (fp, fn) == (0, 0)
    assert (tp, tn) == (2, 1)


def test_confusion_counts_sum_to_the_sample_size():
    assert sum(confusion_counts(SKEWED_TRUE, ALWAYS_NEGATIVE)) == 100


def test_always_negative_model_catches_nothing():
    tp, tn, fp, fn = confusion_counts(SKEWED_TRUE, ALWAYS_NEGATIVE)
    assert tp == 0 and fn == 1 and tn == 99 and fp == 0


def test_confusion_counts_do_not_confuse_fp_with_fn():
    """Ложная тревога и пропуск — разные ошибки с разной ценой."""
    assert confusion_counts([1, 1], [0, 0]) == (0, 0, 0, 2)
    assert confusion_counts([0, 0], [1, 1]) == (0, 0, 2, 0)


# ---------------------------------------------------- precision_recall_f1
def test_precision_recall_f1_on_a_mixed_prediction():
    p, r, f1 = precision_recall_f1([1, 0, 1, 0], [1, 0, 0, 0])
    assert p == APPROX(1.0)
    assert r == APPROX(0.5)
    assert f1 == APPROX(2 / 3)


def test_f1_is_the_harmonic_mean_not_the_average():
    """Среднее арифметическое дало бы 0.75, гармоническое — 0.667."""
    assert precision_recall_f1([1, 0, 1, 0], [1, 0, 0, 0])[2] == APPROX(2 / 3)


def test_ninety_nine_percent_accuracy_scores_zero_f1():
    """Модель «всегда ноль» ошибается один раз из ста — и не стоит ничего."""
    assert precision_recall_f1(SKEWED_TRUE, ALWAYS_NEGATIVE) == APPROX(
        [0.0, 0.0, 0.0]
    )


def test_perfect_prediction_scores_one():
    assert precision_recall_f1([1, 0, 1], [1, 0, 1]) == APPROX([1.0, 1.0, 1.0])


def test_flagging_everything_gives_full_recall_and_low_precision():
    p, r, _ = precision_recall_f1(SKEWED_TRUE, [1] * 100)
    assert r == APPROX(1.0)
    assert p == APPROX(0.01)


def test_precision_survives_a_model_that_never_predicts_one():
    """Ловушка: tp + fp = 0, деление на ноль там, где хочется 0.0."""
    assert precision_recall_f1([0, 0], [0, 0])[0] == APPROX(0.0)


# ------------------------------------------------------ matthews_corrcoef
def test_mcc_of_a_perfect_prediction_is_one():
    assert matthews_corrcoef([1, 0, 1, 0], [1, 0, 1, 0]) == APPROX(1.0)


def test_mcc_of_an_inverted_prediction_is_minus_one():
    assert matthews_corrcoef([1, 0, 1, 0], [0, 1, 0, 1]) == APPROX(-1.0)


def test_mcc_of_the_lazy_majority_model_is_zero():
    """Ловушка и смысл сразу: знаменатель нулевой, и балл честно нулевой."""
    assert matthews_corrcoef(SKEWED_TRUE, ALWAYS_NEGATIVE) == APPROX(0.0)


def test_mcc_needs_success_on_both_classes():
    """Поймали единицу, но завалили нули — MCC заметно ниже единицы."""
    y_true = [1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0]
    assert 0.0 < matthews_corrcoef(y_true, y_pred) < 0.7


# ---------------------------------------------------------- class_weights
def test_class_weights_match_the_lesson_example():
    weights = class_weights([0] * 950 + [1] * 50)
    assert weights[0] == APPROX(1000 / (2 * 950))
    assert weights[1] == APPROX(10.0)


def test_balanced_data_gets_unit_weights():
    assert class_weights([0, 0, 1, 1]) == APPROX({0: 1.0, 1: 1.0})


def test_weight_times_count_is_the_same_for_every_class():
    """В этом вся идея: вклад классов в функцию потерь выравнивается."""
    y = [0] * 90 + [1] * 10
    weights = class_weights(y)
    assert weights[0] * 90 == APPROX(weights[1] * 10)


def test_rarer_class_gets_the_bigger_weight():
    weights = class_weights([0] * 80 + [1] * 20)
    assert weights[1] > weights[0]


def test_class_weights_work_for_three_classes():
    weights = class_weights(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    assert weights["c"] > weights["b"] > weights["a"]


# --------------------------------------------------------------- k_nearest
def test_k_nearest_returns_the_closest_neighbour():
    assert k_nearest([[0.0], [1.0], [5.0]], 0, 1) == [1]


def test_k_nearest_never_returns_the_point_itself():
    assert 0 not in k_nearest([[0.0], [0.0], [1.0]], 0, 2)


def test_k_nearest_is_ordered_by_distance():
    assert k_nearest([[0.0], [3.0], [1.0], [2.0]], 0, 3) == [2, 3, 1]


def test_k_nearest_returns_everything_it_has_when_k_is_too_big():
    assert k_nearest([[0.0], [1.0], [5.0]], 0, 5) == [1, 2]


def test_k_nearest_works_in_two_dimensions():
    points = [[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]]
    assert k_nearest(points, 0, 1) == [2]


# ------------------------------------------------------------------- smote
def test_smote_generates_the_requested_number_of_points():
    assert len(smote([[0.0, 0.0], [1.0, 1.0]], 7, k=1)) == 7


def test_smote_points_land_on_the_segment_between_real_points():
    """Точки строго на отрезке между (0,0) и (1,1): x == y и обе в [0, 1]."""
    for x, y in smote([[0.0, 0.0], [1.0, 1.0]], 20, k=1, seed=1):
        assert x == APPROX(y)
        assert 0.0 <= x <= 1.0


def test_smote_does_not_just_copy_existing_points():
    """Отличие от простого дублирования: хотя бы часть точек строго внутри."""
    generated = smote([[0.0], [1.0]], 20, k=1, seed=2)
    assert any(0.001 < p[0] < 0.999 for p in generated)


def test_smote_is_reproducible_for_a_fixed_seed():
    a = smote([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]], 10, k=2, seed=5)
    b = smote([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]], 10, k=2, seed=5)
    assert flat(a) == APPROX(flat(b))


def test_smote_stays_inside_the_bounding_box_of_the_minority():
    """Интерполяция, а не экстраполяция: за пределы облака выходить нельзя."""
    minority = [[0.0, 0.0], [1.0, 3.0], [2.0, 1.0], [0.5, 2.0]]
    for x, y in smote(minority, 30, k=3, seed=4):
        assert 0.0 <= x <= 2.0
        assert 0.0 <= y <= 3.0


def test_smote_refuses_to_interpolate_a_single_point():
    with pytest.raises(ValueError):
        smote([[1.0, 1.0]], 5)


# -------------------------------------------------------- random_oversample
def test_random_oversample_balances_the_classes():
    X = [[float(i)] for i in range(10)]
    y = [0] * 8 + [1] * 2
    _, y_out = random_oversample(X, y)
    assert y_out.count(0) == y_out.count(1)


def test_random_oversample_keeps_the_original_rows_first():
    X = [[0.0], [1.0], [2.0]]
    y = [0, 0, 1]
    X_out, y_out = random_oversample(X, y)
    assert flat(X_out[:3]) == APPROX([0.0, 1.0, 2.0])
    assert y_out[:3] == [0, 0, 1]


def test_random_oversample_only_adds_duplicates_of_real_points():
    """Ничего нового не изобретается — в этом и разница со SMOTE."""
    X = [[0.0], [1.0], [2.0]]
    X_out, _ = random_oversample(X, [0, 0, 1])
    assert all(row in X for row in X_out)


def test_random_oversample_leaves_balanced_data_alone():
    X = [[0.0], [1.0]]
    X_out, y_out = random_oversample(X, [0, 1])
    assert len(y_out) == 2
    assert flat(X_out) == APPROX([0.0, 1.0])


def test_random_oversample_does_not_mutate_the_input():
    X = [[0.0], [1.0], [2.0]]
    y = [0, 0, 1]
    random_oversample(X, y)
    assert len(X) == 3 and len(y) == 3


# ---------------------------------------------------------- best_threshold
def test_best_threshold_beats_the_default_half():
    """Модель ранжирует верно, но не уверена — 0.5 отсекает вообще всё."""
    y_true = [0] * 18 + [1, 1]
    probs = [0.1] * 18 + [0.2, 0.25]
    threshold, f1 = best_threshold(y_true, probs)
    assert f1 == APPROX(1.0)
    assert threshold < 0.5


def test_best_threshold_returns_the_f1_it_promises():
    y_true = [0, 0, 1, 1, 0, 1]
    probs = [0.1, 0.4, 0.35, 0.8, 0.2, 0.6]
    threshold, f1 = best_threshold(y_true, probs)
    y_pred = [1 if p >= threshold else 0 for p in probs]
    assert precision_recall_f1(y_true, y_pred)[2] == APPROX(f1)


def test_best_threshold_reaches_one_on_separable_scores():
    y_true = [0, 0, 0, 1, 1]
    probs = [0.05, 0.06, 0.07, 0.9, 0.95]
    assert best_threshold(y_true, probs)[1] == APPROX(1.0)


def test_best_threshold_stays_inside_the_sweep_range():
    y_true = [0, 1, 0, 1]
    probs = [0.2, 0.7, 0.3, 0.8]
    threshold, _ = best_threshold(y_true, probs)
    assert 0.05 <= threshold <= 0.95


def test_best_threshold_prefers_the_lower_threshold_on_a_tie():
    """При равном F1 берём порог пониже: пропуск редкого класса дороже."""
    y_true = [0, 0, 1, 1]
    probs = [0.1, 0.1, 0.9, 0.9]
    threshold, f1 = best_threshold(y_true, probs)
    assert f1 == APPROX(1.0)
    assert threshold < 0.2
