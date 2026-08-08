"""Тесты к уроку «Оценка моделей». Правь exercise.py."""

import math

import pytest

from exercise import (
    accuracy,
    auc_roc,
    confusion_matrix,
    cross_val_score,
    kfold_split,
    precision_recall_f1,
    regression_metrics,
    stratified_kfold_split,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# 8 нулей и 2 единицы: классический дисбаланс, на котором accuracy врёт
IMBALANCED_Y = [0] * 8 + [1] * 2
IMBALANCED_X = list(range(10))


def majority_fit(x_train, y_train):
    """«Модель», которая запоминает самый частый класс обучающей выборки."""
    return max(set(y_train), key=y_train.count)


def majority_predict(model, x):
    return model


# -------------------------------------------------------- confusion_matrix
def test_confusion_matrix_counts_all_four_buckets():
    assert confusion_matrix([1, 1, 0, 0], [1, 0, 0, 0]) == (1, 2, 0, 1)


def test_confusion_matrix_of_a_perfect_model_has_no_errors():
    tp, tn, fp, fn = confusion_matrix([1, 0, 1, 0], [1, 0, 1, 0])
    assert (fp, fn) == (0, 0)
    assert (tp, tn) == (2, 2)


def test_confusion_matrix_buckets_sum_to_the_sample_size():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [0, 0, 1, 1, 1, 0, 0]
    assert sum(confusion_matrix(y_true, y_pred)) == len(y_true)


def test_confusion_matrix_does_not_confuse_fp_with_fn():
    """Ловушка порядка: (tp, tn, fp, fn). Ложная тревога — это fp, не fn."""
    tp, tn, fp, fn = confusion_matrix([0], [1])
    assert (fp, fn) == (1, 0)


# ---------------------------------------------------------------- accuracy
def test_accuracy_counts_the_share_of_correct_answers():
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == APPROX(0.75)


def test_accuracy_of_a_perfect_model_is_one():
    assert accuracy([1, 0, 1], [1, 0, 1]) == APPROX(1.0)


def test_accuracy_lies_on_imbalanced_data():
    """Модель «всегда 0» не нашла ни одной единицы, а accuracy у неё 0.8."""
    always_negative = [0] * len(IMBALANCED_Y)
    assert accuracy(IMBALANCED_Y, always_negative) == APPROX(0.8)


# ------------------------------------------------------ precision_recall_f1
def test_precision_recall_f1_known_values():
    assert precision_recall_f1([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(
        (1.0, 0.5, 2 / 3)
    )


def test_precision_recall_f1_of_a_perfect_model_is_all_ones():
    assert precision_recall_f1([1, 0, 1], [1, 0, 1]) == APPROX((1.0, 1.0, 1.0))


def test_precision_recall_f1_of_an_always_negative_model_is_all_zeros():
    """Ловушка: знаменатель precision равен нулю. Ноль, а не ZeroDivisionError."""
    always_negative = [0] * len(IMBALANCED_Y)
    assert precision_recall_f1(IMBALANCED_Y, always_negative) == APPROX((0.0, 0.0, 0.0))


def test_f1_is_the_harmonic_mean_not_the_arithmetic_one():
    """При precision 1.0 и recall 0.2 среднее дало бы 0.6, а f1 честно ниже трети."""
    y_true = [1] * 5 + [0] * 5
    y_pred = [1] + [0] * 9
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    assert (precision, recall) == APPROX((1.0, 0.2))
    assert f1 == APPROX(2 * 1.0 * 0.2 / 1.2)


def test_precision_and_recall_move_in_opposite_directions():
    """Модель, которая кричит «1» на всё: recall максимальный, precision низкий."""
    always_positive = [1] * len(IMBALANCED_Y)
    precision, recall, _ = precision_recall_f1(IMBALANCED_Y, always_positive)
    assert recall == APPROX(1.0)
    assert precision == APPROX(0.2)


# ----------------------------------------------------------------- auc_roc
def test_auc_roc_of_perfect_ranking_is_one():
    assert auc_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == APPROX(1.0)


def test_auc_roc_of_reversed_ranking_is_zero():
    assert auc_roc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == APPROX(0.0)


def test_auc_roc_of_constant_scores_is_a_coin_flip():
    """Ловушка: без стартовой точки (0, 0) здесь получится 0 вместо 0.5."""
    assert auc_roc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == APPROX(0.5)


def test_auc_roc_ignores_the_threshold():
    """Все скоры ниже 0.5 — при обычном пороге модель не угадала бы ничего,
    но ранжирует она идеально, и AUC это видит."""
    assert auc_roc([0, 0, 1, 1], [0.01, 0.02, 0.03, 0.04]) == APPROX(1.0)


def test_auc_roc_survives_a_monotone_transform_of_scores():
    """AUC меряет порядок, а не сами числа: возведение в квадрат ничего не меняет."""
    y_true = [0, 1, 0, 1, 1, 0]
    scores = [0.1, 0.4, 0.35, 0.8, 0.6, 0.2]
    assert auc_roc(y_true, scores) == APPROX(auc_roc(y_true, [s * s for s in scores]))


def test_auc_roc_is_undefined_without_both_classes():
    assert auc_roc([1, 1, 1], [0.1, 0.5, 0.9]) == APPROX(0.5)


# ------------------------------------------------------ regression_metrics
def test_regression_metrics_known_values():
    m = regression_metrics([0, 10], [5, 5])
    assert m["mse"] == APPROX(25.0)
    assert m["rmse"] == APPROX(5.0)
    assert m["mae"] == APPROX(5.0)


def test_rmse_is_the_square_root_of_mse():
    m = regression_metrics([1.0, 2.0, 3.0, 10.0], [1.5, 2.5, 2.0, 4.0])
    assert m["rmse"] == APPROX(math.sqrt(m["mse"]))


def test_r2_of_a_perfect_model_is_one():
    assert regression_metrics([1, 2, 3], [1, 2, 3])["r2"] == APPROX(1.0)


def test_r2_of_the_mean_baseline_is_zero():
    """Ноль означает «не лучше, чем всегда предсказывать среднее»."""
    y_true = [1.0, 2.0, 6.0]
    mean = sum(y_true) / len(y_true)
    assert regression_metrics(y_true, [mean] * 3)["r2"] == APPROX(0.0)


def test_r2_goes_negative_when_the_model_is_worse_than_the_mean():
    assert regression_metrics([1.0, 2.0, 3.0], [100.0, 100.0, 100.0])["r2"] < 0


def test_mae_is_more_robust_to_an_outlier_than_mse():
    """Один промах на 10 даёт mse 100 и mae всего 10 — вот почему при выбросах
    оптимизируют mae."""
    m = regression_metrics([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 20.0])
    assert m["mse"] == APPROX(100.0)
    assert m["mae"] == APPROX(5.0)


# -------------------------------------------------------------- kfold_split
def test_kfold_split_returns_k_folds():
    assert len(kfold_split(10, k=5, seed=0)) == 5


def test_kfold_split_validates_every_index_exactly_once():
    folds = kfold_split(10, k=3, seed=0)
    seen = [i for _, val in folds for i in val]
    assert sorted(seen) == list(range(10))


def test_kfold_split_never_puts_an_index_in_train_and_val_at_once():
    for train, val in kfold_split(12, k=4, seed=1):
        assert set(train).isdisjoint(val)
        assert len(train) + len(val) == 12


def test_kfold_split_spreads_the_remainder_across_folds():
    """Ловушка: остаток раздаётся первым фолдам, а не целиком последнему."""
    sizes = [len(val) for _, val in kfold_split(10, k=3, seed=0)]
    assert sizes == [4, 3, 3]


def test_kfold_split_is_reproducible_for_the_same_seed():
    assert kfold_split(20, k=4, seed=7) == kfold_split(20, k=4, seed=7)


# --------------------------------------------------- stratified_kfold_split
def test_stratified_kfold_keeps_the_class_ratio_in_every_fold():
    y = [0] * 8 + [1] * 4
    for _, val in stratified_kfold_split(y, k=4, seed=0):
        labels = [y[i] for i in val]
        assert labels.count(0) == 2
        assert labels.count(1) == 1


def test_stratified_kfold_gives_every_fold_a_minority_sample():
    """Ради этого стратификация и нужна: recall на фолде без единиц не определён."""
    for _, val in stratified_kfold_split(IMBALANCED_Y, k=2, seed=0):
        assert any(IMBALANCED_Y[i] == 1 for i in val)


def test_stratified_kfold_validates_every_index_exactly_once():
    y = [0] * 9 + [1] * 6
    seen = [i for _, val in stratified_kfold_split(y, k=3, seed=0) for i in val]
    assert sorted(seen) == list(range(15))


def test_stratified_kfold_is_reproducible_for_the_same_seed():
    y = [0] * 9 + [1] * 6
    assert stratified_kfold_split(y, k=3, seed=2) == stratified_kfold_split(y, k=3, seed=2)


# ---------------------------------------------------------- cross_val_score
def test_cross_val_score_returns_one_score_per_fold():
    scores = cross_val_score(
        IMBALANCED_X, IMBALANCED_Y, majority_fit, majority_predict, k=5, seed=0
    )
    assert len(scores) == 5


def test_cross_val_score_of_the_majority_baseline_on_stratified_folds():
    """В каждом фолде 4 нуля и 1 единица, база угадывает нули — ровно 0.8."""
    scores = cross_val_score(
        IMBALANCED_X, IMBALANCED_Y, majority_fit, majority_predict,
        k=2, seed=0, stratified=True,
    )
    assert scores == APPROX([0.8, 0.8])


def test_cross_val_score_uses_the_metric_you_pass():
    """Тот же прогон под f1 вместо accuracy: база не поймала ни одной единицы."""
    f1_only = lambda t, p: precision_recall_f1(t, p)[2]
    scores = cross_val_score(
        IMBALANCED_X, IMBALANCED_Y, majority_fit, majority_predict,
        metric_fn=f1_only, k=2, seed=0, stratified=True,
    )
    assert scores == APPROX([0.0, 0.0])


def test_cross_val_score_is_reproducible_for_the_same_seed():
    args = (IMBALANCED_X, IMBALANCED_Y, majority_fit, majority_predict)
    assert cross_val_score(*args, k=5, seed=3) == cross_val_score(*args, k=5, seed=3)


def test_cross_val_score_refits_the_model_on_every_fold():
    """Ловушка утечки: обучать один раз на всех данных нельзя. Считаем, сколько
    раз позвали fit_fn — обязано быть ровно k."""
    calls = []

    def counting_fit(x_train, y_train):
        calls.append(len(x_train))
        return majority_fit(x_train, y_train)

    cross_val_score(IMBALANCED_X, IMBALANCED_Y, counting_fit, majority_predict, k=5, seed=0)
    assert len(calls) == 5
    assert all(size == 8 for size in calls)
