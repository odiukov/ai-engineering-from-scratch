"""Тесты к уроку «Что такое машинное обучение». Правь exercise.py."""

import random

import pytest

from exercise import (
    accuracy,
    confusion_counts,
    f1,
    majority_baseline,
    precision,
    recall,
    train_test_split,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# -------------------------------------------------------- train_test_split
def test_split_sizes_follow_test_size():
    X, y = list(range(10)), [0] * 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3, seed=0)
    assert (len(X_train), len(X_test)) == (7, 3)
    assert (len(y_train), len(y_test)) == (7, 3)


def test_split_loses_nothing_and_duplicates_nothing():
    """Объединение train и test — ровно исходная выборка, без потерь и копий."""
    X = list(range(20))
    X_train, X_test, _, _ = train_test_split(X, X, 0.25, seed=3)
    assert sorted(X_train + X_test) == X


def test_split_keeps_objects_glued_to_their_labels():
    """Ловушка: X и y перемешаны одной перестановкой, не двумя разными."""
    X = list(range(20))
    y = [x * 10 for x in X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.25, seed=1)
    assert all(label == obj * 10 for obj, label in zip(X_train, y_train))
    assert all(label == obj * 10 for obj, label in zip(X_test, y_test))


def test_split_is_reproducible_with_the_same_seed():
    X, y = list(range(30)), list(range(30))
    assert train_test_split(X, y, 0.2, seed=7) == train_test_split(X, y, 0.2, seed=7)


def test_split_differs_with_another_seed():
    X, y = list(range(30)), list(range(30))
    assert train_test_split(X, y, 0.2, seed=1) != train_test_split(X, y, 0.2, seed=2)


def test_split_does_not_touch_the_global_random():
    """Ловушка: random.seed(seed) внутри функции ломает случайность снаружи."""
    random.seed(123)
    expected = [random.random() for _ in range(3)]
    random.seed(123)
    train_test_split(list(range(20)), [0] * 20, 0.25, seed=7)
    assert [random.random() for _ in range(3)] == expected


def test_split_with_zero_test_size_gives_an_empty_test_set():
    X_train, X_test, _, _ = train_test_split(list(range(10)), [0] * 10, 0.0)
    assert len(X_train) == 10
    assert X_test == []


# -------------------------------------------------------- confusion_counts
def test_confusion_counts_on_a_hand_checked_example():
    assert confusion_counts([1, 1, 0, 0], [1, 0, 0, 0]) == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "tn": 2,
    }


def test_confusion_counts_distinguishes_false_positive_from_false_negative():
    """Ловушка: fp — ложная тревога, fn — пропуск. Перепутать легко."""
    assert confusion_counts([1, 0], [0, 1]) == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}


def test_confusion_counts_sum_up_to_the_sample_size():
    y_true = [1, 0, 1, 1, 0, 0, 1]
    y_pred = [1, 1, 0, 1, 0, 1, 1]
    assert sum(confusion_counts(y_true, y_pred).values()) == len(y_true)


def test_confusion_counts_on_a_perfect_model_has_no_errors():
    counts = confusion_counts([1, 0, 1, 0], [1, 0, 1, 0])
    assert counts["fp"] == 0 and counts["fn"] == 0


# ---------------------------------------------------------------- accuracy
def test_accuracy_counts_both_kinds_of_correct_answers():
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == APPROX(0.75)


def test_accuracy_of_a_perfect_model_is_one():
    assert accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == APPROX(1.0)


def test_accuracy_of_an_inverted_model_is_zero():
    assert accuracy([1, 0, 1, 0], [0, 1, 0, 1]) == APPROX(0.0)


def test_accuracy_on_empty_input_is_zero_not_a_crash():
    assert accuracy([], []) == APPROX(0.0)


# --------------------------------------------------------------- precision
def test_precision_counts_only_the_raised_alarms():
    assert precision([1, 1, 0], [1, 1, 1]) == APPROX(2 / 3)


def test_precision_without_any_positive_prediction_is_zero():
    """Ловушка: знаменатель tp + fp нулевой — делить нельзя."""
    assert precision([1, 0], [0, 0]) == APPROX(0.0)


def test_precision_ignores_true_negatives():
    """Гора верно отброшенных нулей не улучшает precision."""
    few = precision([1, 1, 0], [1, 0, 1])
    many = precision([1, 1, 0] + [0] * 100, [1, 0, 1] + [0] * 100)
    assert few == APPROX(many)


# ------------------------------------------------------------------ recall
def test_recall_counts_the_caught_positives():
    assert recall([1, 1, 0], [1, 0, 0]) == APPROX(0.5)


def test_recall_without_any_real_positive_is_zero():
    """Ловушка: знаменатель tp + fn нулевой — делить нельзя."""
    assert recall([0, 0], [0, 0]) == APPROX(0.0)


def test_recall_is_one_when_the_model_shouts_yes_at_everything():
    """Полнота максимальна у модели "всё положительное" — и она бесполезна."""
    assert recall([1, 0, 0, 0], [1, 1, 1, 1]) == APPROX(1.0)


def test_precision_and_recall_are_not_the_same_number():
    y_true, y_pred = [1, 1, 0], [1, 0, 0]
    assert precision(y_true, y_pred) == APPROX(1.0)
    assert recall(y_true, y_pred) == APPROX(0.5)


# ---------------------------------------------------------------------- f1
def test_f1_of_a_hand_checked_example():
    assert f1([1, 1, 0], [1, 0, 0]) == APPROX(2 / 3)


def test_f1_of_a_perfect_model_is_one():
    assert f1([1, 0, 1], [1, 0, 1]) == APPROX(1.0)


def test_f1_punishes_a_lopsided_model_harder_than_the_plain_average():
    """Гармоническое среднее: p = 1.0 и r = 0.02 дают меньше 0.05, не 0.51."""
    y_true = [1] * 50 + [0] * 50
    y_pred = [1] + [0] * 49 + [0] * 50
    assert precision(y_true, y_pred) == APPROX(1.0)
    assert recall(y_true, y_pred) == APPROX(0.02)
    assert f1(y_true, y_pred) < 0.05


def test_f1_is_zero_when_precision_and_recall_are_both_zero():
    assert f1([1, 1], [0, 0]) == APPROX(0.0)


# -------------------------------------------------------- majority_baseline
def test_majority_baseline_repeats_the_most_common_class():
    assert majority_baseline([0, 0, 0, 1], 3) == [0, 0, 0]


def test_majority_baseline_picks_the_smaller_label_on_a_tie():
    """Детерминированность: при ничьей ответ не должен зависеть от порядка."""
    assert majority_baseline([1, 0], 2) == [0, 0]


def test_majority_baseline_returns_exactly_n_predictions():
    assert len(majority_baseline([1, 1, 0], 7)) == 7


# ----------------------------------------------- accuracy на перекосе классов
def test_useless_model_gets_ninety_percent_accuracy_on_imbalanced_data():
    """Главный урок: 90% accuracy у модели, которая не поймала ни одного класса 1.

    90 нулей и 10 единиц. Baseline "всегда 0" не находит ни одной единицы,
    но accuracy показывает 0.9 — цифра, которую не стыдно показать
    начальнику. precision, recall и f1 при этом честные нули.
    """
    y_true = [0] * 90 + [1] * 10
    y_pred = majority_baseline(y_true, len(y_true))
    assert accuracy(y_true, y_pred) == APPROX(0.9)
    assert precision(y_true, y_pred) == APPROX(0.0)
    assert recall(y_true, y_pred) == APPROX(0.0)
    assert f1(y_true, y_pred) == APPROX(0.0)


def test_a_worse_looking_accuracy_can_hide_a_better_model():
    """Модель с accuracy 0.85 полезнее baseline с 0.9, если она ловит редкий класс."""
    y_true = [0] * 90 + [1] * 10
    baseline = majority_baseline(y_true, len(y_true))
    # ловит 8 единиц из 10 ценой 13 ложных тревог
    smart = [0] * 77 + [1] * 13 + [1] * 8 + [0] * 2
    assert accuracy(y_true, smart) < accuracy(y_true, baseline)
    assert recall(y_true, smart) == APPROX(0.8)
    assert f1(y_true, smart) > f1(y_true, baseline)
