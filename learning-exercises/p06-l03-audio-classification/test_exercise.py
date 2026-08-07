"""Тесты к уроку «Классификация аудио». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    average_precision,
    confusion_matrix,
    cosine_similarity,
    knn_classify,
    macro_f1,
    mixup,
    spec_augment,
    summarize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


def _grid(n_t, n_f):
    """Матрица без нулей: любой ноль в результате точно поставлен маской."""
    return [[1.0 + t * n_f + f for f in range(n_f)] for t in range(n_t)]


# ----------------------------------------------------------------- summarize
def test_summarize_returns_mean_then_variance():
    assert summarize([[1.0, 10.0], [3.0, 10.0]]) == APPROX([2.0, 10.0, 1.0, 0.0])


def test_summarize_doubles_the_feature_count():
    frames = [[float(i + j) for j in range(13)] for i in range(7)]
    assert len(summarize(frames)) == 26


def test_summarize_ignores_frame_order():
    """Mean+var pooling теряет время: перемешанные кадры дают тот же вектор.

    Это цена фиксированной длины — по такому эмбеддингу «да» и «ад»
    неразличимы.
    """
    frames = [[1.0, 5.0], [4.0, 2.0], [7.0, 9.0]]
    assert summarize(frames[::-1]) == APPROX(summarize(frames))


def test_summarize_scales_variance_quadratically():
    """Умножили сигнал на 2: среднее выросло вдвое, дисперсия — вчетверо."""
    frames = [[1.0], [3.0], [8.0]]
    louder = [[2 * v for v in f] for f in frames]
    base = summarize(frames)
    assert summarize(louder) == APPROX([2 * base[0], 4 * base[1]])


# --------------------------------------------------------- cosine_similarity
def test_cosine_of_a_vector_with_itself_is_one():
    assert cosine_similarity([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0, abs=1e-12)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_ignores_vector_length():
    """Громкость клипа меняет длину эмбеддинга, но не направление."""
    quiet = [1.0, 2.0, 3.0]
    loud = [100.0, 200.0, 300.0]
    assert cosine_similarity(quiet, loud) == pytest.approx(1.0, abs=1e-12)


def test_cosine_with_a_zero_vector_does_not_divide_by_zero():
    """Ловушка: тихий клип после нормализации бывает нулевым вектором."""
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == APPROX(0.0)


# ------------------------------------------------------------- knn_classify
def test_knn_with_k_one_returns_the_nearest_label():
    bank = [[1.0, 0.0], [0.0, 1.0]]
    assert knn_classify([1.0, 0.1], bank, ["a", "b"], k=1) == "a"


def test_knn_majority_outvotes_the_single_nearest_neighbour():
    """Смысл k>1: одинокий близкий сосед не должен решать за всех."""
    bank = [[1.0, 0.01], [1.0, 0.5], [1.0, 0.6]]
    labels = ["b", "a", "a"]
    assert knn_classify([1.0, 0.0], bank, labels, k=1) == "b"
    assert knn_classify([1.0, 0.0], bank, labels, k=3) == "a"


def test_knn_answer_does_not_depend_on_query_scale():
    bank = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.2]]
    labels = ["a", "b", "a"]
    assert knn_classify([2.0, 0.0], bank, labels, k=3) == knn_classify(
        [0.001, 0.0], bank, labels, k=3
    )


def test_knn_breaks_a_tie_towards_the_closer_neighbour():
    """Ничья 1:1 разруливается детерминированно, иначе тесты бы плавали."""
    bank = [[1.0, 0.0], [0.0, 1.0]]
    assert knn_classify([1.0, 0.0], bank, ["a", "b"], k=2) == "a"
    assert knn_classify([0.0, 1.0], bank, ["a", "b"], k=2) == "b"


# -------------------------------------------------------------------- mixup
def test_mixup_takes_the_midpoint_of_both_features_and_labels():
    x, y = mixup([0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [0.0, 1.0], 0.5)
    assert x == APPROX([5.0, 5.0])
    assert y == APPROX([0.5, 0.5])


def test_lambda_one_returns_the_first_example_untouched():
    x, y = mixup([1.0], [1.0, 0.0], [9.0], [0.0, 1.0], 1.0)
    assert x == APPROX([1.0])
    assert y == APPROX([1.0, 0.0])


def test_mixup_labels_still_sum_to_one():
    """Ловушка: метку тоже надо смешивать, иначе получается битая разметка."""
    _, y = mixup([0.0], [1.0, 0.0, 0.0], [1.0], [0.0, 0.0, 1.0], 0.3)
    assert sum(y) == APPROX(1.0)


def test_swapping_the_pair_mirrors_lambda():
    a = mixup([0.0], [1.0, 0.0], [4.0], [0.0, 1.0], 0.25)
    b = mixup([4.0], [0.0, 1.0], [0.0], [1.0, 0.0], 0.75)
    assert a[0] == APPROX(b[0])
    assert a[1] == APPROX(b[1])


# ------------------------------------------------------------- spec_augment
def test_same_seed_gives_the_same_augmentation():
    spec = _grid(8, 6)
    a = spec_augment(spec, random.Random(7), 2, 2, 3, 3)
    b = spec_augment(spec, random.Random(7), 2, 2, 3, 3)
    assert flat(a) == APPROX(flat(b))


def test_spec_augment_does_not_mutate_the_input():
    """Ловушка: правка на месте испортит датасет для следующей эпохи."""
    spec = _grid(8, 6)
    before = flat(spec)
    spec_augment(spec, random.Random(3), 2, 2, 4, 4)
    assert flat(spec) == APPROX(before)


def test_zero_width_masks_leave_the_spectrogram_alone():
    spec = _grid(4, 5)
    out = spec_augment(spec, random.Random(0), 2, 2, 0, 0)
    assert flat(out) == APPROX(flat(spec))


def test_a_time_mask_zeroes_whole_frames():
    """Маска по времени вырезает кадр целиком, а не отдельные бины."""
    spec = _grid(10, 6)
    out = spec_augment(spec, random.Random(11), n_time_masks=2, n_freq_masks=0,
                       time_width=3)
    for row, orig in zip(out, spec):
        assert row == APPROX(orig) or row == APPROX([0.0] * 6)


def test_a_freq_mask_zeroes_the_same_bin_in_every_frame():
    """Маска по частоте — вертикальная полоса: один бин выбит во всех кадрах."""
    spec = _grid(10, 6)
    out = spec_augment(spec, random.Random(5), n_time_masks=0, n_freq_masks=2,
                       freq_width=3)
    for j in range(6):
        column = [row[j] for row in out]
        original = [row[j] for row in spec]
        assert column == APPROX(original) or column == APPROX([0.0] * 10)


# -------------------------------------------------------- confusion_matrix
def test_confusion_matrix_counts_true_rows_by_predicted_columns():
    assert confusion_matrix([0, 0, 1], [0, 1, 1], 2) == [[1, 1], [0, 1]]


def test_perfect_prediction_fills_only_the_diagonal():
    cm = confusion_matrix([0, 1, 2, 2], [0, 1, 2, 2], 3)
    assert flat(cm) == [1, 0, 0, 0, 1, 0, 0, 0, 2]


def test_row_sums_are_the_true_class_counts():
    """Строка — истина: её сумма не зависит от того, что предсказала модель."""
    cm = confusion_matrix([0, 0, 0, 1], [1, 1, 0, 0], 2)
    assert [sum(row) for row in cm] == [3, 1]


# ------------------------------------------------------------------ macro_f1
def test_macro_f1_of_a_perfect_classifier_is_one():
    assert macro_f1([[5, 0], [0, 3]]) == APPROX(1.0)


def test_macro_f1_of_a_fully_swapped_classifier_is_zero():
    assert macro_f1([[0, 4], [4, 0]]) == APPROX(0.0)


def test_macro_f1_collapses_when_a_rare_class_is_never_predicted():
    """Ради этого macro-F1 и берут: accuracy 0.98 прячет невыученный класс.

    98 клипов класса 0 угаданы, 2 клипа класса 1 — нет.
    """
    cm = [[98, 0], [2, 0]]
    accuracy = 98 / 100
    assert accuracy == APPROX(0.98)
    assert macro_f1(cm) < 0.55


def test_macro_f1_does_not_care_which_class_is_called_zero():
    """Классы усредняются с равным весом, поэтому их можно переименовать."""
    cm = [[7, 1], [2, 5]]
    swapped = [[5, 2], [1, 7]]
    assert macro_f1(swapped) == APPROX(macro_f1(cm))


# --------------------------------------------------------- average_precision
def test_average_precision_of_a_perfect_ranking_is_one():
    assert average_precision([0.9, 0.8, 0.1], [1, 1, 0]) == APPROX(1.0)


def test_average_precision_averages_precision_at_each_hit():
    assert average_precision([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx(
        (1.0 + 2 / 3) / 2, abs=1e-12
    )


def test_average_precision_depends_only_on_the_order_of_scores():
    """Ловушка: значения не участвуют в формуле, только их порядок.

    Прогоняем scores через строго возрастающее преобразование — ответ обязан
    остаться прежним.
    """
    scores = [0.9, 0.2, 0.55, 0.31]
    labels = [1, 0, 1, 0]
    rescaled = [math.exp(3 * s) + 100 for s in scores]
    assert average_precision(rescaled, labels) == APPROX(
        average_precision(scores, labels)
    )


def test_positives_at_the_bottom_score_worse_than_at_the_top():
    labels = [1, 1, 0, 0]
    good = average_precision([0.9, 0.8, 0.2, 0.1], labels)
    bad = average_precision([0.2, 0.1, 0.9, 0.8], labels)
    assert bad < good


def test_average_precision_without_positives_is_zero():
    """Делить на ноль положительных нельзя — метрика обязана вернуть 0.0."""
    assert average_precision([0.5, 0.9], [0, 0]) == APPROX(0.0)
