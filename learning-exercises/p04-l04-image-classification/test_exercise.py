"""Тесты к уроку «Классификация изображений». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    class_report,
    confusion_matrix,
    cross_entropy,
    mixup_batch,
    one_hot,
    softmax,
    soft_cross_entropy,
    top_k_accuracy,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


# ----------------------------------------------------------------- softmax
def test_equal_logits_give_a_uniform_distribution():
    assert softmax([0.0, 0.0]) == APPROX([0.5, 0.5])
    assert softmax([3.0, 3.0, 3.0, 3.0]) == APPROX([0.25] * 4)


def test_probabilities_sum_to_one():
    assert sum(softmax([2.0, -1.0, 0.5, 7.0])) == APPROX(1.0)


def test_order_of_logits_is_preserved():
    """softmax монотонен: самый большой логит даёт самую большую вероятность."""
    p = softmax([1.0, 3.0, 2.0])
    assert p[1] > p[2] > p[0]


def test_huge_logits_do_not_overflow():
    """Наивный exp(1000) — OverflowError, хотя ответ определён идеально."""
    assert softmax([1000.0, 1000.0, 1000.0]) == pytest.approx([1 / 3] * 3, abs=1e-12)


# ----------------------------------------------------------- cross_entropy
def test_uniform_logits_cost_log_of_num_classes():
    assert cross_entropy([0.0, 0.0, 0.0], 1) == APPROX(math.log(3))


def test_confident_and_correct_costs_almost_nothing():
    assert cross_entropy([10.0, 0.0], 0) == pytest.approx(0.0, abs=1e-4)


def test_confident_and_wrong_costs_a_lot():
    assert cross_entropy([10.0, 0.0], 1) > 9.0


def test_loss_matches_minus_log_of_the_softmax_probability():
    logits = [1.5, -0.5, 2.0]
    assert cross_entropy(logits, 2) == APPROX(-math.log(softmax(logits)[2]))


def test_extreme_logits_stay_finite():
    """Через log(softmax(...)) здесь получился бы inf: вероятность легла в ноль."""
    loss = cross_entropy([0.0, 800.0], 0)
    assert math.isfinite(loss) and loss > 700.0


# ---------------------------------------------------------------- one_hot
def test_hard_target_puts_all_mass_on_the_class():
    assert one_hot(1, 4) == APPROX([0.0, 1.0, 0.0, 0.0])


def test_smoothing_leaves_a_crumb_on_every_class():
    assert one_hot(1, 4, smoothing=0.2) == APPROX([0.05, 0.85, 0.05, 0.05])


def test_smoothed_target_still_sums_to_one():
    for eps in (0.0, 0.05, 0.1, 0.9):
        assert sum(one_hot(2, 5, smoothing=eps)) == APPROX(1.0)


def test_full_smoothing_erases_the_label():
    """eps = 1 — цель равномерная, учить уже нечему. Полезный крайний случай."""
    assert one_hot(0, 4, smoothing=1.0) == APPROX([0.25] * 4)


# ------------------------------------------------------ soft_cross_entropy
def test_soft_loss_on_a_hard_target_equals_plain_cross_entropy():
    """Обобщение обязано совпадать с частным случаем — иначе где-то множитель."""
    logits = [1.0, -2.0, 0.5, 3.0]
    for target in range(4):
        assert soft_cross_entropy(logits, one_hot(target, 4)) == APPROX(
            cross_entropy(logits, target)
        )


def test_uniform_target_on_uniform_logits_costs_log_of_num_classes():
    assert soft_cross_entropy([0.0, 0.0], [0.5, 0.5]) == APPROX(math.log(2))


def test_soft_loss_is_minimal_when_predictions_match_the_target():
    """Ниже энтропии самой цели loss не опускается — это её нижняя граница."""
    target = [0.7, 0.3]
    entropy = -sum(p * math.log(p) for p in target)
    matched = [math.log(p) for p in target]
    assert soft_cross_entropy(matched, target) == APPROX(entropy)
    assert soft_cross_entropy([0.0, 0.0], target) > entropy


def test_smoothed_target_costs_more_than_a_hard_one_on_a_confident_model():
    """Плата за сглаживание: уверенная и верная модель получает штраф."""
    logits = [8.0, 0.0, 0.0]
    hard = soft_cross_entropy(logits, one_hot(0, 3))
    soft = soft_cross_entropy(logits, one_hot(0, 3, smoothing=0.1))
    assert soft > hard


# ------------------------------------------------------------ mixup_batch
def test_lambda_one_returns_the_batch_untouched():
    images = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    x, y = mixup_batch(images, [0, 1, 1], 2, 1.0, random.Random(0))
    assert flat(x) == APPROX(flat(images))
    assert flat(y) == APPROX(flat([one_hot(c, 2) for c in [0, 1, 1]]))


def test_mixed_labels_still_sum_to_one():
    rng = random.Random(3)
    _, y = mixup_batch([[0.0]] * 6, [0, 1, 2, 0, 1, 2], 3, 0.3, rng)
    for dist in y:
        assert sum(dist) == APPROX(1.0)


def test_mixing_two_identical_labels_leaves_a_hard_target():
    """Если партнёр того же класса, смешивать нечего — цель остаётся one-hot."""
    _, y = mixup_batch([[1.0], [2.0], [3.0]], [1, 1, 1], 3, 0.4, random.Random(1))
    for dist in y:
        assert dist == APPROX(one_hot(1, 3))


def test_mixed_pixels_stay_inside_the_range_of_the_batch():
    """lam из [0,1] даёт выпуклую комбинацию: новых экстремумов не появляется."""
    images = [[0.0, 0.2], [1.0, 0.8], [0.5, 0.5], [0.25, 0.75]]
    x, _ = mixup_batch(images, [0, 1, 0, 1], 2, 0.7, random.Random(9))
    values = [v for row in x for v in row]
    assert min(values) >= 0.0 - 1e-12
    assert max(values) <= 1.0 + 1e-12


def test_same_seed_gives_the_same_batch():
    """Случайность идёт только через rng — прогон обязан повторяться."""
    images = [[float(i)] for i in range(8)]
    labels = [i % 2 for i in range(8)]
    a_x, a_y = mixup_batch(images, labels, 2, 0.6, random.Random(42))
    b_x, b_y = mixup_batch(images, labels, 2, 0.6, random.Random(42))
    assert flat(a_x) == APPROX(flat(b_x))
    assert flat(a_y) == APPROX(flat(b_y))


def test_mixup_does_not_mutate_the_input_images():
    images = [[0.0, 1.0], [1.0, 0.0]]
    mixup_batch(images, [0, 1], 2, 0.5, random.Random(0))
    assert flat(images) == APPROX([0.0, 1.0, 1.0, 0.0])


# ------------------------------------------------------- confusion_matrix
def test_counts_land_in_row_true_column_predicted():
    assert confusion_matrix([0, 1, 1], [0, 1, 0], 2) == [[1, 0], [1, 1]]


def test_perfect_predictions_fill_only_the_diagonal():
    cm = confusion_matrix([0, 1, 2, 2], [0, 1, 2, 2], 3)
    assert flat(cm) == [1, 0, 0, 0, 1, 0, 0, 0, 2]


def test_row_sums_are_the_true_class_counts():
    true = [0, 0, 0, 1, 2, 2]
    cm = confusion_matrix(true, [1, 2, 0, 1, 0, 2], 3)
    assert [sum(row) for row in cm] == [3, 1, 2]


# ----------------------------------------------------------- class_report
def test_report_on_the_worked_example():
    row = class_report([[1, 0], [1, 1]])[0]
    assert row["precision"] == APPROX(0.5)
    assert row["recall"] == APPROX(1.0)
    assert row["f1"] == APPROX(2 / 3)


def test_perfect_matrix_scores_one_everywhere():
    for row in class_report([[5, 0], [0, 3]]):
        assert row["precision"] == APPROX(1.0)
        assert row["recall"] == APPROX(1.0)
        assert row["f1"] == APPROX(1.0)


def test_majority_classifier_hides_a_zero_recall_behind_high_accuracy():
    """90 объектов класса 0, 10 класса 1, модель всегда говорит 0."""
    cm = confusion_matrix([0] * 90 + [1] * 10, [0] * 100, 2)
    report = class_report(cm)
    assert sum(cm[i][i] for i in range(2)) / 100 == APPROX(0.9)
    assert report[1]["recall"] == APPROX(0.0)
    assert report[1]["f1"] == APPROX(0.0)


def test_transposing_the_matrix_swaps_precision_and_recall():
    """Ловушка «строки это истина или предсказание»: отчёт врёт молча."""
    cm = [[4, 2, 0], [1, 3, 1], [0, 0, 5]]
    transposed = [[cm[j][i] for j in range(3)] for i in range(3)]
    direct = class_report(cm)
    swapped = class_report(transposed)
    for i in range(3):
        assert swapped[i]["precision"] == APPROX(direct[i]["recall"])
        assert swapped[i]["recall"] == APPROX(direct[i]["precision"])


def test_class_never_predicted_gets_zero_instead_of_a_crash():
    report = class_report([[2, 0], [3, 0]])
    assert report[1]["precision"] == APPROX(0.0)
    assert report[1]["recall"] == APPROX(0.0)


# --------------------------------------------------------- top_k_accuracy
def test_top1_counts_only_the_argmax():
    assert top_k_accuracy([[0.1, 0.9], [0.8, 0.2]], [1, 1]) == APPROX(0.5)


def test_second_guess_is_enough_for_top2():
    assert top_k_accuracy([[0.1, 0.9], [0.8, 0.2]], [1, 1], k=2) == APPROX(1.0)


def test_k_equal_to_num_classes_is_always_perfect():
    logits = [[3.0, -1.0, 0.0], [0.0, 0.0, 0.0], [-5.0, 2.0, 1.0]]
    assert top_k_accuracy(logits, [2, 0, 0], k=3) == APPROX(1.0)


def test_accuracy_never_drops_as_k_grows():
    logits = [[0.2, 0.5, 0.3], [1.0, 0.9, 0.8], [0.0, 0.1, 2.0]]
    targets = [2, 2, 1]
    scores = [top_k_accuracy(logits, targets, k=k) for k in (1, 2, 3)]
    assert scores == sorted(scores)
