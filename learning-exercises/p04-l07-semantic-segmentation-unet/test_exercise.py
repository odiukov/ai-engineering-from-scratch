"""Тесты к уроку «Семантическая сегментация и U-Net». Правь exercise.py."""

import math

import pytest

from exercise import (
    combined_loss,
    dice_coefficient,
    dice_loss,
    iou_per_class,
    mean_iou,
    pixel_accuracy,
    pixel_cross_entropy,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([2.0, 1.0, 0.0, -3.0])) == APPROX(1.0)


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([7.0, 7.0, 7.0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_shift_invariant():
    """Прибавить константу ко всем логитам — ответ не меняется."""
    a = softmax([2.0, 1.0, 0.0])
    b = softmax([102.0, 101.0, 100.0])
    assert a == pytest.approx(b)


def test_softmax_survives_huge_logits():
    """Наивный math.exp(1000) падает с OverflowError."""
    assert softmax([1000.0, 0.0]) == pytest.approx([1.0, 0.0])


def test_softmax_preserves_order():
    p = softmax([0.5, 3.0, -1.0])
    assert p[1] > p[0] > p[2]


# --------------------------------------------------------- pixel_accuracy
def test_pixel_accuracy_perfect():
    assert pixel_accuracy([[0, 1], [1, 1]], [[0, 1], [1, 1]]) == APPROX(1.0)


def test_pixel_accuracy_counts_every_pixel():
    assert pixel_accuracy([[0, 0], [0, 0]], [[0, 0], [0, 1]]) == APPROX(0.75)


def test_pixel_accuracy_is_fooled_by_class_imbalance():
    """99 фоновых пикселей и один объект: «всё фон» даёт 0.99 и нулевую пользу."""
    preds = [[0] * 100]
    targets = [[0] * 99 + [1]]
    assert pixel_accuracy(preds, targets) == APPROX(0.99)


# ---------------------------------------------------- pixel_cross_entropy
def test_cross_entropy_of_uniform_logits_is_log_two():
    assert pixel_cross_entropy([[[0.0, 0.0]]], [[0]]) == pytest.approx(math.log(2))


def test_cross_entropy_of_confident_hit_is_zero():
    assert pixel_cross_entropy([[[100.0, 0.0]]], [[0]]) == pytest.approx(0.0, abs=1e-9)


def test_cross_entropy_punishes_confident_miss():
    """Уверенная ошибка стоит дороже неуверенной."""
    sure_miss = pixel_cross_entropy([[[0.0, 20.0]]], [[0]])
    unsure = pixel_cross_entropy([[[0.0, 1.0]]], [[0]])
    assert sure_miss > unsure > 0


def test_cross_entropy_averages_over_pixels():
    """Один идеальный пиксель и один неуверенный дают ровно половину."""
    logits = [[[100.0, 0.0], [0.0, 0.0]]]
    assert pixel_cross_entropy(logits, [[0, 0]]) == pytest.approx(math.log(2) / 2)


def test_cross_entropy_does_not_blow_up_on_hopeless_pixel():
    """log(0) обязан быть подрезан, а не выдать -inf или ValueError."""
    assert math.isfinite(pixel_cross_entropy([[[0.0, 10000.0]]], [[0]]))


# ------------------------------------------------------ dice_coefficient
def test_dice_of_identical_masks_is_one():
    assert dice_coefficient([[1, 1], [0, 0]], [[1, 1], [0, 0]]) == pytest.approx(1.0, abs=1e-5)


def test_dice_of_disjoint_masks_is_zero():
    assert dice_coefficient([[1, 0], [0, 0]], [[0, 1], [0, 0]]) == pytest.approx(0.0, abs=1e-5)


def test_dice_of_half_overlap():
    """|A|=2, |B|=1, пересечение 1  ->  2*1/3."""
    assert dice_coefficient([[1, 1], [0, 0]], [[1, 0], [0, 0]]) == pytest.approx(2 / 3, abs=1e-5)


def test_dice_of_two_empty_masks_is_one():
    """Обе пустые — деление 0/0, спасает eps: класса нет и не предсказан."""
    assert dice_coefficient([[0, 0]], [[0, 0]]) == pytest.approx(1.0, abs=1e-5)


def test_dice_is_symmetric():
    a = [[1, 1, 0], [0, 1, 0]]
    b = [[1, 0, 0], [0, 1, 1]]
    assert dice_coefficient(a, b) == pytest.approx(dice_coefficient(b, a))


def test_dice_matches_the_iou_relation():
    """Dice = 2*IoU/(1+IoU) — метрики монотонно связаны, а не независимы."""
    a = [[1, 1, 0], [0, 1, 0]]
    b = [[1, 0, 0], [0, 1, 1]]
    iou = iou_per_class(a, b, 2)[1]
    assert dice_coefficient(a, b) == pytest.approx(2 * iou / (1 + iou), abs=1e-5)


# -------------------------------------------------------------- dice_loss
def test_dice_loss_of_perfect_prediction_is_zero():
    logits = [[[100.0, 0.0], [0.0, 100.0]]]
    assert dice_loss(logits, [[0, 1]], 2) == pytest.approx(0.0, abs=1e-5)


def test_dice_loss_of_total_uncertainty_is_half():
    logits = [[[0.0, 0.0], [0.0, 0.0]]]
    assert dice_loss(logits, [[0, 1]], 2) == pytest.approx(0.5, abs=1e-5)


def test_dice_loss_stays_large_when_the_rare_class_is_missed():
    """Модель «везде фон» набирает accuracy 0.99, но Dice-лосс её не прощает."""
    logits = [[[10.0, 0.0]] * 100]
    targets = [[0] * 99 + [1]]
    assert pixel_accuracy([[0] * 100], targets) == pytest.approx(0.99)
    assert dice_loss(logits, targets, 2) > 0.45


def test_dice_loss_is_lower_for_better_overlap():
    good = [[[5.0, 0.0], [0.0, 5.0]]]
    bad = [[[0.0, 5.0], [5.0, 0.0]]]
    assert dice_loss(good, [[0, 1]], 2) < dice_loss(bad, [[0, 1]], 2)


# ---------------------------------------------------------- combined_loss
def test_combined_loss_is_ce_plus_lambda_dice():
    logits = [[[1.0, -1.0], [0.0, 2.0]]]
    targets = [[0, 1]]
    total, parts = combined_loss(logits, targets, 2, lam=1.0)
    assert total == pytest.approx(parts["ce"] + parts["dice"])


def test_combined_loss_reports_both_parts():
    logits = [[[1.0, -1.0], [0.0, 2.0]]]
    total, parts = combined_loss(logits, [[0, 1]], 2)
    assert parts["ce"] == pytest.approx(pixel_cross_entropy(logits, [[0, 1]]))
    assert parts["dice"] == pytest.approx(dice_loss(logits, [[0, 1]], 2))


def test_combined_loss_with_zero_lambda_is_pure_cross_entropy():
    logits = [[[1.0, -1.0], [0.0, 2.0]]]
    total, parts = combined_loss(logits, [[0, 1]], 2, lam=0.0)
    assert total == pytest.approx(parts["ce"])


def test_combined_loss_lambda_scales_the_dice_term():
    logits = [[[1.0, -1.0], [0.0, 2.0]]]
    t1, _ = combined_loss(logits, [[0, 1]], 2, lam=1.0)
    t2, parts = combined_loss(logits, [[0, 1]], 2, lam=2.0)
    assert t2 - t1 == pytest.approx(parts["dice"])


# --------------------------------------------------------- iou_per_class
def test_iou_of_perfect_prediction_is_one_everywhere():
    assert iou_per_class([[0, 1]], [[0, 1]], 2) == pytest.approx([1.0, 1.0])


def test_iou_of_missed_class_is_zero():
    assert iou_per_class([[0, 0]], [[0, 1]], 2) == pytest.approx([0.5, 0.0])


def test_iou_of_absent_class_is_none_not_zero():
    """Классa нет ни в истине, ни в предсказании — метрика не определена."""
    assert iou_per_class([[0, 0]], [[0, 0]], 3) == [1.0, None, None]


def test_iou_is_symmetric_in_its_arguments():
    a = [[0, 1, 1], [2, 1, 0]]
    b = [[0, 1, 2], [2, 0, 0]]
    assert iou_per_class(a, b, 3) == pytest.approx(iou_per_class(b, a, 3))


def test_iou_never_exceeds_one():
    a = [[0, 1, 1], [2, 1, 0]]
    b = [[0, 1, 2], [2, 0, 0]]
    assert all(v <= 1.0 for v in iou_per_class(a, b, 3) if v is not None)


# -------------------------------------------------------------- mean_iou
def test_mean_iou_skips_absent_classes():
    assert mean_iou([1.0, 0.5, None]) == APPROX(0.75)


def test_mean_iou_of_all_absent_is_none():
    assert mean_iou([None, None]) is None


def test_mean_iou_hides_a_failing_class():
    """Девять классов по 0.85 и один по 0.15 всё ещё дают приличные 0.78."""
    assert mean_iou([0.85] * 9 + [0.15]) == pytest.approx(0.78, abs=1e-9)
