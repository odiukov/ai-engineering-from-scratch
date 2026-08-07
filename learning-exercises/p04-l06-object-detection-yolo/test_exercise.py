"""Тесты к уроку «Детекция объектов — YOLO». Правь exercise.py."""

import math

import pytest

from exercise import (
    best_anchor,
    decode_box,
    encode_box,
    iou,
    nms,
    precision_recall,
    sigmoid,
    yolo_loss,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ANCHORS = [(30, 60), (75, 170), (200, 380)]


# ----------------------------------------------------------------- sigmoid
def test_zero_maps_to_one_half():
    assert sigmoid(0.0) == APPROX(0.5)


def test_sigmoid_is_symmetric_around_zero():
    for x in (0.3, 2.0, 7.5):
        assert sigmoid(-x) == APPROX(1.0 - sigmoid(x))


def test_output_always_stays_inside_the_unit_interval():
    """Это и есть гарантия «центр не уедет из своей ячейки»."""
    for x in (-30.0, -1.0, 0.0, 1.0, 30.0):
        assert 0.0 < sigmoid(x) < 1.0
    for x in (-1e9, 1e9):  # на краях float честно упирается в границы
        assert 0.0 <= sigmoid(x) <= 1.0


def test_huge_negative_argument_does_not_overflow():
    """Наивная 1/(1+exp(-x)) на x=-1000 бросает OverflowError."""
    assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)
    assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------- iou
def test_identical_boxes_overlap_completely():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == APPROX(1.0)


def test_boxes_touching_by_an_edge_still_score_zero():
    """Ловушка знака: два отрицательных размера перемножатся в плюс."""
    assert iou((0, 0, 2, 2), (2, 0, 4, 2)) == APPROX(0.0)
    assert iou((0, 0, 2, 2), (3, 3, 5, 5)) == APPROX(0.0)


def test_partial_overlap_matches_the_hand_computation():
    assert iou((0, 0, 2, 2), (1, 1, 3, 3)) == APPROX(1 / 7)


def test_iou_is_symmetric():
    a, b = (0, 0, 4, 3), (2, 1, 8, 9)
    assert iou(a, b) == APPROX(iou(b, a))


def test_degenerate_box_does_not_divide_by_zero():
    assert iou((1, 1, 1, 1), (0, 0, 2, 2)) == APPROX(0.0)


# --------------------------------------------------------------------- nms
def test_duplicate_of_the_top_box_is_dropped():
    boxes = [(0, 0, 2, 2), (0, 0, 2, 2), (9, 9, 10, 10)]
    assert nms(boxes, [0.9, 0.8, 0.7]) == [0, 2]


def test_survivors_never_overlap_above_the_threshold():
    """Главное свойство NMS: среди оставшихся нет пары выше порога."""
    boxes = [
        (0, 0, 10, 10),
        (1, 1, 11, 11),
        (2, 2, 12, 12),
        (30, 30, 40, 40),
        (31, 30, 41, 40),
    ]
    scores = [0.9, 0.85, 0.8, 0.7, 0.6]
    keep = nms(boxes, scores, iou_threshold=0.45)
    for i in keep:
        for j in keep:
            if i != j:
                assert iou(boxes[i], boxes[j]) <= 0.45


def test_the_highest_score_always_survives():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (0, 0, 10, 10)]
    assert nms(boxes, [0.1, 0.5, 0.9], iou_threshold=0.3)[0] == 2


def test_non_overlapping_boxes_all_survive():
    boxes = [(0, 0, 1, 1), (5, 5, 6, 6), (9, 9, 10, 10)]
    assert sorted(nms(boxes, [0.3, 0.9, 0.6])) == [0, 1, 2]


def test_a_looser_threshold_never_keeps_fewer_boxes():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (2, 2, 12, 12), (3, 3, 13, 13)]
    scores = [0.9, 0.8, 0.7, 0.6]
    counts = [len(nms(boxes, scores, t)) for t in (0.1, 0.45, 0.9, 1.0)]
    assert counts == sorted(counts)


# -------------------------------------------------------------- decode_box
def test_zero_prediction_lands_on_the_cell_centre_with_anchor_size():
    assert decode_box((0.0, 0.0, 0.0, 0.0), 3, 4, 32, (30, 60)) == APPROX(
        (97.0, 114.0, 127.0, 174.0)
    )


def test_log_two_doubles_the_anchor_width():
    """exp даёт умножение: tw = log 2 — это ровно вдвое шире анкера."""
    box = decode_box((0.0, 0.0, math.log(2), 0.0), 0, 0, 32, (30, 60))
    assert box[2] - box[0] == APPROX(60.0)
    assert box[3] - box[1] == APPROX(60.0)


def test_centre_never_leaves_its_own_cell():
    """Что бы сеть ни выдала, sigmoid запирает центр между границами ячейки."""
    for tx in (-40.0, -1.0, 0.0, 1.0, 40.0):
        box = decode_box((tx, 0.0, 0.0, 0.0), 3, 4, 32, (30, 60))
        cx = 0.5 * (box[0] + box[2])
        assert 3 * 32 <= cx <= 4 * 32


def test_width_stays_positive_for_any_regression_output():
    for tw in (-20.0, 0.0, 5.0):
        box = decode_box((0.0, 0.0, tw, tw), 1, 1, 32, (30, 60))
        assert box[2] > box[0] and box[3] > box[1]


# -------------------------------------------------------------- encode_box
def test_encoding_a_cell_centre_gives_zeros():
    assert encode_box((97, 114, 127, 174), 3, 4, 32, (30, 60)) == APPROX(
        (0.0, 0.0, 0.0, 0.0)
    )


def test_encode_then_decode_returns_the_original_box():
    """Без логита в encode эта проверка развалится — цели будут не те."""
    box = (100.0, 130.0, 148.0, 218.0)
    cell_x, cell_y, stride = 3, 5, 32
    t = encode_box(box, cell_x, cell_y, stride, (30, 60))
    assert decode_box(t, cell_x, cell_y, stride, (30, 60)) == pytest.approx(
        box, abs=1e-6
    )


def test_a_box_twice_the_anchor_encodes_as_log_two():
    _, _, tw, th = encode_box((0, 0, 60, 120), 0, 0, 32, (30, 60))
    assert (tw, th) == APPROX((math.log(2), math.log(2)))


# ------------------------------------------------------------- best_anchor
def test_exact_shape_match_wins():
    assert best_anchor((30, 60), ANCHORS) == 0
    assert best_anchor((200, 380), ANCHORS) == 2


def test_a_tall_object_picks_a_tall_anchor():
    assert best_anchor((70, 180), ANCHORS) == 1


def test_scaling_an_object_moves_it_to_a_bigger_anchor():
    assert best_anchor((28, 55), ANCHORS) == 0
    assert best_anchor((28 * 7, 55 * 7), ANCHORS) == 2


# --------------------------------------------------------------- yolo_loss
def _slots():
    """Два слота: первый с объектом, второй пустой. Классов два."""
    pred = [
        [0.1, 0.2, 0.3, 0.4, 2.0, 1.0, -1.0],
        [9.9, 9.9, 9.9, 9.9, -2.0, 5.0, 5.0],
    ]
    target = [
        [0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    return pred, target, [True, False]


def test_perfect_box_prediction_costs_nothing():
    pred, target, has_obj = _slots()
    assert yolo_loss(pred, target, has_obj)["box"] == APPROX(0.0)


def test_empty_slots_contribute_only_to_objectness():
    """Координаты и классы пустой ячейки не определены — штрафовать нечего."""
    pred, target, has_obj = _slots()
    base = yolo_loss(pred, target, has_obj)
    pred[1][0] = -1234.0  # координаты пустого слота
    pred[1][5] = 999.0  # и его классы
    changed = yolo_loss(pred, target, has_obj)
    assert changed["total"] == APPROX(base["total"])


def test_total_is_the_weighted_sum_of_the_parts():
    pred, target, has_obj = _slots()
    r = yolo_loss(pred, target, has_obj, 5.0, 1.0, 0.5, 1.0)
    assert r["total"] == APPROX(
        5.0 * r["box"] + 1.0 * r["obj_pos"] + 0.5 * r["obj_neg"] + 1.0 * r["cls"]
    )


def test_noobj_weight_scales_only_the_empty_slots():
    pred, target, has_obj = _slots()
    a = yolo_loss(pred, target, has_obj, lambda_noobj=0.5)
    b = yolo_loss(pred, target, has_obj, lambda_noobj=1.0)
    assert b["total"] - a["total"] == APPROX(0.5 * a["obj_neg"])
    assert a["obj_neg"] == APPROX(b["obj_neg"])


def test_objectness_loss_survives_extreme_logits():
    """BCE через log(sigmoid) даёт inf; устойчивая форма — конечное число."""
    pred = [[0.0, 0.0, 0.0, 0.0, -800.0], [0.0, 0.0, 0.0, 0.0, 800.0]]
    target = [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    r = yolo_loss(pred, target, [True, False])
    assert math.isfinite(r["total"])
    assert r["obj_pos"] == pytest.approx(800.0, abs=1e-6)


# --------------------------------------------------------- precision_recall
def test_a_single_perfect_prediction_scores_one_on_both():
    r = precision_recall([(0, 0, 2, 2)], [(0, 0, 2, 2)])
    assert (r["precision"], r["recall"]) == APPROX((1.0, 1.0))


def test_a_second_box_on_the_same_object_is_a_false_positive():
    """Один объект — одно попадание. Ради этого правила и существует NMS."""
    r = precision_recall([(0, 0, 2, 2), (0, 0, 2, 2)], [(0, 0, 2, 2)])
    assert r["tp"] == 1 and r["fp"] == 1
    assert (r["precision"], r["recall"]) == APPROX((0.5, 1.0))


def test_a_missed_object_costs_recall_not_precision():
    r = precision_recall([(0, 0, 2, 2)], [(0, 0, 2, 2), (50, 50, 52, 52)])
    assert r["fn"] == 1
    assert (r["precision"], r["recall"]) == APPROX((1.0, 0.5))


def test_a_loose_box_below_the_threshold_counts_as_both_errors():
    """Промах по локализации — сразу и лишний бокс, и потерянный объект."""
    r = precision_recall([(0, 0, 10, 10)], [(0, 0, 3, 3)], iou_threshold=0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 1, 1)


def test_a_silent_detector_scores_zero_instead_of_crashing():
    r = precision_recall([], [(0, 0, 2, 2)])
    assert (r["precision"], r["recall"]) == APPROX((0.0, 0.0))


def test_lowering_the_iou_threshold_never_lowers_recall():
    preds = [(0, 0, 10, 10), (20, 20, 34, 34)]
    gts = [(0, 0, 8, 8), (20, 20, 30, 30)]
    recalls = [precision_recall(preds, gts, t)["recall"] for t in (0.9, 0.5, 0.3)]
    assert recalls == sorted(recalls)
