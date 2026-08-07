"""Тесты к уроку «Instance-сегментация и Mask R-CNN». Правь exercise.py."""

import pytest

from exercise import (
    bilinear_sample,
    box_iou,
    decode_box_delta,
    mask_iou,
    nms,
    paste_mask,
    roi_align,
    roi_pool,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(grid):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    out = []
    for row in grid:
        out.extend(row)
    return out


RAMP = [[0.0, 1.0, 2.0, 3.0] for _ in range(4)]  # значение равно номеру столбца


# ---------------------------------------------------------------- box_iou
def test_iou_of_identical_boxes_is_one():
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == APPROX(1.0)


def test_iou_of_touching_boxes_is_zero():
    """Общий угол — это не пересечение."""
    assert box_iou((0, 0, 2, 2), (2, 2, 4, 4)) == APPROX(0.0)


def test_iou_of_separated_boxes_is_zero_not_positive():
    """Ловушка: два отрицательных перекрытия перемножаются в плюс."""
    assert box_iou((0, 0, 1, 1), (10, 10, 11, 11)) == APPROX(0.0)


def test_iou_of_quarter_overlap():
    assert box_iou((0, 0, 4, 4), (2, 2, 6, 6)) == pytest.approx(4 / 28)


# -------------------------------------------------------------------- nms
def test_nms_keeps_the_higher_score_of_a_duplicate_pair():
    assert nms([(0, 0, 2, 2), (0, 0, 2, 2)], [0.9, 0.1], 0.5) == [0]


def test_nms_keeps_boxes_that_do_not_overlap():
    assert sorted(nms([(0, 0, 2, 2), (5, 5, 7, 7)], [0.1, 0.9], 0.5)) == [0, 1]


def test_nms_returns_indices_sorted_by_score():
    """Порядок ответа — по убыванию score, а не по исходным индексам."""
    assert nms([(0, 0, 2, 2), (5, 5, 7, 7)], [0.1, 0.9], 0.5) == [1, 0]


def test_nms_threshold_zero_keeps_only_non_overlapping():
    boxes = [(0, 0, 4, 4), (1, 1, 5, 5), (20, 20, 24, 24)]
    assert sorted(nms(boxes, [0.9, 0.8, 0.7], 0.0)) == [0, 2]


# ------------------------------------------------------- decode_box_delta
def test_zero_delta_returns_the_anchor_unchanged():
    assert decode_box_delta((0, 0, 10, 10), (0, 0, 0, 0)) == pytest.approx((0.0, 0.0, 10.0, 10.0))


def test_delta_shift_is_measured_in_anchor_widths():
    """dx = 0.1 сдвигает на 0.1 ширины якоря, а не на 0.1 пикселя."""
    assert decode_box_delta((0, 0, 10, 10), (0.1, 0, 0, 0)) == pytest.approx((1.0, 0.0, 11.0, 10.0))


def test_delta_scale_goes_through_exp():
    x1, y1, x2, y2 = decode_box_delta((0, 0, 10, 10), (0, 0, 0.6931471805599453, 0))
    assert (x2 - x1) == pytest.approx(20.0)


def test_decoded_width_is_always_positive():
    """exp физически не даёт отрицательную ширину, каким бы диким dw ни был."""
    for dw in (-20.0, -5.0, 0.0, 5.0):
        x1, _, x2, _ = decode_box_delta((0, 0, 10, 10), (0, 0, dw, 0))
        assert x2 > x1


def test_decoding_keeps_the_centre_when_only_size_changes():
    x1, y1, x2, y2 = decode_box_delta((0, 0, 10, 10), (0, 0, 1.0, 1.0))
    assert (x1 + x2) / 2 == pytest.approx(5.0)
    assert (y1 + y2) / 2 == pytest.approx(5.0)


# -------------------------------------------------------- bilinear_sample
def test_sampling_at_integer_coordinates_returns_the_pixel():
    f = [[0.0, 1.0], [2.0, 3.0]]
    assert bilinear_sample(f, 1, 0) == APPROX(2.0)


def test_sampling_between_two_pixels_is_their_average():
    f = [[0.0, 1.0], [2.0, 3.0]]
    assert bilinear_sample(f, 0.0, 0.5) == APPROX(0.5)


def test_sampling_at_the_centre_averages_all_four():
    f = [[0.0, 1.0], [2.0, 3.0]]
    assert bilinear_sample(f, 0.5, 0.5) == APPROX(1.5)


def test_sampling_outside_clamps_to_the_border():
    f = [[0.0, 1.0], [2.0, 3.0]]
    assert bilinear_sample(f, -5, -5) == APPROX(0.0)
    assert bilinear_sample(f, 99, 99) == APPROX(3.0)


def test_sampling_arguments_are_y_then_x():
    """Ловушка: (y, x) как индексация, а не (x, y) как координаты бокса."""
    f = [[0.0, 10.0], [20.0, 30.0]]
    assert bilinear_sample(f, 1, 0) == APPROX(20.0)
    assert bilinear_sample(f, 0, 1) == APPROX(10.0)


# -------------------------------------------------------------- roi_align
def test_roi_align_output_is_a_square_grid():
    out = roi_align(RAMP, (0, 0, 4, 4), 3, 1.0)
    assert len(out) == 3 and all(len(row) == 3 for row in out)


def test_roi_align_on_a_ramp_hits_bin_centres():
    out = roi_align(RAMP, (0, 0, 4, 4), 2, 1.0)
    assert flat(out) == pytest.approx([0.5, 2.5, 0.5, 2.5])


def test_roi_align_of_a_constant_map_is_constant():
    const = [[5.0] * 4 for _ in range(4)]
    assert flat(roi_align(const, (1.3, 0.7, 3.9, 2.2), 3, 1.0)) == pytest.approx([5.0] * 9)


def test_roi_align_reacts_to_a_sub_pixel_shift():
    """В этом весь смысл: сдвиг бокса на 0.1 пикселя меняет ответ."""
    a = flat(roi_align(RAMP, (0.0, 0.0, 4.0, 4.0), 2, 1.0))
    b = flat(roi_align(RAMP, (0.1, 0.0, 4.1, 4.0), 2, 1.0))
    assert a != pytest.approx(b, abs=1e-6)


def test_roi_align_respects_spatial_scale():
    """Бокс в пикселях кадра и та же область на карте stride 2 — один ответ."""
    a = flat(roi_align(RAMP, (0, 0, 8, 8), 2, 0.5))
    b = flat(roi_align(RAMP, (0, 0, 4, 4), 2, 1.0))
    assert a == pytest.approx(b)


# --------------------------------------------------------------- roi_pool
def test_roi_pool_takes_the_maximum_of_each_bin():
    assert flat(roi_pool(RAMP, (0, 0, 4, 4), 2, 1.0)) == pytest.approx([1.0, 3.0, 1.0, 3.0])


def test_roi_pool_ignores_a_sub_pixel_shift():
    """Округление съедает дробную часть — ровно та ошибка, которую убрал RoIAlign."""
    a = flat(roi_pool(RAMP, (0.0, 0.0, 4.0, 4.0), 2, 1.0))
    b = flat(roi_pool(RAMP, (0.1, 0.0, 4.1, 4.0), 2, 1.0))
    assert a == pytest.approx(b)


def test_roi_pool_is_biased_upwards_against_roi_align():
    """Max по ячейке всегда >= билинейного значения в её центре."""
    pooled = flat(roi_pool(RAMP, (0, 0, 4, 4), 2, 1.0))
    aligned = flat(roi_align(RAMP, (0, 0, 4, 4), 2, 1.0))
    assert all(p >= a for p, a in zip(pooled, aligned))
    assert pooled != pytest.approx(aligned)


# ------------------------------------------------------------- paste_mask
def test_paste_mask_fills_the_whole_box():
    assert paste_mask([[1.0]], (0, 0, 2, 2), 2, 2) == [[1, 1], [1, 1]]


def test_paste_mask_stays_inside_a_small_box():
    assert paste_mask([[1.0]], (0, 0, 1, 1), 2, 2) == [[1, 0], [0, 0]]


def test_paste_mask_thresholds_low_probabilities_away():
    assert paste_mask([[0.2]], (0, 0, 2, 2), 2, 2) == [[0, 0], [0, 0]]


def test_paste_mask_upsamples_a_two_by_two_mask():
    """Левая половина маски горит — горит левая половина бокса."""
    mask = [[1.0, 0.0], [1.0, 0.0]]
    out = paste_mask(mask, (0, 0, 4, 4), 4, 4)
    assert all(row[0] == 1 and row[3] == 0 for row in out)


def test_paste_mask_leaves_everything_outside_the_box_at_zero():
    out = paste_mask([[1.0]], (1, 1, 3, 3), 4, 4)
    assert out[0] == [0, 0, 0, 0]
    assert all(row[0] == 0 for row in out)


# --------------------------------------------------------------- mask_iou
def test_mask_iou_of_identical_masks_is_one():
    assert mask_iou([[1, 1], [0, 0]], [[1, 1], [0, 0]]) == APPROX(1.0)


def test_mask_iou_of_half_overlap():
    assert mask_iou([[1, 1], [0, 0]], [[1, 0], [0, 0]]) == APPROX(0.5)


def test_mask_iou_of_two_empty_masks_is_zero():
    assert mask_iou([[0, 0]], [[0, 0]]) == APPROX(0.0)


def test_mask_iou_can_be_low_while_box_iou_is_perfect():
    """Бокс идеальный, силуэт мимо — вот почему mask AP считают отдельно."""
    a = [[1, 0], [0, 1]]
    b = [[0, 1], [1, 0]]
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == APPROX(1.0)
    assert mask_iou(a, b) == APPROX(0.0)
