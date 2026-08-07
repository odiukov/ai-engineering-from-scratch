"""Тесты к уроку «Ключевые точки и оценка позы». Правь exercise.py."""

import math

import pytest

from exercise import (
    argmax_coords,
    gaussian_heatmap,
    heatmaps_to_keypoints,
    mean_l2_error,
    oks,
    paf_line_integral,
    pck,
    subpixel_offset,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in rows for x in row]


# -------------------------------------------------------- gaussian_heatmap
def test_heatmap_peak_is_exactly_one_at_the_centre():
    assert gaussian_heatmap(3, 1, 1, sigma=1.0)[1][1] == APPROX(1.0)


def test_heatmap_follows_the_gaussian_formula():
    assert gaussian_heatmap(3, 1, 1, sigma=1.0)[0][1] == APPROX(math.exp(-0.5))


def test_heatmap_is_indexed_row_y_then_column_x():
    """Ловушка порядка: центр (cx=3, cy=1) лежит в heatmap[1][3], не в [3][1]."""
    hm = gaussian_heatmap(5, 3, 1, sigma=1.0)
    assert hm[1][3] == APPROX(1.0)
    assert hm[3][1] < 0.5


def test_heatmap_is_symmetric_around_its_centre():
    hm = gaussian_heatmap(7, 3, 3, sigma=2.0)
    assert flat(hm) == pytest.approx(flat([row[::-1] for row in hm]), abs=1e-12)


def test_a_wider_sigma_spreads_the_blob():
    narrow = gaussian_heatmap(9, 4, 4, sigma=1.0)
    wide = gaussian_heatmap(9, 4, 4, sigma=3.0)
    assert wide[4][0] > narrow[4][0]


# ----------------------------------------------------------- argmax_coords
def test_argmax_returns_x_then_y():
    assert argmax_coords([[0.0, 0.0], [0.0, 1.0]]) == (1, 1)


def test_argmax_finds_the_centre_of_a_gaussian_blob():
    assert argmax_coords(gaussian_heatmap(11, 7, 2, sigma=2.0)) == (7, 2)


def test_argmax_survives_negative_values():
    """Сырой выход сети — логиты, а не вероятности: минусы там норма."""
    assert argmax_coords([[-5.0, -1.0], [-9.0, -7.0]]) == (1, 0)


# --------------------------------------------------------- subpixel_offset
def test_subpixel_finds_the_vertex_of_a_parabola():
    hm = [[0.0, 0.0, 0.0], [1.0, 4.0, 3.0], [0.0, 0.0, 0.0]]
    assert subpixel_offset(hm, 1, 1) == APPROX((0.25, 0.0))


def test_subpixel_of_a_symmetric_peak_is_zero():
    assert subpixel_offset(gaussian_heatmap(9, 4, 4, sigma=2.0), 4, 4) == APPROX((0.0, 0.0))


def test_subpixel_at_the_border_has_no_neighbour_and_stays_zero():
    hm = gaussian_heatmap(5, 0, 0, sigma=2.0)
    assert subpixel_offset(hm, 0, 0) == APPROX((0.0, 0.0))


def test_subpixel_on_a_flat_plateau_does_not_divide_by_zero():
    hm = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert subpixel_offset(hm, 1, 1) == APPROX((0.0, 0.0))


def test_subpixel_beats_plain_argmax_on_an_off_grid_peak():
    """Настоящий пик в 10.3, argmax даёт 10 — поправка обязана его приблизить."""
    hm = gaussian_heatmap(21, 10.3, 10.0, sigma=2.0)
    x, y = argmax_coords(hm)
    dx, _ = subpixel_offset(hm, x, y)
    assert abs(x + dx - 10.3) < abs(x - 10.3)


# ---------------------------------------------------- heatmaps_to_keypoints
def test_keypoints_land_on_integer_centres():
    hms = [gaussian_heatmap(9, 4, 4, sigma=2.0), gaussian_heatmap(9, 1, 7, sigma=2.0)]
    assert flat(heatmaps_to_keypoints(hms)) == APPROX([4.0, 4.0, 1.0, 7.0])


def test_channel_order_is_the_keypoint_order():
    """Поза — упорядоченный набор: канал i всегда один и тот же сустав."""
    hms = [gaussian_heatmap(9, 1, 1), gaussian_heatmap(9, 6, 2), gaussian_heatmap(9, 3, 7)]
    assert [p[0] for p in heatmaps_to_keypoints(hms)] == APPROX([1.0, 6.0, 3.0])


def test_keypoints_are_subpixel_not_integer():
    hms = [gaussian_heatmap(21, 10.4, 10.6, sigma=2.0)]
    x, y = heatmaps_to_keypoints(hms)[0]
    assert abs(x - 10.4) < 0.1 and abs(y - 10.6) < 0.1


# ------------------------------------------------------------ mean_l2_error
def test_mean_error_averages_over_keypoints():
    assert mean_l2_error([(0.0, 0.0), (0.0, 0.0)], [(3.0, 4.0), (0.0, 0.0)]) == APPROX(2.5)


def test_mean_error_of_a_perfect_pose_is_zero():
    pts = [(1.0, 2.0), (3.0, 4.0)]
    assert mean_l2_error(pts, pts) == APPROX(0.0)


def test_mean_error_grows_with_the_image_scale():
    """Пиксельная ошибка не безразмерна: тот же промах на ресайзе x4 стоит вчетверо
    дороже, хотя поза не изменилась. Ровно поэтому дальше идут PCK и OKS."""
    pred, true = [(1.0, 2.0), (5.0, 1.0)], [(1.4, 2.0), (5.0, 1.9)]
    s = 4.0
    big_pred = [(s * x, s * y) for x, y in pred]
    big_true = [(s * x, s * y) for x, y in true]
    assert mean_l2_error(big_pred, big_true) == APPROX(s * mean_l2_error(pred, true))


# --------------------------------------------------------------------- pck
def test_pck_counts_a_keypoint_inside_the_radius():
    assert pck([(0.0, 0.0)], [(1.0, 0.0)], threshold=0.5, normalizer=10.0) == APPROX(1.0)


def test_pck_rejects_a_keypoint_outside_the_radius():
    assert pck([(0.0, 0.0)], [(9.0, 0.0)], threshold=0.5, normalizer=10.0) == APPROX(0.0)


def test_pck_is_a_fraction_of_all_keypoints():
    pred = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    true = [(0.0, 0.0), (0.5, 0.0), (9.0, 0.0), (9.0, 0.0)]
    assert pck(pred, true, threshold=0.2, normalizer=10.0) == APPROX(0.5)


def test_pck_does_not_change_when_the_image_is_rescaled():
    """Главное свойство: увеличили картинку и normalizer втрое — PCK тот же."""
    pred = [(1.0, 2.0), (5.0, 1.0), (0.0, 0.0)]
    true = [(1.4, 2.0), (5.0, 4.0), (0.0, 0.0)]
    s = 3.0
    big_pred = [(s * x, s * y) for x, y in pred]
    big_true = [(s * x, s * y) for x, y in true]
    assert pck(big_pred, big_true, 0.2, s * 10.0) == APPROX(pck(pred, true, 0.2, 10.0))


def test_pck_never_decreases_as_the_threshold_grows():
    pred = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    true = [(1.0, 0.0), (4.0, 0.0), (8.0, 0.0)]
    values = [pck(pred, true, t, 10.0) for t in (0.05, 0.2, 0.5, 1.0)]
    assert values == sorted(values)


# --------------------------------------------------------------------- oks
def test_oks_of_a_perfect_pose_is_one():
    assert oks([(3.0, 4.0)], [(3.0, 4.0)], scale=10.0) == APPROX(1.0)


def test_oks_falls_off_with_the_error():
    close = oks([(0.0, 0.0)], [(0.3, 0.0)], scale=10.0)
    far = oks([(0.0, 0.0)], [(1.0, 0.0)], scale=10.0)
    assert 0.0 < far < close < 1.0


def test_oks_does_not_change_when_the_image_is_rescaled():
    pred, true = [(1.0, 2.0), (5.0, 1.0)], [(1.4, 2.0), (5.0, 1.9)]
    s = 4.0
    big_pred = [(s * x, s * y) for x, y in pred]
    big_true = [(s * x, s * y) for x, y in true]
    assert oks(big_pred, big_true, scale=s * 10.0) == pytest.approx(
        oks(pred, true, scale=10.0), abs=1e-12
    )


def test_a_smaller_kappa_is_a_stricter_joint():
    err = ([(0.0, 0.0)], [(0.5, 0.0)])
    strict = oks(*err, scale=10.0, kappas=[0.02])
    loose = oks(*err, scale=10.0, kappas=[0.10])
    assert strict < loose


def test_oks_stays_inside_zero_and_one():
    values = [oks([(0.0, 0.0)], [(d, 0.0)], scale=10.0) for d in (0.0, 0.5, 2.0, 50.0)]
    assert all(0.0 <= v <= 1.0 for v in values)


# --------------------------------------------------------- paf_line_integral
def test_paf_aligned_with_the_limb_integrates_to_one():
    paf = [[(1.0, 0.0)] * 4]
    assert paf_line_integral(paf, (0, 0), (3, 0)) == APPROX(1.0)


def test_paf_against_the_limb_integrates_to_minus_one():
    """Направление конечности несёт смысл: плечо -> локоть, а не наоборот."""
    paf = [[(1.0, 0.0)] * 4]
    assert paf_line_integral(paf, (3, 0), (0, 0)) == APPROX(-1.0)


def test_paf_perpendicular_to_the_limb_integrates_to_zero():
    paf = [[(0.0, 1.0)] * 4]
    assert paf_line_integral(paf, (0, 0), (3, 0)) == APPROX(0.0)


def test_paf_prefers_the_pair_whose_field_points_at_it():
    """Так и работает жадное связывание: выигрывает пара с большим интегралом."""
    paf = [[(1.0, 0.0)] * 4 for _ in range(4)]
    along = paf_line_integral(paf, (0, 0), (3, 0))
    across = paf_line_integral(paf, (0, 0), (0, 3))
    assert along > across


def test_paf_of_a_zero_length_limb_is_zero():
    paf = [[(1.0, 0.0)] * 4]
    assert paf_line_integral(paf, (2, 0), (2, 0)) == APPROX(0.0)


def test_paf_clamps_samples_that_fall_outside_the_grid():
    paf = [[(1.0, 0.0)] * 4]
    assert paf_line_integral(paf, (0, 0), (99, 0)) == APPROX(1.0)
