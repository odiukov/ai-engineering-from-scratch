"""Тесты к уроку «Монокулярная глубина и геометрия». Правь exercise.py."""

import pytest

from exercise import (
    abs_rel_error,
    align_scale_shift,
    aligned_abs_rel,
    delta_accuracy,
    depth_to_point_cloud,
    flatten_valid,
    lift_box_to_3d,
    pixel_to_camera,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(nested):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    out = []
    for item in nested:
        out.extend(flat(item) if isinstance(item, (list, tuple)) else [item])
    return out


# ------------------------------------------------------------ flatten_valid
def test_flatten_valid_drops_zero_ground_truth():
    assert flatten_valid([[1.0, 2.0]], [[1.0, 0.0]]) == ([1.0], [1.0])


def test_flatten_valid_drops_nan_and_inf():
    p, t = flatten_valid([[1.0, float("nan"), 3.0]], [[1.0, 2.0, float("inf")]])
    assert (p, t) == ([1.0], [1.0])


def test_flatten_valid_walks_row_by_row():
    p, t = flatten_valid([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [1.0, 1.0]])
    assert p == [1.0, 2.0, 3.0, 4.0]


def test_flatten_valid_rejects_maps_of_different_shape():
    with pytest.raises(ValueError):
        flatten_valid([[1.0, 2.0]], [[1.0]])


# ------------------------------------------------------------ abs_rel_error
def test_abs_rel_error_divides_by_ground_truth():
    assert abs_rel_error([[1.0, 4.0]], [[1.0, 2.0]]) == APPROX(0.5)


def test_abs_rel_error_is_not_symmetric():
    """Ошибка нормируется на истину — перепутанные аргументы дают другое число."""
    a = abs_rel_error([[2.0]], [[1.0]])
    b = abs_rel_error([[1.0]], [[2.0]])
    assert a == APPROX(1.0)
    assert b == APPROX(0.5)


def test_abs_rel_error_ignores_masked_pixels():
    """Дырка в ground truth не должна портить метрику."""
    assert abs_rel_error([[5.0, 999.0]], [[5.0, 0.0]]) == APPROX(0.0)


def test_abs_rel_error_without_valid_pixels_raises():
    with pytest.raises(ValueError):
        abs_rel_error([[1.0]], [[0.0]])


# ----------------------------------------------------------- delta_accuracy
def test_delta_accuracy_of_a_perfect_prediction_is_one():
    assert delta_accuracy([[1.0, 5.0]], [[1.0, 5.0]]) == APPROX(1.0)


def test_delta_accuracy_counts_the_fraction_within_the_threshold():
    assert delta_accuracy([[1.0, 1.0]], [[1.0, 10.0]]) == APPROX(0.5)


def test_delta_accuracy_is_symmetric_unlike_abs_rel():
    """max(p/t, t/p) не различает, кто из двух больше."""
    pred, target = [[1.0, 1.2, 4.0]], [[1.0, 1.0, 1.0]]
    assert delta_accuracy(pred, target) == APPROX(delta_accuracy(target, pred))


def test_delta_accuracy_grows_with_a_looser_threshold():
    pred, target = [[1.0, 1.3, 2.0]], [[1.0, 1.0, 1.0]]
    assert delta_accuracy(pred, target, 1.25) < delta_accuracy(pred, target, 1.25 ** 3)


def test_delta_accuracy_treats_a_non_positive_prediction_as_a_miss():
    """Модель выдала ноль — отношение не определено, пиксель точным не считаем."""
    assert delta_accuracy([[0.0, 1.0]], [[1.0, 1.0]]) == APPROX(0.5)


# --------------------------------------------------------- align_scale_shift
def test_align_recovers_a_known_scale_and_shift():
    assert align_scale_shift([[1.0, 2.0]], [[3.0, 5.0]]) == pytest.approx((2.0, 1.0))


def test_align_of_an_already_correct_prediction_is_identity():
    assert align_scale_shift([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]]) == pytest.approx((1.0, 0.0))


def test_align_finds_a_stationary_point_of_the_squared_error():
    """Численная проверка: в найденной точке производные SSE по a и b нулевые."""
    pred, target = [[1.0, 2.0, 3.0, 4.0]], [[2.1, 3.9, 6.2, 7.8]]
    a, b = align_scale_shift(pred, target)
    p, t = pred[0], target[0]

    def sse(aa, bb):
        return sum((aa * pi + bb - ti) ** 2 for pi, ti in zip(p, t))

    h = 1e-5
    d_a = (sse(a + h, b) - sse(a - h, b)) / (2 * h)
    d_b = (sse(a, b + h) - sse(a, b - h)) / (2 * h)
    assert d_a == pytest.approx(0.0, abs=1e-6)
    assert d_b == pytest.approx(0.0, abs=1e-6)


def test_align_beats_any_nearby_pair_on_squared_error():
    pred, target = [[1.0, 2.0, 3.0, 4.0]], [[2.1, 3.9, 6.2, 7.8]]
    a, b = align_scale_shift(pred, target)
    p, t = pred[0], target[0]
    sse = lambda aa, bb: sum((aa * pi + bb - ti) ** 2 for pi, ti in zip(p, t))
    best = sse(a, b)
    assert best <= min(sse(a + da, b + db) for da in (-0.1, 0.0, 0.1) for db in (-0.1, 0.0, 0.1))


def test_align_rejects_a_constant_prediction():
    """Плоская карта не задаёт масштаб — прямую через одну точку не провести."""
    with pytest.raises(ValueError):
        align_scale_shift([[2.0, 2.0, 2.0]], [[1.0, 5.0, 9.0]])


# --------------------------------------------------------- aligned_abs_rel
def test_aligned_abs_rel_is_zero_for_an_affinely_related_prediction():
    assert aligned_abs_rel([[1.0, 2.0, 3.0]], [[3.0, 5.0, 7.0]]) == APPROX(0.0)


def test_aligned_abs_rel_is_invariant_to_scaling_the_prediction():
    """Умножили относительную глубину на 100 — метрика обязана не заметить."""
    pred, target = [[1.0, 2.0, 3.1, 4.0]], [[1.1, 2.0, 2.9, 4.2]]
    scaled = [[100.0 * v for v in row] for row in pred]
    assert aligned_abs_rel(scaled, target) == pytest.approx(aligned_abs_rel(pred, target), abs=1e-9)


def test_aligned_abs_rel_is_invariant_to_shifting_the_prediction():
    pred, target = [[1.0, 2.0, 3.1, 4.0]], [[1.1, 2.0, 2.9, 4.2]]
    shifted = [[v + 7.0 for v in row] for row in pred]
    assert aligned_abs_rel(shifted, target) == pytest.approx(aligned_abs_rel(pred, target), abs=1e-9)


def test_aligning_never_hurts_a_badly_scaled_prediction():
    pred, target = [[1.0, 2.0, 3.0, 4.0]], [[10.0, 20.0, 30.0, 40.0]]
    assert aligned_abs_rel(pred, target) < abs_rel_error(pred, target)


# ---------------------------------------------------------- pixel_to_camera
def test_pixel_at_the_principal_point_looks_straight_ahead():
    assert pixel_to_camera(160, 120, 2.0, (320.0, 320.0, 160.0, 120.0)) == pytest.approx((0.0, 0.0, 2.0))


def test_pixel_offset_scales_with_depth_over_focal_length():
    assert pixel_to_camera(320, 120, 2.0, (320.0, 320.0, 160.0, 120.0)) == pytest.approx((1.0, 0.0, 2.0))


def test_doubling_depth_moves_the_point_along_the_same_ray():
    """Пиксель задаёт луч: вдвое дальше — все три координаты вдвое больше."""
    near = pixel_to_camera(200, 50, 1.0, (320.0, 320.0, 160.0, 120.0))
    far = pixel_to_camera(200, 50, 2.0, (320.0, 320.0, 160.0, 120.0))
    assert far == pytest.approx(tuple(2 * c for c in near))


def test_zero_focal_length_raises_instead_of_dividing_by_zero():
    with pytest.raises(ValueError):
        pixel_to_camera(10, 10, 1.0, (0.0, 320.0, 160.0, 120.0))


# ----------------------------------------------------- depth_to_point_cloud
def test_point_cloud_keeps_the_map_shape():
    depth = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    cloud = depth_to_point_cloud(depth, (320.0, 320.0, 160.0, 120.0))
    assert len(cloud) == 2
    assert all(len(row) == 3 for row in cloud)
    assert all(len(point) == 3 for row in cloud for point in row)


def test_point_cloud_z_channel_is_the_depth_itself():
    depth = [[1.0, 2.0], [3.0, 4.0]]
    cloud = depth_to_point_cloud(depth, (320.0, 320.0, 160.0, 120.0))
    assert [point[2] for row in cloud for point in row] == APPROX([1.0, 2.0, 3.0, 4.0])


def test_point_cloud_indexing_is_row_v_column_u():
    """Перепутанные u и v дают зеркальное облако, которое на глаз незаметно."""
    depth = [[1.0, 1.0], [1.0, 1.0]]
    cloud = depth_to_point_cloud(depth, (1.0, 1.0, 0.0, 0.0))
    # пиксель строки 0, столбца 1 -> X = 1, Y = 0
    assert cloud[0][1][0] == APPROX(1.0)
    assert cloud[0][1][1] == APPROX(0.0)


def test_point_cloud_of_a_single_pixel():
    assert flat(depth_to_point_cloud([[2.0]], (1.0, 1.0, 0.0, 0.0))) == APPROX([0.0, 0.0, 2.0])


# ---------------------------------------------------------- lift_box_to_3d
def test_lift_box_uses_the_box_centre_and_its_depth():
    depth = [[2.0, 2.0], [2.0, 2.0]]
    assert lift_box_to_3d(depth, (0, 0, 2, 2), (1.0, 1.0, 1.0, 1.0)) == pytest.approx((0.0, 0.0, 2.0))


def test_lift_box_is_robust_to_a_few_background_pixels():
    """Один выброс на 100 метров медиану не сдвинет, а среднее — утащит."""
    depth = [[2.0, 2.0, 100.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    assert lift_box_to_3d(depth, (0, 0, 3, 3), (1.0, 1.0, 1.5, 1.5))[2] == APPROX(2.0)


def test_lift_box_ignores_invalid_depth_inside_the_box():
    depth = [[0.0, 3.0], [0.0, 3.0]]
    assert lift_box_to_3d(depth, (0, 0, 2, 2), (1.0, 1.0, 1.0, 1.0))[2] == APPROX(3.0)


def test_lift_box_without_any_valid_depth_raises():
    with pytest.raises(ValueError):
        lift_box_to_3d([[0.0, 0.0], [0.0, 0.0]], (0, 0, 2, 2), (1.0, 1.0, 1.0, 1.0))


def test_lift_box_of_an_empty_box_raises():
    with pytest.raises(ValueError):
        lift_box_to_3d([[1.0, 1.0], [1.0, 1.0]], (1, 1, 1, 1), (1.0, 1.0, 1.0, 1.0))
