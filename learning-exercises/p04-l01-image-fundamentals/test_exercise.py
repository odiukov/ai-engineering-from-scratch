"""Тесты к уроку «Основы изображений: пиксели, каналы, цветовые пространства». Правь exercise.py."""

import pytest

from exercise import (
    chw_to_hwc,
    deprocess_imagenet,
    hwc_to_chw,
    preprocess_imagenet,
    resize_bilinear,
    resize_nearest,
    rgb_to_grayscale,
    rgb_to_hsv,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(x):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flat(item))
        return out
    return [x]


def ramp_hwc(h=3, w=4):
    """Детерминированное HWC-изображение: каждый канал со своим наклоном."""
    return [
        [[(i * w + j) % 256, (2 * (i * w + j)) % 256, (255 - i * w - j) % 256] for j in range(w)]
        for i in range(h)
    ]


# ------------------------------------------------------------- hwc_to_chw
def test_hwc_to_chw_worked_example():
    assert hwc_to_chw([[[1, 2, 3], [4, 5, 6]]]) == [[[1, 4]], [[2, 5]], [[3, 6]]]


def test_hwc_to_chw_moves_channels_to_the_outer_axis():
    img = ramp_hwc(3, 4)
    chw = hwc_to_chw(img)
    assert (len(chw), len(chw[0]), len(chw[0][0])) == (3, 3, 4)


def test_hwc_to_chw_keeps_every_pixel_value():
    img = ramp_hwc(3, 4)
    chw = hwc_to_chw(img)
    assert chw[2][1][3] == img[1][3][2]


def test_hwc_to_chw_does_not_alias_the_input():
    """Правка результата не должна портить исходное изображение."""
    img = ramp_hwc(2, 2)
    chw = hwc_to_chw(img)
    chw[0][0][0] = -999
    assert img[0][0][0] != -999


# ------------------------------------------------------------- chw_to_hwc
def test_chw_to_hwc_worked_example():
    assert chw_to_hwc([[[1, 4]], [[2, 5]], [[3, 6]]]) == [[[1, 2, 3], [4, 5, 6]]]


def test_layout_roundtrip_is_identity():
    img = ramp_hwc(5, 3)
    assert chw_to_hwc(hwc_to_chw(img)) == img


def test_chw_to_hwc_restores_the_original_shape():
    img = ramp_hwc(5, 3)
    back = chw_to_hwc(hwc_to_chw(img))
    assert (len(back), len(back[0]), len(back[0][0])) == (5, 3, 3)


# -------------------------------------------------------- rgb_to_grayscale
def test_grayscale_of_pure_red_uses_the_bt601_weight():
    assert flat(rgb_to_grayscale([[[255, 0, 0]]])) == pytest.approx([76.245])


def test_grayscale_is_not_a_plain_channel_average():
    """Среднее дало бы 85, правильный ответ 76.245 — веса неравные."""
    value = rgb_to_grayscale([[[255, 0, 0]]])[0][0]
    assert abs(value - 85.0) > 5


def test_grayscale_keeps_gray_unchanged():
    """Сумма весов равна 1.0, значит серый переходит сам в себя."""
    assert flat(rgb_to_grayscale([[[100, 100, 100]]])) == pytest.approx([100.0])


def test_grayscale_ranks_green_above_red_above_blue():
    """Глаз чувствительнее всего к зелёному — веса отражают это."""
    r = rgb_to_grayscale([[[255, 0, 0]]])[0][0]
    g = rgb_to_grayscale([[[0, 255, 0]]])[0][0]
    b = rgb_to_grayscale([[[0, 0, 255]]])[0][0]
    assert b < r < g


def test_grayscale_drops_the_channel_axis():
    gray = rgb_to_grayscale(ramp_hwc(3, 4))
    assert (len(gray), len(gray[0])) == (3, 4)
    assert not isinstance(gray[0][0], list)


# --------------------------------------------------------------- rgb_to_hsv
def test_hsv_primary_colors_land_on_their_hues():
    assert rgb_to_hsv((255, 0, 0))[0] == APPROX(0.0)
    assert rgb_to_hsv((0, 255, 0))[0] == APPROX(120.0)
    assert rgb_to_hsv((0, 0, 255))[0] == APPROX(240.0)


def test_hsv_of_gray_has_zero_saturation():
    """Ловушка delta == 0: деления на ноль быть не должно."""
    h, s, v = rgb_to_hsv((10, 10, 10))
    assert (h, s) == APPROX((0.0, 0.0))
    assert v == APPROX(10 / 255)


def test_hsv_of_black_does_not_divide_by_zero():
    assert rgb_to_hsv((0, 0, 0)) == APPROX((0.0, 0.0, 0.0))


def test_hue_and_saturation_ignore_brightness():
    """Затемнили вдвое — оттенок и насыщенность те же, поехал только V."""
    bright = rgb_to_hsv((200, 100, 50))
    dark = rgb_to_hsv((100, 50, 25))
    assert bright[0] == pytest.approx(dark[0], abs=1e-6)
    assert bright[1] == pytest.approx(dark[1], abs=1e-6)
    assert dark[2] < bright[2]


def test_hsv_value_is_the_brightest_channel():
    assert rgb_to_hsv((30, 200, 90))[2] == APPROX(200 / 255)


# --------------------------------------------- preprocess / deprocess
def test_preprocess_returns_chw_layout():
    x = preprocess_imagenet(ramp_hwc(4, 6))
    assert (len(x), len(x[0]), len(x[0][0])) == (3, 4, 6)


def test_preprocess_of_black_matches_minus_mean_over_std():
    x = preprocess_imagenet([[[0, 0, 0]]])
    assert flat(x) == pytest.approx(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], abs=1e-9
    )


def test_preprocess_centers_a_mean_valued_image_at_zero():
    """Пиксель, равный ImageNet-среднему, обязан стать нулём."""
    px = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    x = preprocess_imagenet([[px]])
    assert flat(x) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_preprocess_uses_per_channel_statistics():
    """Одно и то же байтовое значение в трёх каналах даёт три разных числа."""
    x = flat(preprocess_imagenet([[[128, 128, 128]]]))
    assert x[0] != pytest.approx(x[1], abs=1e-6)
    assert x[1] != pytest.approx(x[2], abs=1e-6)


def test_preprocess_deprocess_roundtrip_is_lossless():
    img = ramp_hwc(6, 5)
    assert deprocess_imagenet(preprocess_imagenet(img)) == img


def test_deprocess_clips_out_of_range_values():
    """После аугментаций значения вылетают за границы — clip обязателен."""
    out = deprocess_imagenet([[[-50.0]], [[50.0]], [[0.0]]])
    assert out[0][0][0] == 0
    assert out[0][0][1] == 255


# ------------------------------------------------------------ resize_nearest
def test_nearest_upscale_replicates_pixels():
    assert resize_nearest([[1, 2]], 1, 4) == [[1, 1, 2, 2]]


def test_nearest_to_the_same_size_is_identity():
    img = [[1, 2, 3], [4, 5, 6]]
    assert resize_nearest(img, 2, 3) == img


def test_nearest_never_invents_new_values():
    """Поэтому только он годится для масок с id классов."""
    img = [[3, 5], [5, 3]]
    out = resize_nearest(img, 5, 7)
    assert set(v for row in out for v in row) <= {3, 5}


def test_nearest_downscale_drops_rows_and_columns():
    img = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    assert resize_nearest(img, 2, 2) == [[1, 3], [9, 11]]


# ----------------------------------------------------------- resize_bilinear
def test_bilinear_preserves_a_linear_ramp():
    assert flat(resize_bilinear([[0, 1, 2]], 1, 5)) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_bilinear_to_the_same_size_is_identity():
    img = [[1.0, 2.0], [3.0, 4.0]]
    assert flat(resize_bilinear(img, 2, 2)) == pytest.approx(flat(img))


def test_bilinear_of_a_constant_image_stays_constant():
    out = resize_bilinear([[5, 5], [5, 5]], 3, 3)
    assert flat(out) == pytest.approx([5.0] * 9)


def test_bilinear_keeps_the_corners():
    """Отображение «по углам»: крайние пиксели попадают в крайние."""
    img = [[0, 10], [20, 30]]
    out = resize_bilinear(img, 5, 5)
    assert (out[0][0], out[0][4], out[4][0], out[4][4]) == pytest.approx((0, 10, 20, 30))


def test_bilinear_creates_intermediate_values_unlike_nearest():
    img = [[0, 100]]
    smooth = resize_bilinear(img, 1, 3)
    blocky = resize_nearest(img, 1, 3)
    assert smooth[0][1] == pytest.approx(50.0)
    assert blocky[0][1] in (0, 100)


def test_bilinear_survives_a_single_row_input():
    """H == 1 или out_h == 1 — деление на ноль в формуле координат."""
    assert flat(resize_bilinear([[1, 3]], 2, 2)) == pytest.approx([1.0, 3.0, 1.0, 3.0])
