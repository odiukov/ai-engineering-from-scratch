"""Тесты к уроку «Свёртки с нуля». Правь exercise.py."""

import random

import pytest

from exercise import (
    conv2d,
    conv2d_im2col,
    conv2d_multichannel,
    conv_output_size,
    conv_params,
    im2col,
    pad2d,
    receptive_field,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

DELTA = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
BLUR = [[1 / 9] * 3 for _ in range(3)]
SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def flat(x):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flat(item))
        return out
    return [x]


def rand_tensor(shape, seed):
    """Воспроизводимый тензор произвольной формы из N(0, 1)."""
    rng = random.Random(seed)

    def build(dims):
        if len(dims) == 1:
            return [rng.gauss(0.0, 1.0) for _ in range(dims[0])]
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(shape)


def step_image(size=6):
    """Вертикальный перепад яркости: слева нули, справа единицы."""
    return [[0.0 if j < size // 2 else 1.0 for j in range(size)] for _ in range(size)]


# ------------------------------------------------------- conv_output_size
def test_output_size_matches_the_lesson_table():
    assert conv_output_size(32, 3) == 30
    assert conv_output_size(32, 3, padding=1) == 32
    assert conv_output_size(32, 3, padding=1, stride=2) == 16
    assert conv_output_size(32, 2, stride=2) == 16
    assert conv_output_size(32, 7, padding=3, stride=2) == 16


def test_same_padding_rule_preserves_size_for_odd_kernels():
    """P = (K-1)/2 при stride 1 обязано вернуть исходный размер."""
    for k in (1, 3, 5, 7, 9):
        assert conv_output_size(64, k, padding=(k - 1) // 2) == 64


def test_output_size_floors_the_leftover_window():
    """Хвост, который не покрывается последним окном, отбрасывается."""
    assert conv_output_size(10, 3, stride=3) == 3


def test_stride_two_roughly_halves_the_map():
    assert conv_output_size(224, 3, padding=1, stride=2) == 112


# ----------------------------------------------------------- conv_params
def test_param_count_of_a_64_channel_conv_on_rgb():
    assert conv_params(3, 64, 3) == 1792


def test_bias_adds_exactly_one_number_per_output_channel():
    assert conv_params(3, 64, 3) - conv_params(3, 64, 3, bias=False) == 64


def test_params_do_not_depend_on_image_size():
    """Parameter sharing: одно ядро на всю картинку любого размера."""
    assert conv_params(3, 64, 3) == 1792


def test_two_3x3_convs_are_cheaper_than_one_5x5():
    """Тот самый расчёт, из-за которого VGG отказалась от больших ядер."""
    c = 64
    assert 2 * conv_params(c, c, 3) < conv_params(c, c, 5)


# --------------------------------------------------------------- pad2d
def test_pad_surrounds_the_input_with_zeros():
    assert pad2d([[1]], 1) == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_pad_zero_returns_the_same_values():
    assert pad2d([[1, 2], [3, 4]], 0) == [[1, 2], [3, 4]]


def test_pad_grows_both_axes_by_two_p():
    out = pad2d([[1, 2, 3], [4, 5, 6]], 2)
    assert (len(out), len(out[0])) == (6, 7)


def test_pad_does_not_mutate_the_input():
    x = [[1, 2], [3, 4]]
    pad2d(x, 1)
    assert x == [[1, 2], [3, 4]]


# --------------------------------------------------------------- conv2d
def test_delta_kernel_is_the_identity():
    """Ядро с единицей в центре и padding=1 обязано вернуть вход как есть."""
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert flat(conv2d(x, DELTA, padding=1)) == APPROX(flat(x))


def test_blur_of_a_constant_image_is_the_same_constant():
    """Веса усредняющего ядра дают в сумме 1, значит плоскость не меняется."""
    x = [[7.0] * 5 for _ in range(5)]
    assert flat(conv2d(x, BLUR)) == pytest.approx([7.0] * 9)


def test_sobel_x_lights_up_a_vertical_edge():
    y = conv2d(step_image(6), SOBEL_X)
    assert y[0] == pytest.approx([0.0, 4.0, 4.0, 0.0])


def test_sobel_y_is_blind_to_a_vertical_edge():
    """Горизонтальный детектор на вертикальном крае обязан молчать."""
    y = conv2d(step_image(6), SOBEL_Y)
    assert flat(y) == pytest.approx([0.0] * 16)


def test_conv_is_translation_equivariant():
    """Сдвинули вход на пиксель — выход сдвинулся ровно на пиксель."""
    kernel = [[1, 2, 3]]
    a = conv2d([[0, 0, 1, 0, 0, 0, 0]], kernel)
    b = conv2d([[0, 0, 0, 1, 0, 0, 0]], kernel)
    assert b[0][1:] == APPROX(a[0][:-1])


def test_conv_is_cross_correlation_not_flipped():
    """Ядро не переворачивается: иначе знак Sobel окажется противоположным."""
    y = conv2d([[0, 0, 1]], [[0, 0, 1]])
    assert y[0] == APPROX([1.0])


def test_stride_two_skips_every_other_window():
    x = [[1, 2, 3, 4, 5]]
    dense = conv2d(x, [[1]])
    strided = conv2d(x, [[1]], stride=2)
    assert strided[0] == APPROX([dense[0][0], dense[0][2], dense[0][4]])


# --------------------------------------------------- conv2d_multichannel
def test_multichannel_sums_across_input_channels():
    x = [[[1, 2]], [[10, 20]]]
    w = [[[[1]], [[1]]]]
    assert flat(conv2d_multichannel(x, w)) == APPROX([11.0, 22.0])


def test_multichannel_output_has_one_plane_per_kernel():
    x = rand_tensor((3, 8, 8), seed=1)
    w = rand_tensor((5, 3, 3, 3), seed=2)
    y = conv2d_multichannel(x, w, padding=1)
    assert (len(y), len(y[0]), len(y[0][0])) == (5, 8, 8)


def test_bias_shifts_each_output_channel_by_its_own_number():
    x = [[[0, 0], [0, 0]]]
    w = [[[[1, 1], [1, 1]]], [[[1, 1], [1, 1]]]]
    y = conv2d_multichannel(x, w, bias=[3.0, -7.0])
    assert (y[0][0][0], y[1][0][0]) == APPROX((3.0, -7.0))


def test_multichannel_kernel_can_ignore_a_channel():
    """Нулевой срез ядра означает, что канал не влияет на выход."""
    w = [[[[1]], [[0]]]]
    a = conv2d_multichannel([[[5]], [[100]]], w)
    b = conv2d_multichannel([[[5]], [[-100]]], w)
    assert flat(a) == APPROX(flat(b))


# --------------------------------------------------------------- im2col
def test_im2col_matrix_has_one_column_per_output_position():
    x = rand_tensor((3, 5, 5), seed=3)
    cols, h_out, w_out = im2col(x, 3, 3, padding=1)
    assert (h_out, w_out) == (5, 5)
    assert (len(cols), len(cols[0])) == (3 * 3 * 3, 5 * 5)


def test_im2col_column_holds_exactly_one_window():
    cols, h_out, w_out = im2col([[[1, 2], [3, 4]]], 2, 2)
    assert (h_out, w_out) == (1, 1)
    assert [row[0] for row in cols] == APPROX([1, 2, 3, 4])


def test_im2col_orders_channel_then_row_then_column():
    """Порядок обязан совпасть с разворачиванием ядра, иначе матмул соврёт."""
    x = [[[1]], [[2]]]
    cols, _, _ = im2col(x, 1, 1)
    assert [row[0] for row in cols] == APPROX([1, 2])


def test_im2col_duplicates_overlapping_pixels():
    """Плата за скорость: перекрытие окон копируется, память растёт в K*K раз."""
    cols, _, _ = im2col([[[1, 2, 3]]], 1, 2)
    assert sum(len(row) for row in cols) > 3


# --------------------------------------------------------- conv2d_im2col
def test_im2col_conv_matches_the_naive_conv():
    x = rand_tensor((3, 9, 9), seed=4)
    w = rand_tensor((4, 3, 3, 3), seed=5)
    b = rand_tensor((4,), seed=6)
    slow = conv2d_multichannel(x, w, b, padding=1)
    fast = conv2d_im2col(x, w, b, padding=1)
    assert flat(fast) == pytest.approx(flat(slow), abs=1e-9)


def test_im2col_conv_matches_the_naive_conv_with_stride():
    x = rand_tensor((2, 8, 8), seed=7)
    w = rand_tensor((3, 2, 3, 3), seed=8)
    slow = conv2d_multichannel(x, w, stride=2, padding=1)
    fast = conv2d_im2col(x, w, stride=2, padding=1)
    assert flat(fast) == pytest.approx(flat(slow), abs=1e-9)


def test_im2col_conv_keeps_the_delta_kernel_identity():
    x = rand_tensor((1, 6, 6), seed=9)
    w = [[DELTA]]
    assert flat(conv2d_im2col(x, w, padding=1)) == pytest.approx(flat(x), abs=1e-9)


# ---------------------------------------------------------- receptive_field
def test_two_stacked_3x3_see_the_same_area_as_one_5x5():
    assert receptive_field([(3, 1), (3, 1)]) == receptive_field([(5, 1)])


def test_receptive_field_grows_linearly_without_stride():
    assert [receptive_field([(3, 1)] * n) for n in (1, 2, 3, 4)] == [3, 5, 7, 9]


def test_stride_makes_the_receptive_field_grow_faster():
    assert receptive_field([(3, 2), (3, 1)]) > receptive_field([(3, 1), (3, 1)])


def test_single_1x1_conv_sees_exactly_one_pixel():
    assert receptive_field([(1, 1)]) == 1


def test_empty_stack_sees_one_pixel():
    assert receptive_field([]) == 1
