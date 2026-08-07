"""Тесты к уроку «CNN от LeNet до ResNet». Правь exercise.py."""

import pytest

from exercise import (
    conv_params,
    dense_params,
    gradient_scale,
    lenet5_params,
    lenet5_shapes,
    residual_forward,
    shortcut_kind,
    spatial_out,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------- spatial_out
def test_conv5x5_on_32_gives_28():
    assert spatial_out(32, 5) == 28


def test_pool2x2_halves_the_map():
    assert spatial_out(28, 2, stride=2) == 14
    assert spatial_out(14, 2, stride=2) == 7


def test_padding_one_on_3x3_keeps_size():
    """«Same padding» для 3x3 — это p=1. Вся VGG держится на этом равенстве."""
    for size in (8, 32, 224):
        assert spatial_out(size, 3, padding=1) == size


def test_stride_two_with_padding_downsamples_by_half():
    """Ветка stride=2 в BasicBlock: 32 -> 16 -> 8 -> 4."""
    assert spatial_out(32, 3, stride=2, padding=1) == 16
    assert spatial_out(16, 3, stride=2, padding=1) == 8


def test_kernel_equal_to_size_collapses_to_one_pixel():
    assert spatial_out(7, 7) == 1


# ------------------------------------------------------------- conv_params
def test_lenet_first_conv_has_156_parameters():
    assert conv_params(1, 6, 5) == 156


def test_vgg_style_conv_counts_bias_per_output_channel():
    assert conv_params(3, 64, 3) == 1792
    assert conv_params(3, 64, 3, bias=False) == 1728


def test_disabling_bias_saves_exactly_c_out():
    """Конвенция ResNet: bias=False перед BatchNorm убирает ровно c_out чисел."""
    assert conv_params(64, 128, 3) - conv_params(64, 128, 3, bias=False) == 128


def test_1x1_conv_is_pure_channel_mixing():
    """Ядро 1x1 из Inception: c_in*c_out весов, пространства в формуле нет."""
    assert conv_params(256, 64, 1, bias=False) == 256 * 64


def test_two_3x3_convs_are_cheaper_than_one_5x5():
    """Главный аргумент VGG: 2*9*C^2 = 18C^2 меньше, чем 25*C^2."""
    c = 64
    two_small = 2 * conv_params(c, c, 3, bias=False)
    one_big = conv_params(c, c, 5, bias=False)
    assert two_small < one_big


# ------------------------------------------------------------ dense_params
def test_lenet_first_dense_has_48120_parameters():
    assert dense_params(400, 120) == 48120


def test_last_dense_of_lenet():
    assert dense_params(84, 10) == 850


def test_dense_bias_adds_one_per_output():
    assert dense_params(400, 120) - dense_params(400, 120, bias=False) == 120


def test_dense_layer_dwarfs_a_conv_layer_of_the_same_widths():
    """400->120 стоит в сотни раз дороже свёртки 1->6: отсюда 138M у VGG-16."""
    assert dense_params(400, 120) > 100 * conv_params(1, 6, 5)


# ------------------------------------------------------------ lenet5_shapes
def test_lenet_shape_chain_matches_the_paper():
    assert lenet5_shapes() == [
        ("input", (1, 32, 32)),
        ("conv1", (6, 28, 28)),
        ("pool1", (6, 14, 14)),
        ("conv2", (16, 10, 10)),
        ("pool2", (16, 5, 5)),
    ]


def test_last_feature_map_flattens_to_400():
    c, h, w = lenet5_shapes()[-1][1]
    assert c * h * w == 400


def test_28x28_input_breaks_the_dense_head():
    """Ловушка MNIST: на 28x28 flatten даёт 256, а fc1 ждёт 400."""
    c, h, w = lenet5_shapes(28)[-1][1]
    assert c * h * w == 256


def test_channels_grow_while_spatial_size_shrinks():
    """Общий закон CNN: пространство сжимается, каналы расширяются."""
    shapes = [s for _, s in lenet5_shapes()]
    sizes = [h for _, h, _ in shapes]
    channels = [c for c, _, _ in shapes]
    assert sizes == sorted(sizes, reverse=True)
    assert channels[0] < channels[-1]


# ------------------------------------------------------------ lenet5_params
def test_lenet_total_is_61706():
    assert sum(n for _, n in lenet5_params()) == 61706


def test_lenet_layer_names_are_in_forward_order():
    assert [name for name, _ in lenet5_params()] == [
        "conv1",
        "conv2",
        "fc1",
        "fc2",
        "fc3",
    ]


def test_first_dense_holds_most_of_the_budget():
    """78% параметров LeNet сидят в fc1 — вся история CNN про то, как его убрать."""
    layers = dict(lenet5_params())
    assert layers["fc1"] / sum(layers.values()) > 0.75


def test_both_convolutions_together_are_under_five_percent():
    layers = dict(lenet5_params())
    convs = layers["conv1"] + layers["conv2"]
    assert convs / sum(layers.values()) < 0.05


# ------------------------------------------------------------ shortcut_kind
def test_same_shape_needs_no_projection():
    assert shortcut_kind(64, 64, 1) == "identity"


def test_channel_growth_forces_projection():
    assert shortcut_kind(64, 128, 1) == "projection"


def test_stride_two_forces_projection_even_at_equal_channels():
    """Ошибка номер один при ручной сборке ResNet: забыть про stride."""
    assert shortcut_kind(64, 64, 2) == "projection"


# --------------------------------------------------------- residual_forward
def test_zero_branch_makes_the_block_an_identity():
    """Тот самый аварийный выход: F(x)=0 превращает блок в тождество."""
    x = [1.0, 2.0, -3.0]
    assert residual_forward(x, lambda v: [0.0] * len(v)) == APPROX(x)


def test_branch_output_is_added_not_replaced():
    assert residual_forward([1.0, 2.0], lambda v: [10.0, 20.0]) == APPROX([11.0, 22.0])


def test_block_does_not_mutate_its_input():
    x = [1.0, 2.0]
    residual_forward(x, lambda v: [5.0, 5.0])
    assert x == [1.0, 2.0]


def test_stacking_identity_blocks_never_degrades_the_signal():
    """Сто блоков с нулевой веткой — ровно вход. Плата за глубину нулевая."""
    x = [0.5, -0.5, 2.0]
    y = list(x)
    for _ in range(100):
        y = residual_forward(y, lambda v: [0.0] * len(v))
    assert y == APPROX(x)


# ---------------------------------------------------------- gradient_scale
def test_plain_stack_multiplies_the_gains():
    assert gradient_scale([0.5, 0.5]) == APPROX(0.25)


def test_residual_stack_multiplies_gain_plus_one():
    assert gradient_scale([0.5, 0.5], residual=True) == APPROX(2.25)


def test_dead_residual_block_passes_the_gradient_through_unchanged():
    """g=0 даёт множитель ровно 1: градиент проходит насквозь без потерь."""
    assert gradient_scale([0.0] * 50, residual=True) == APPROX(1.0)


def test_deep_plain_stack_kills_the_gradient():
    """Degradation problem: 50 слоёв с усилением 0.5 — и до входа не доходит ничего."""
    assert gradient_scale([0.5] * 50) < 1e-12


def test_residual_survives_the_same_depth_that_kills_plain():
    gains = [0.5] * 50
    assert gradient_scale(gains, residual=True) > gradient_scale(gains) * 1e12


def test_empty_stack_scales_by_one():
    """Ноль слоёв — градиент не тронут. Пустое произведение равно единице."""
    assert gradient_scale([]) == APPROX(1.0)
