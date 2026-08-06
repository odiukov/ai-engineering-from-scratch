"""Тесты к уроку «Vision Transformers: патчи вместо пикселей». Правь exercise.py."""

import pytest

from exercise import (
    add_cls_and_pos,
    attention_pairs,
    linear_project,
    patch_grid,
    patchify,
    pos_2d,
    unpatchify,
    vit_param_count,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def gray(height, width):
    """Однокональная картинка, пиксель равен своему номеру: 0, 1, 2, ..."""
    return [[[i * width + j] for j in range(width)] for i in range(height)]


def rgb(height, width):
    """Трёхканальная картинка: пиксель (i, j) это [i, j, i + j]."""
    return [[[float(i), float(j), float(i + j)] for j in range(width)] for i in range(height)]


# ------------------------------------------------------------- patch_grid
def test_patch_grid_counts_patches_in_both_directions():
    assert patch_grid(8, 12, 4) == (2, 3)


def test_vit_base_cuts_the_image_into_196_patches():
    """224 / 16 = 14, сетка 14x14 — та самая 196 из статьи про ViT."""
    grid_h, grid_w = patch_grid(224, 224, 16)
    assert grid_h * grid_w == 196


def test_patch_grid_rejects_a_size_that_does_not_divide():
    with pytest.raises(ValueError):
        patch_grid(30, 30, 16)


def test_patch_grid_rejects_a_nonpositive_patch_size():
    with pytest.raises(ValueError):
        patch_grid(24, 24, 0)


# --------------------------------------------------------------- patchify
def test_patchify_produces_one_vector_per_grid_cell():
    assert len(patchify(gray(8, 12), 4)) == 6


def test_flat_patch_length_is_patch_area_times_channels():
    """6x6 патч трёхканальной картинки — это вектор из 108 чисел."""
    patches = patchify(rgb(24, 24), 6)
    assert len(patches[0]) == 6 * 6 * 3


def test_patchify_walks_the_grid_in_raster_order():
    """Слева направо, потом сверху вниз — и внутри патча так же."""
    assert patchify(gray(4, 4), 2) == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]


def test_patchify_keeps_the_channels_of_a_pixel_together():
    """Каналы одного пикселя идут подряд, а не тремя отдельными плоскостями."""
    assert patchify(rgb(2, 2), 2)[0] == [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 2]


# ------------------------------------------------------------- unpatchify
def test_unpatchify_restores_the_image():
    """Round-trip — единственная дешёвая проверка порядка обхода."""
    image = rgb(6, 6)
    assert unpatchify(patchify(image, 3), 3, 6, 6) == image


def test_unpatchify_matches_the_hand_written_example():
    assert unpatchify([[1, 2, 3, 4]], 2, 2, 2) == [[[1], [2]], [[3], [4]]]


def test_unpatchify_rejects_a_wrong_patch_count():
    with pytest.raises(ValueError):
        unpatchify([[1, 2, 3, 4]], 2, 4, 4)


# ---------------------------------------------------------- linear_project
def test_projection_gives_one_token_of_d_model_per_patch():
    patches = patchify(rgb(4, 4), 2)
    W = [[0.1] * 5 for _ in range(2 * 2 * 3)]
    tokens = linear_project(patches, W)
    assert len(tokens) == 4 and len(tokens[0]) == 5


def test_identity_matrix_leaves_the_patch_unchanged():
    W = [[1.0, 0.0], [0.0, 1.0]]
    assert flat(linear_project([[3.0, 4.0]], W)) == APPROX([3.0, 4.0])


def test_projection_is_linear():
    W = [[2.0], [3.0]]
    single = linear_project([[1.0, 1.0]], W)[0][0]
    doubled = linear_project([[2.0, 2.0]], W)[0][0]
    assert doubled == APPROX(2 * single)


def test_patch_embedding_only_sees_its_own_patch():
    """Проекция локальна: правка пикселя в последнем патче не может задеть
    нулевой токен. Именно поэтому patch embedding — это свёртка с шагом,
    равным ядру, а не полносвязный слой по всей картинке."""
    W = [[float(k + 1)] for k in range(4)]
    image = gray(4, 4)
    before = linear_project(patchify(image, 2), W)
    image[3][3] = [999]
    after = linear_project(patchify(image, 2), W)
    assert before[0] == APPROX(after[0]) and before[3][0] != after[3][0]


def test_linear_project_rejects_a_width_mismatch():
    with pytest.raises(ValueError):
        linear_project([[1.0, 2.0, 3.0]], [[1.0], [1.0]])


# ----------------------------------------------------------------- pos_2d
def test_one_positional_vector_of_d_model_per_patch():
    pos = pos_2d(3, 4, 8)
    assert len(pos) == 12 and all(len(v) == 8 for v in pos)


def test_first_patch_encodes_zero_as_sin_and_cos():
    """sin(0) = 0, cos(0) = 1 — у патча (0, 0) обе половины такие."""
    assert pos_2d(1, 1, 8)[0] == APPROX([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_patches_of_one_row_share_the_row_half():
    pos = pos_2d(3, 4, 8)
    assert pos[0][:4] == APPROX(pos[3][:4])      # обе из строки 0
    assert pos[0][:4] != pytest.approx(pos[4][:4], abs=1e-9)   # строка 1


def test_patches_of_one_column_share_the_column_half():
    pos = pos_2d(3, 4, 8)
    assert pos[0][4:] == APPROX(pos[4][4:])      # оба из столбца 0
    assert pos[0][4:] != pytest.approx(pos[1][4:], abs=1e-9)   # столбец 1


def test_positional_values_stay_between_minus_one_and_one():
    assert all(-1.0 <= v <= 1.0 for vec in pos_2d(5, 7, 16) for v in vec)


def test_pos_2d_rejects_d_model_not_divisible_by_four():
    with pytest.raises(ValueError):
        pos_2d(2, 2, 6)


# -------------------------------------------------------- add_cls_and_pos
def test_cls_makes_the_sequence_one_token_longer():
    """196 патчей ViT-Base превращаются в 197 токенов — это и есть 197 в
    форме (1, 197, 768) из HuggingFace."""
    tokens = [[0.0] * 4 for _ in range(196)]
    seq = add_cls_and_pos(tokens, [0.0] * 4, pos_2d(14, 14, 4))
    assert len(seq) == 197


def test_cls_goes_first_and_gets_no_positional_encoding():
    seq = add_cls_and_pos([[1.0], [2.0]], [7.0], [[10.0], [20.0]])
    assert seq[0] == APPROX([7.0])


def test_positional_encoding_is_added_to_every_patch():
    seq = add_cls_and_pos([[1.0], [2.0]], [0.0], [[10.0], [20.0]])
    assert flat(seq[1:]) == APPROX([11.0, 22.0])


def test_add_cls_and_pos_does_not_mutate_its_inputs():
    tokens = [[1.0], [2.0]]
    pos = [[10.0], [20.0]]
    add_cls_and_pos(tokens, [0.0], pos)
    assert flat(tokens) == APPROX([1.0, 2.0]) and flat(pos) == APPROX([10.0, 20.0])


def test_add_cls_and_pos_rejects_a_length_mismatch():
    with pytest.raises(ValueError):
        add_cls_and_pos([[1.0], [2.0]], [0.0], [[10.0]])


# -------------------------------------------------------- attention_pairs
def test_attention_pairs_counts_the_cls_token_too():
    """4 патча плюс [CLS] это 5 токенов, 25 пар — без [CLS] было бы 16."""
    assert attention_pairs(4, 4, 2) == 25


def test_halving_the_patch_makes_attention_sixteen_times_dearer():
    """Вчетверо больше токенов — примерно в шестнадцать раз дороже attention.
    Вот вся цена перехода с патча 16x16 на 8x8."""
    ratio = attention_pairs(224, 224, 8) / attention_pairs(224, 224, 16)
    assert 15.0 < ratio < 16.0


# -------------------------------------------------------- vit_param_count
def test_vit_base_lands_near_86_million():
    """Ориентир из статьи: ViT-Base/16 это ~86M против ~25M у ResNet-50."""
    assert 80e6 < vit_param_count(768, 12, 196, 16) < 92e6


def test_parameters_grow_quadratically_with_d_model():
    """Удвоение d_model это примерно четырёхкратный рост: и attention, и FFN
    квадратичны по ширине."""
    small = vit_param_count(256, 1, 4, 2, channels=1, n_classes=0)
    big = vit_param_count(512, 1, 4, 2, channels=1, n_classes=0)
    assert 3.9 < big / small < 4.1


def test_parameters_grow_linearly_with_depth():
    """Каждый следующий слой стоит ровно столько же, сколько предыдущий."""
    zero = vit_param_count(128, 0, 16, 4)
    twelve = vit_param_count(128, 12, 16, 4)
    twenty_four = vit_param_count(128, 24, 16, 4)
    assert twenty_four - twelve == twelve - zero


def test_classifier_head_scales_with_the_number_of_classes():
    thousand = vit_param_count(768, 12, 196, 16, n_classes=1000)
    ten = vit_param_count(768, 12, 196, 16, n_classes=10)
    assert thousand - ten == 990 * 768
