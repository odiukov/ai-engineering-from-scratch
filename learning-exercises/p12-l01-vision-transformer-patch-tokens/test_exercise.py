"""Тесты к уроку «Vision Transformer и патч-токены». Правь exercise.py."""

import pytest

from exercise import (
    add_position_embeddings,
    extract_patches,
    grid_shape,
    mean_pool,
    project_patches,
    sequence_length,
    vit_param_count,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def make_image(height, width, channels=1):
    """Пиксель (r, c) — это [r*10+c, -(r*10+c), ...] по числу каналов."""
    return [
        [[(r * 10 + c) * (-1) ** ch for ch in range(channels)] for c in range(width)]
        for r in range(height)
    ]


# -------------------------------------------------------------- grid_shape
def test_grid_shape_of_vit_b16():
    assert grid_shape(224, 224, 16) == (14, 14)


def test_grid_shape_keeps_rows_and_cols_separate():
    """Портретная картинка даёт больше строк, чем столбцов, а не квадрат."""
    assert grid_shape(336, 224, 14) == (24, 16)


def test_grid_shape_rejects_indivisible_size():
    with pytest.raises(ValueError):
        grid_shape(225, 224, 16)


def test_grid_shape_rejects_nonpositive_patch():
    with pytest.raises(ValueError):
        grid_shape(224, 224, 0)


# --------------------------------------------------------- sequence_length
def test_sequence_length_adds_the_cls_token():
    assert sequence_length(224, 224, 16) == 197


def test_sequence_length_without_cls_is_just_the_patches():
    assert sequence_length(224, 224, 16, cls=False) == 196


def test_sequence_length_counts_registers():
    assert sequence_length(224, 224, 16, cls=True, registers=4) == 201


def test_sequence_length_grows_quadratically_with_resolution():
    """Удвоение стороны учетверяет число патчей — отсюда цена разрешения."""
    small = sequence_length(224, 224, 14, cls=False)
    big = sequence_length(448, 448, 14, cls=False)
    assert big == 4 * small


def test_finer_patch_gives_more_tokens():
    assert sequence_length(224, 224, 14, cls=False) > sequence_length(
        224, 224, 16, cls=False
    )


# ---------------------------------------------------------- extract_patches
def test_extract_patches_is_row_major_over_the_grid():
    patches = extract_patches(make_image(4, 4), 2)
    assert patches[0] == [0, 1, 10, 11]
    assert patches[1] == [2, 3, 12, 13]
    assert patches[2] == [20, 21, 30, 31]
    assert patches[3] == [22, 23, 32, 33]


def test_extract_patches_interleaves_channels_last():
    """Внутри патча порядок строка -> столбец -> канал, а не канал первым."""
    patches = extract_patches(make_image(2, 2, channels=2), 2)
    assert patches[0] == [0, 0, 1, -1, 10, -10, 11, -11]


def test_extract_patches_count_matches_the_grid():
    patches = extract_patches(make_image(6, 4), 2)
    rows, cols = grid_shape(6, 4, 2)
    assert len(patches) == rows * cols


def test_extract_patches_preserves_every_pixel_value():
    """Нарезка ничего не теряет и не дублирует: мультимножество совпадает."""
    image = make_image(4, 6, channels=3)
    original = sorted(v for row in image for px in row for v in px)
    carved = sorted(v for p in extract_patches(image, 2) for v in p)
    assert carved == original


def test_extract_patches_rejects_indivisible_image():
    with pytest.raises(ValueError):
        extract_patches(make_image(5, 4), 2)


# --------------------------------------------------------- project_patches
def test_project_patches_applies_the_matrix():
    W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert flat(project_patches([[2.0, 3.0]], W)) == APPROX([2.0, 3.0, 5.0])


def test_project_patches_shares_one_matrix_across_positions():
    """Одинаковые патчи в разных местах картинки дают одинаковые токены."""
    W = [[0.5, -1.0], [2.0, 0.25]]
    out = project_patches([[1.0, 2.0], [7.0, 7.0], [1.0, 2.0]], W)
    assert out[0] == APPROX(out[2])
    assert out[0] != APPROX(out[1])


def test_project_patches_is_linear_without_bias():
    W = [[0.3, -0.7, 1.1], [2.0, 0.5, -0.25]]
    a, b = [1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]
    out = project_patches([a, b, [x + y for x, y in zip(a, b)]], W)
    assert out[2] == APPROX([x + y for x, y in zip(out[0], out[1])])


def test_project_patches_adds_bias():
    W = [[1.0], [1.0]]
    assert flat(project_patches([[2.0]], W, bias=[10.0, -10.0])) == APPROX([12.0, -8.0])


def test_project_patches_rejects_wrong_patch_length():
    with pytest.raises(ValueError):
        project_patches([[1.0, 2.0, 3.0]], [[1.0, 0.0]])


# ------------------------------------------------- add_position_embeddings
def test_add_position_embeddings_sums_elementwise():
    got = add_position_embeddings([[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [5.0, -5.0]])
    assert flat(got) == APPROX([1.0, 1.0, 6.0, -4.0])


def test_position_embeddings_separate_identical_patches():
    """Два одинаковых куска неба перестают быть неразличимыми — вся суть шага."""
    same = [[2.0, 2.0], [2.0, 2.0]]
    got = add_position_embeddings(same, [[0.1, 0.0], [0.0, 0.1]])
    assert got[0] != APPROX(got[1])


def test_add_position_embeddings_does_not_mutate_input():
    tokens = [[1.0, 1.0]]
    add_position_embeddings(tokens, [[9.0, 9.0]])
    assert flat(tokens) == APPROX([1.0, 1.0])


def test_add_position_embeddings_rejects_length_mismatch():
    with pytest.raises(ValueError):
        add_position_embeddings([[1.0], [2.0]], [[0.0]])


# ----------------------------------------------------------------- mean_pool
def test_mean_pool_averages_each_coordinate():
    assert mean_pool([[1.0, 2.0], [3.0, 4.0]]) == APPROX([2.0, 3.0])


def test_mean_pool_ignores_token_order():
    """Pooling симметричен: перестановка патчей ничего не меняет."""
    tokens = [[1.0, 5.0], [-2.0, 0.0], [7.0, 3.0]]
    assert mean_pool(tokens) == APPROX(mean_pool(list(reversed(tokens))))


def test_mean_pool_of_identical_tokens_is_that_token():
    assert mean_pool([[4.0, -1.0]] * 9) == APPROX([4.0, -1.0])


def test_mean_pool_rejects_empty_sequence():
    with pytest.raises(ValueError):
        mean_pool([])


# ----------------------------------------------------------- vit_param_count
def test_vit_b16_is_about_86_million_params():
    assert vit_param_count(224, 16, 768, 12)["total"] == 85_798_656


def test_vit_param_count_parts_sum_to_total():
    parts = vit_param_count(224, 16, 768, 12)
    assert parts["total"] == sum(v for k, v in parts.items() if k != "total")


def test_transformer_blocks_dominate_the_parameter_budget():
    """Патч-эмбеддинг и позиции — мелочь, почти всё весят блоки."""
    parts = vit_param_count(224, 16, 768, 12)
    assert parts["blocks"] / parts["total"] > 0.9


def test_doubling_depth_doubles_the_block_params():
    a = vit_param_count(224, 16, 768, 12)["blocks"]
    b = vit_param_count(224, 16, 768, 24)["blocks"]
    assert b == 2 * a


def test_registers_cost_parameters_in_two_places():
    """Регистр — это и обучаемый вектор, и лишняя строка позиционной таблицы."""
    base = vit_param_count(224, 14, 768, 12)
    with_reg = vit_param_count(224, 14, 768, 12, registers=4)
    assert with_reg["total"] - base["total"] == 2 * 4 * 768


def test_resolution_changes_only_the_position_table():
    """Веса блоков от разрешения не зависят — растёт только таблица позиций."""
    small = vit_param_count(224, 16, 768, 12)
    big = vit_param_count(384, 16, 768, 12)
    assert big["blocks"] == small["blocks"]
    assert big["position"] > small["position"]
