"""Тесты к уроку «Visual Autoregressive (VAR): предсказание следующего масштаба». Правь exercise.py."""

import random

import pytest

from exercise import (
    detokenize_multiscale,
    downsample,
    encode_grid,
    generate_scales,
    scale_causal_mask,
    scale_positions,
    tokenize_multiscale,
    upsample,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

IMG = 8
SCALES = (1, 2, 4, 8)
# кодбук с шагом 1/16 на отрезке [-1, 1]: один и тот же для всех масштабов
BOOK = [i / 16.0 - 1.0 for i in range(33)]
BOOKS = [BOOK] * len(SCALES)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


def gradient_image(size=IMG):
    """Гладкий градиент: низкие частоты сильные, высокие слабые."""
    return [[(x + y) / (2.0 * (size - 1)) for x in range(size)] for y in range(size)]


def constant_image(value=0.5, size=IMG):
    return [[value] * size for _ in range(size)]


def img_mse(a, b):
    return sum((va - vb) ** 2 for ra, rb in zip(a, b)
               for va, vb in zip(ra, rb)) / (len(a) * len(a[0]))


def recon_mse(img, num_scales):
    """Ошибка восстановления, если оставить первые num_scales масштабов."""
    scales = SCALES[:num_scales]
    tokens = tokenize_multiscale(img, BOOKS[:num_scales], scales)
    return img_mse(detokenize_multiscale(tokens, BOOKS[:num_scales], len(img)), img)


# --------------------------------------------------------------- downsample
def test_downsample_to_one_returns_the_mean():
    assert flat(downsample([[1.0, 3.0], [5.0, 7.0]], 1)) == APPROX([4.0])


def test_downsample_to_the_same_size_changes_nothing():
    assert flat(downsample([[1.0, 3.0], [5.0, 7.0]], 2)) == APPROX([1.0, 3.0, 5.0, 7.0])


def test_downsample_averages_each_block():
    img = [[1.0, 1.0, 9.0, 9.0],
           [1.0, 1.0, 9.0, 9.0],
           [0.0, 0.0, 4.0, 4.0],
           [0.0, 0.0, 4.0, 4.0]]
    assert flat(downsample(img, 2)) == APPROX([1.0, 9.0, 0.0, 4.0])


def test_downsample_rejects_a_size_that_does_not_divide():
    """8 на 3 не делится — резать остаток молча нельзя."""
    with pytest.raises(ValueError):
        downsample(gradient_image(), 3)


# ----------------------------------------------------------------- upsample
def test_upsample_repeats_the_single_value():
    assert flat(upsample([[2.0]], 2)) == APPROX([2.0, 2.0, 2.0, 2.0])


def test_upsample_repeats_each_cell_into_a_block():
    assert flat(upsample([[1.0, 2.0], [3.0, 4.0]], 4)) == APPROX(
        [1.0, 1.0, 2.0, 2.0,
         1.0, 1.0, 2.0, 2.0,
         3.0, 3.0, 4.0, 4.0,
         3.0, 3.0, 4.0, 4.0])


def test_upsample_rows_are_independent_objects():
    """Если строки блока — один и тот же список, правка одной портит все."""
    out = upsample([[1.0]], 3)
    out[0][0] = 99.0
    assert out[1][0] == APPROX(1.0)


def test_upsample_rejects_a_size_that_is_not_a_multiple():
    with pytest.raises(ValueError):
        upsample([[1.0, 2.0], [3.0, 4.0]], 5)


def test_downsample_undoes_upsample_exactly():
    """Обратный прогон обязан вернуть сетку как была — иначе пирамида поедет."""
    grid = [[0.25, -0.5], [0.75, 0.0]]
    assert flat(downsample(upsample(grid, 8), 2)) == APPROX(flat(grid))


# -------------------------------------------------------------- encode_grid
def test_encode_grid_picks_the_nearest_code():
    assert encode_grid([[0.1, 0.9]], [0.0, 0.5, 1.0]) == [[0, 2]]


def test_encode_grid_keeps_the_shape():
    tokens = encode_grid(gradient_image(4), BOOK)
    assert len(tokens) == 4 and all(len(row) == 4 for row in tokens)


def test_encode_grid_returns_indices_inside_the_codebook():
    tokens = encode_grid(gradient_image(), BOOK)
    assert all(isinstance(t, int) and 0 <= t < len(BOOK) for row in tokens for t in row)


# ------------------------------------------------ tokenize / detokenize
def test_tokenize_returns_one_grid_per_scale_with_the_right_shape():
    tokens = tokenize_multiscale(gradient_image(), BOOKS, SCALES)
    assert [len(t) for t in tokens] == list(SCALES)
    assert [len(t[0]) for t in tokens] == list(SCALES)


def test_reconstruction_keeps_the_original_resolution():
    """Ключ VAR: по грубому масштабу восстанавливается полное разрешение."""
    img = gradient_image()
    tokens = tokenize_multiscale(img, BOOKS[:1], SCALES[:1])
    recon = detokenize_multiscale(tokens, BOOKS[:1], IMG)
    assert len(recon) == IMG and all(len(row) == IMG for row in recon)


def test_the_coarsest_scale_alone_encodes_the_image_mean():
    """Один токен 1x1 — это «конспект» картинки, её среднее значение."""
    img = gradient_image()
    recon = detokenize_multiscale(tokenize_multiscale(img, BOOKS[:1], SCALES[:1]),
                                 BOOKS[:1], IMG)
    mean = sum(flat(img)) / (IMG * IMG)
    assert recon[0][0] == pytest.approx(mean, abs=1.0 / 32)


def test_each_extra_scale_lowers_the_reconstruction_error():
    """Residual VQ: масштаб k снимает то, что масштабы 1..k-1 не объяснили."""
    img = gradient_image()
    errors = [recon_mse(img, k) for k in (1, 2, 3, 4)]
    assert errors[0] > errors[1] > errors[2] > errors[3]


def test_a_flat_image_leaves_nothing_for_the_finer_scales():
    """Остаток после первого масштаба нулевой — дальше кодируются нули."""
    tokens = tokenize_multiscale(constant_image(0.5), BOOKS, SCALES)
    zero_code = BOOK.index(0.0)
    assert all(t == zero_code for grid in tokens[1:] for row in grid for t in row)


def test_tokenize_does_not_mutate_the_image():
    img = gradient_image()
    before = flat(img)
    tokenize_multiscale(img, BOOKS, SCALES)
    assert flat(img) == APPROX(before)


def test_detokenize_sums_the_scales_rather_than_replacing_them():
    """Два масштаба с кодом 0.5 каждый дают 1.0, а не 0.5."""
    half = BOOK.index(0.5)
    recon = detokenize_multiscale([[[half]], [[half, half], [half, half]]],
                                 [BOOK, BOOK], 2)
    assert flat(recon) == APPROX([1.0] * 4)


# ----------------------------------------------------------- scale_positions
def test_scale_positions_lists_scale_row_and_column():
    assert scale_positions((1, 2)) == [(0, 0, 0), (1, 0, 0), (1, 0, 1),
                                       (1, 1, 0), (1, 1, 1)]


def test_scale_positions_length_is_the_total_token_count():
    assert len(scale_positions(SCALES)) == sum(s * s for s in SCALES)


def test_scale_positions_never_go_back_to_a_coarser_scale():
    indices = [k for k, _, _ in scale_positions(SCALES)]
    assert indices == sorted(indices)


# -------------------------------------------------------- scale_causal_mask
def test_the_first_scale_has_nothing_to_attend_to():
    mask = scale_causal_mask(SCALES)
    assert not any(mask[0])


def test_every_token_attends_to_all_tokens_of_all_earlier_scales():
    mask = scale_causal_mask(SCALES)
    positions = scale_positions(SCALES)
    for row, (k, _, _) in zip(mask, positions):
        assert sum(row) == sum(s * s for s in SCALES[:k])


def test_no_token_attends_to_its_own_scale():
    """Параллельно внутри масштаба: значений соседей ещё не существует."""
    mask = scale_causal_mask(SCALES)
    positions = scale_positions(SCALES)
    for i, (ki, _, _) in enumerate(positions):
        assert not any(mask[i][j] for j, (kj, _, _) in enumerate(positions) if kj == ki)


def test_the_mask_is_not_symmetric():
    """Обычная GPT-маска симметричной тоже не бывает — направление важно."""
    mask = scale_causal_mask((1, 2))
    assert mask[1][0] is True
    assert mask[0][1] is False


def test_the_mask_is_square_and_matches_the_position_count():
    mask = scale_causal_mask(SCALES)
    n = len(scale_positions(SCALES))
    assert len(mask) == n and all(len(row) == n for row in mask)


# --------------------------------------------------------- generate_scales
def test_generate_scales_produces_one_grid_per_scale():
    drawn = generate_scales(lambda k, prev: [0.0, 1.0], SCALES, random.Random(0))
    assert [len(g) for g in drawn] == list(SCALES)
    assert [len(g[0]) for g in drawn] == list(SCALES)


def test_generate_scales_calls_the_predictor_once_per_scale():
    """Вся выгода VAR: 4 прохода на 85 токенов, а не 85 проходов."""
    calls = []

    def predictor(k, prev):
        calls.append(k)
        return [0.0, 1.0]

    generate_scales(predictor, SCALES, random.Random(0))
    assert calls == [0, 1, 2, 3]


def test_generate_scales_shows_the_predictor_only_the_earlier_scales():
    seen = []

    def predictor(k, prev):
        seen.append([len(g) for g in prev])
        return [0.0, 1.0]

    generate_scales(predictor, SCALES, random.Random(0))
    assert seen == [[], [1], [1, 2], [1, 2, 4]]


def test_a_deterministic_predictor_gives_a_deterministic_pyramid():
    drawn = generate_scales(lambda k, prev: [0.0, 0.0, 1.0], (1, 2), random.Random(5))
    assert drawn == [[[2]], [[2, 2], [2, 2]]]


def test_generate_scales_is_reproducible_for_a_given_seed():
    uniform = lambda k, prev: [0.25] * 4
    a = generate_scales(uniform, SCALES, random.Random(11))
    b = generate_scales(uniform, SCALES, random.Random(11))
    assert a == b


def test_generated_pyramid_decodes_to_a_full_resolution_image():
    """Пирамида токенов -> сумма масштабов -> картинка исходного размера."""
    drawn = generate_scales(lambda k, prev: [0.25] * 4, SCALES, random.Random(3))
    books = [[0.0, 0.1, 0.2, 0.3]] * len(SCALES)
    recon = detokenize_multiscale(drawn, books, IMG)
    assert len(recon) == IMG and all(len(row) == IMG for row in recon)
