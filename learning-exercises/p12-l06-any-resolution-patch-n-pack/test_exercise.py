"""Тесты к уроку «Any-resolution: patch-n'-pack и NaFlex». Правь exercise.py."""

import random

import pytest

from exercise import (
    block_diagonal_mask,
    drop_patches,
    fit_to_token_budget,
    mask_density,
    pack_batch,
    padded_batch_cost,
    patch_count,
    square_resize_cost,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


# ------------------------------------------------------------- patch_count
def test_patch_count_of_a_224_square():
    assert patch_count(224, 224, 14) == 256


def test_patch_count_of_a_tall_receipt():
    """600x1500 при патче 14 — это 42 * 107 токенов, и ни одного лишнего."""
    assert patch_count(600, 1500, 14) == 4494


def test_patch_count_crops_the_leftover_strip():
    """Остаток меньше патча отбрасывается, а не добивается паддингом."""
    assert patch_count(237, 224, 14) == patch_count(224, 224, 14)


def test_patch_count_rejects_a_zero_side():
    with pytest.raises(ValueError):
        patch_count(0, 224, 14)


# -------------------------------------------------------------- pack_batch
def test_pack_batch_lays_images_end_to_end():
    assert pack_batch([(224, 224), (336, 336)], 14) == [(0, 256), (256, 832)]


def test_packed_spans_have_no_gaps():
    spans = pack_batch([(224, 224), (336, 336), (448, 224), (140, 700)], 14)
    assert spans[0][0] == 0
    for left, right in zip(spans, spans[1:]):
        assert left[1] == right[0]


def test_packed_total_equals_the_sum_of_native_token_counts():
    """Ни одного токена сверх суммы — в этом и весь смысл упаковки."""
    sizes = [(224, 224), (336, 336), (448, 224)]
    spans = pack_batch(sizes, 14)
    assert spans[-1][1] == sum(patch_count(h, w, 14) for h, w in sizes)


def test_pack_batch_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        pack_batch([], 14)


def test_pack_batch_rejects_an_image_smaller_than_one_patch():
    with pytest.raises(ValueError):
        pack_batch([(224, 224), (10, 10)], 14)


# ------------------------------------------------------ block_diagonal_mask
def test_block_diagonal_mask_small_case():
    assert flat(block_diagonal_mask([(0, 2), (2, 3)])) == [1, 1, 0, 1, 1, 0, 0, 0, 1]


def test_mask_is_symmetric():
    mask = block_diagonal_mask(pack_batch([(28, 42), (42, 28)], 14))
    n = len(mask)
    for i in range(n):
        for j in range(n):
            assert mask[i][j] == mask[j][i]


def test_mask_diagonal_is_all_ones():
    """Патч всегда видит сам себя — иначе softmax внимания остался бы без якоря."""
    mask = block_diagonal_mask(pack_batch([(28, 28), (42, 42)], 14))
    assert all(mask[i][i] == 1 for i in range(len(mask)))


def test_mask_has_exactly_sum_of_squares_ones():
    """Задача 2 из урока: 256^2 + 576^2 + ... единиц, и ни одной больше."""
    spans = pack_batch([(224, 224), (336, 336), (392, 224)], 14)
    lengths = [end - start for start, end in spans]
    assert sum(flat(block_diagonal_mask(spans))) == sum(n * n for n in lengths)


def test_neighbours_in_the_pack_cannot_see_each_other():
    """Соседняя картинка в склейке — не контекст, а чужие пиксели."""
    spans = pack_batch([(28, 28), (28, 28)], 14)
    mask = block_diagonal_mask(spans)
    last_of_first = spans[0][1] - 1
    first_of_second = spans[1][0]
    assert mask[last_of_first][first_of_second] == 0


# ------------------------------------------------------------ mask_density
def test_mask_density_of_two_equal_blocks_is_one_half():
    assert mask_density([(0, 2), (2, 4)]) == APPROX(0.5)


def test_mask_density_of_a_single_image_is_one():
    assert mask_density([(0, 5)]) == APPROX(1.0)


def test_density_falls_like_one_over_the_batch_size():
    """Восемь одинаковых картинок: семь восьмых плотной маски посчитаны зря."""
    spans = pack_batch([(224, 224)] * 8, 14)
    assert mask_density(spans) == APPROX(1 / 8)


def test_mask_density_rejects_an_empty_pack():
    with pytest.raises(ValueError):
        mask_density([])


# ------------------------------------------------------- padded_batch_cost
def test_padded_cost_of_a_mixed_batch():
    assert padded_batch_cost([(224, 224), (336, 336)], 14) == 1152


def test_padding_equals_packing_only_when_sizes_match():
    sizes = [(336, 336)] * 3
    assert padded_batch_cost(sizes, 14) == pack_batch(sizes, 14)[-1][1]


def test_padding_is_never_cheaper_than_packing():
    for sizes in ([(224, 224), (336, 336)], [(140, 700), (700, 140), (224, 224)]):
        assert padded_batch_cost(sizes, 14) >= pack_batch(sizes, 14)[-1][1]


# ------------------------------------------------------- square_resize_cost
def test_square_resize_cost_of_two_images():
    assert square_resize_cost([(600, 1500), (224, 224)], 336, 14) == 1152


def test_square_resize_charges_the_same_for_any_shape():
    """Расписка 1:2.5 и квадратный кадр стоят одинаково — вот в чём подвох."""
    a = square_resize_cost([(600, 1500)], 336, 14)
    b = square_resize_cost([(336, 336)], 336, 14)
    assert a == b


def test_square_resize_cost_rejects_a_zero_side():
    with pytest.raises(ValueError):
        square_resize_cost([(224, 224)], 0, 14)


# ------------------------------------------------------------ drop_patches
def test_drop_patches_is_reproducible_for_the_same_seed():
    """Ради этого rng и передаётся аргументом, а не берётся из модуля."""
    a = drop_patches([(0, 100), (100, 250)], 0.5, random.Random(7))
    b = drop_patches([(0, 100), (100, 250)], 0.5, random.Random(7))
    assert a == b


def test_keeping_everything_leaves_the_pack_untouched():
    spans = pack_batch([(224, 224), (336, 336)], 14)
    assert drop_patches(spans, 1.0, random.Random(0)) == spans


def test_dropping_everything_empties_every_span():
    spans = pack_batch([(224, 224), (336, 336)], 14)
    assert drop_patches(spans, 0.0, random.Random(0)) == [(0, 0), (0, 0)]


def test_dropped_spans_stay_contiguous():
    """Дыр в упаковке быть не должно — маска строится по стыкам."""
    got = drop_patches([(0, 200), (200, 500), (500, 560)], 0.6, random.Random(3))
    assert got[0][0] == 0
    for left, right in zip(got, got[1:]):
        assert left[1] == right[0]


def test_drop_patches_rejects_a_probability_outside_the_unit_interval():
    with pytest.raises(ValueError):
        drop_patches([(0, 10)], 1.5, random.Random(0))


# ------------------------------------------------------- fit_to_token_budget
def test_budget_is_never_exceeded():
    for height, width in ((1920, 1080), (600, 1500), (4096, 4096), (2532, 1170)):
        h, w = fit_to_token_budget(height, width, 14, 1024)
        assert patch_count(h, w, 14) <= 1024


def test_an_image_that_already_fits_is_not_upscaled():
    """Апскейл не добавляет информации, а платить за него пришлось бы квадратично."""
    assert fit_to_token_budget(224, 224, 14, 4096) == (224, 224)


def test_both_sides_are_whole_patches():
    h, w = fit_to_token_budget(1920, 1080, 14, 1024)
    assert h % 14 == 0 and w % 14 == 0
    assert h >= 14 and w >= 14


def test_aspect_ratio_survives_the_shrink():
    """Иначе получится тот же squish, от которого мы и уходили."""
    for height, width in ((1920, 1080), (600, 1500), (2532, 1170)):
        h, w = fit_to_token_budget(height, width, 14, 2048)
        assert (h / w) == pytest.approx(height / width, rel=0.1)


def test_a_tiny_budget_still_returns_a_legal_image():
    h, w = fit_to_token_budget(4096, 4096, 14, 1)
    assert (h, w) == (14, 14)


def test_fit_to_token_budget_rejects_a_zero_budget():
    with pytest.raises(ValueError):
        fit_to_token_budget(224, 224, 14, 0)
