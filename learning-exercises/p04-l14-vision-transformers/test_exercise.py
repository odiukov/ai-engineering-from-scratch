"""Тесты к уроку «Vision Transformers». Правь exercise.py."""

import math

import pytest

from exercise import (
    add_cls_and_positions,
    layer_norm,
    patch_embed,
    patchify,
    prenorm_residual,
    scaled_dot_product_attention,
    softmax,
    unpatchify,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в вектор."""
    return [x for row in M for x in row]


def make_image(height, width):
    """Картинка, у которой каждый пиксель уникален: перепутанный порядок сразу видно."""
    return [[float(r * width + c) for c in range(width)] for r in range(height)]


# ---------------------------------------------------------------- patchify
def test_patchify_walks_the_patch_grid_row_by_row():
    image = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert patchify(image, 2) == [[1, 2, 5, 6], [3, 4, 7, 8]]


def test_patchify_flattens_each_patch_row_by_row():
    """Патч 2x2 из левого верхнего угла разворачивается как [строка0, строка1]."""
    image = make_image(4, 4)
    assert patchify(image, 2)[0] == [0.0, 1.0, 4.0, 5.0]


def test_patchify_count_is_height_over_p_times_width_over_p():
    """64x64 при p=16 даёт ровно 16 патчей — сетка из урока."""
    patches = patchify(make_image(64, 64), 16)
    assert len(patches) == (64 // 16) * (64 // 16) == 16
    assert all(len(p) == 16 * 16 for p in patches)


def test_patchify_handles_non_square_images():
    patches = patchify(make_image(4, 6), 2)
    assert len(patches) == (4 // 2) * (6 // 2) == 6


def test_patchify_rejects_size_not_divisible_by_patch():
    """В ViT это assert: молча отрезать хвост картинки нельзя."""
    with pytest.raises(ValueError):
        patchify(make_image(3, 4), 2)


# -------------------------------------------------------------- unpatchify
def test_unpatchify_rebuilds_a_known_image():
    assert unpatchify([[1, 2, 5, 6], [3, 4, 7, 8]], 2, 2, 4) == [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]


def test_unpatchify_inverts_patchify_on_a_non_square_image():
    """Разбиение на патчи обратимо — иначе MAE не смог бы собрать реконструкцию."""
    image = make_image(6, 8)
    assert unpatchify(patchify(image, 2), 2, 6, 8) == image


def test_unpatchify_inverts_patchify_on_the_lesson_grid():
    image = make_image(64, 64)
    assert unpatchify(patchify(image, 16), 16, 64, 64) == image


# ------------------------------------------------------------- patch_embed
def test_patch_embed_returns_one_token_per_patch_of_length_dim():
    image = make_image(64, 64)
    dim, patch_len = 8, 16 * 16
    W = [[0.01] * patch_len for _ in range(dim)]
    b = [0.0] * dim
    tokens = patch_embed(image, 16, W, b)
    assert len(tokens) == len(patchify(image, 16)) == 16
    assert all(len(t) == dim for t in tokens)


def test_patch_embed_applies_weights_and_bias():
    image = [[1.0, 2.0], [3.0, 4.0]]
    W = [[1, 0, 0, 0], [0, 0, 0, 1]]
    assert flat(patch_embed(image, 2, W, [0.0, 10.0])) == APPROX([1.0, 14.0])


def test_patch_embed_with_identity_weights_returns_the_patches():
    """Единичная W и нулевой b — проекция ничего не меняет, токен = патч."""
    image = make_image(4, 4)
    identity = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    tokens = patch_embed(image, 2, identity, [0.0] * 4)
    assert flat(tokens) == APPROX(flat(patchify(image, 2)))


def test_patch_embed_rejects_indivisible_image():
    W = [[1.0] * 4]
    with pytest.raises(ValueError):
        patch_embed(make_image(5, 4), 2, W, [0.0])


# --------------------------------------------------- add_cls_and_positions
def test_add_cls_grows_the_sequence_by_exactly_one():
    tokens = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    pos = [[0.0, 0.0]] * 4
    assert len(add_cls_and_positions(tokens, [9.0, 9.0], pos)) == len(tokens) + 1


def test_add_cls_puts_the_class_token_first():
    tokens = [[1.0, 1.0]]
    out = add_cls_and_positions(tokens, [9.0, 8.0], [[0.0, 0.0], [0.0, 0.0]])
    assert out[0] == APPROX([9.0, 8.0])


def test_add_cls_adds_positions_after_prepending():
    """Позиция 0 достаётся [CLS], позиция 1 — первому патчу, а не наоборот."""
    out = add_cls_and_positions([[1.0, 2.0]], [0.0, 0.0], [[10.0, 20.0], [0.5, 0.5]])
    assert flat(out) == APPROX([10.0, 20.0, 1.5, 2.5])


def test_add_cls_rejects_pos_embed_of_wrong_length():
    """Тот самый баг переноса ViT: патчей стало больше, pos_embed остался старый."""
    tokens = [[1.0], [2.0], [3.0]]
    with pytest.raises(ValueError):
        add_cls_and_positions(tokens, [0.0], [[0.0]] * len(tokens))


def test_add_cls_does_not_mutate_the_input_tokens():
    tokens = [[1.0, 2.0]]
    add_cls_and_positions(tokens, [0.0, 0.0], [[5.0, 5.0], [5.0, 5.0]])
    assert tokens == [[1.0, 2.0]]


# ----------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, -2.0, 0.5, 3.0])) == pytest.approx(1.0, abs=1e-12)


def test_softmax_is_uniform_for_equal_scores():
    assert softmax([7.0] * 4) == APPROX([0.25] * 4)


def test_softmax_survives_huge_logits():
    """Без вычитания максимума math.exp(1000) — это OverflowError."""
    probs = softmax([1000.0, 1001.0])
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert probs[1] > probs[0]


def test_softmax_is_invariant_to_shifting_all_logits():
    base = [0.3, -1.2, 2.0]
    shifted = [x + 100.0 for x in base]
    assert softmax(shifted) == APPROX(softmax(base))


# ------------------------------------------ scaled_dot_product_attention
def test_attention_every_row_of_weights_sums_to_one():
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    K = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    V = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    _, weights = scaled_dot_product_attention(Q, K, V)
    assert len(weights) == len(Q)
    for row in weights:
        assert sum(row) == pytest.approx(1.0, abs=1e-12)


def test_attention_gives_the_matching_key_the_largest_weight():
    """Запрос совпал с ключом 1, остальные ортогональны — туда и смотрим."""
    Q = [[0.0, 3.0, 0.0]]
    K = [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]
    V = [[1.0], [2.0], [3.0]]
    _, weights = scaled_dot_product_attention(Q, K, V)
    assert weights[0][1] == max(weights[0])
    assert weights[0][1] > weights[0][0]


def test_attention_with_identical_keys_splits_weight_evenly():
    """Различить одинаковые ключи невозможно — остаётся усреднение по 1/n."""
    K = [[1.0, 0.0]] * 4
    V = [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]]
    out, weights = scaled_dot_product_attention([[1.0, 0.0]], K, V)
    assert weights[0] == APPROX([0.25] * 4)
    assert flat(out) == APPROX([4.0, 4.0])


def test_attention_output_is_a_convex_combination_of_values():
    """Веса неотрицательны и дают в сумме 1, поэтому выход не выходит за пределы V."""
    Q = [[1.0, 2.0], [-1.0, 0.5]]
    K = [[0.0, 1.0], [2.0, -1.0], [1.0, 1.0]]
    V = [[1.0, 10.0], [-4.0, 2.0], [3.0, 7.0]]
    out, _ = scaled_dot_product_attention(Q, K, V)
    for row in out:
        for j, value in enumerate(row):
            column = [v[j] for v in V]
            assert min(column) - 1e-12 <= value <= max(column) + 1e-12


def test_attention_divides_by_sqrt_of_dk_not_by_dk():
    """d_k = 4: сырой скор 4 даёт 4/sqrt(4) = 2, а деление на d_k дало бы 1."""
    Q = [[1.0, 1.0, 1.0, 1.0]]
    K = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    V = [[1.0], [0.0]]
    _, weights = scaled_dot_product_attention(Q, K, V)
    assert weights[0][0] == pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-12)


# -------------------------------------------------------------- layer_norm
def test_layer_norm_output_has_zero_mean_and_unit_variance():
    vec = [1.0, 2.0, 3.0, 10.0]
    out = layer_norm(vec, [1.0] * 4, [0.0] * 4)
    mean = sum(out) / len(out)
    var = sum((x - mean) ** 2 for x in out) / len(out)
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert var == pytest.approx(1.0, abs=1e-5)


def test_layer_norm_is_invariant_to_shift_and_scale_of_the_input():
    vec = [1.0, 2.0, 3.0]
    scaled = [5.0 * x + 100.0 for x in vec]
    got = layer_norm(scaled, [1.0] * 3, [0.0] * 3)
    want = layer_norm(vec, [1.0] * 3, [0.0] * 3)
    assert got == pytest.approx(want, abs=1e-5)


def test_layer_norm_applies_gamma_and_beta():
    """gamma растягивает нормированный вектор, beta сдвигает."""
    vec = [1.0, 2.0, 3.0]
    plain = layer_norm(vec, [1.0] * 3, [0.0] * 3)
    styled = layer_norm(vec, [2.0] * 3, [7.0] * 3)
    assert styled == APPROX([2.0 * x + 7.0 for x in plain])


def test_layer_norm_uses_population_variance():
    """Делить надо на n, как nn.LayerNorm, а не на n-1, как statistics.variance."""
    vec = [1.0, 2.0, 3.0]
    out = layer_norm(vec, [1.0] * 3, [0.0] * 3, eps=0.0)
    assert out[2] == pytest.approx(math.sqrt(1.5), abs=1e-9)


def test_layer_norm_survives_a_constant_vector():
    """Дисперсия ровно 0 — спасает только eps."""
    assert layer_norm([5.0, 5.0, 5.0], [1.0] * 3, [0.0] * 3) == APPROX([0.0] * 3)


# --------------------------------------------------------- prenorm_residual
def test_prenorm_residual_with_zero_sublayer_returns_x_unchanged():
    """Вот чем pre-LN отличается от post-LN: post-LN отнормировал бы сам x."""
    x = [1.0, 2.0, 3.0]
    zero = lambda v: [0.0] * len(v)
    assert prenorm_residual(x, [1.0] * 3, [0.0] * 3, zero) == APPROX(x)


def test_prenorm_residual_adds_the_sublayer_output_to_x():
    x = [1.0, 2.0, 3.0]
    ones = lambda v: [1.0] * len(v)
    assert prenorm_residual(x, [1.0] * 3, [0.0] * 3, ones) == APPROX([2.0, 3.0, 4.0])


def test_prenorm_residual_feeds_the_sublayer_the_normalised_vector():
    """Под-слой видит LN(x), а не сырой x."""
    x = [1.0, 2.0, 3.0]
    seen = []
    prenorm_residual(x, [1.0] * 3, [0.0] * 3, lambda v: seen.append(list(v)) or v)
    assert seen[0] == APPROX(layer_norm(x, [1.0] * 3, [0.0] * 3))


def test_prenorm_residual_does_not_mutate_x():
    x = [1.0, 2.0, 3.0]
    prenorm_residual(x, [1.0] * 3, [0.0] * 3, lambda v: v)
    assert x == [1.0, 2.0, 3.0]
