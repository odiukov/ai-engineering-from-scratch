"""Тесты к уроку «Разбор архитектур открытых моделей». Правь exercise.py."""

import math

import pytest

from exercise import (
    kv_cache_bytes,
    moe_block,
    param_count,
    rms_norm,
    rope_rotate,
    softmax,
    swiglu_mlp,
    top_k_route,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

LLAMA3_8B = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ----------------------------------------------------------------- rms_norm
def test_rms_norm_scales_to_unit_root_mean_square():
    out = rms_norm([3.0, 4.0], [1.0, 1.0], 0.0)
    assert math.sqrt(sum(v * v for v in out) / 2) == pytest.approx(1.0)


def test_rms_norm_is_invariant_to_input_scale():
    """Умножить вход на константу — выход не меняется."""
    a = rms_norm([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.0)
    b = rms_norm([10.0, 20.0, 30.0], [1.0, 1.0, 1.0], 0.0)
    assert a == pytest.approx(b)


def test_rms_norm_does_not_subtract_the_mean():
    """Отличие от LayerNorm: постоянный сдвиг входа виден в выходе."""
    a = rms_norm([1.0, 3.0], [1.0, 1.0], 0.0)
    b = rms_norm([0.0, 2.0], [1.0, 1.0], 0.0)
    assert a != pytest.approx(b)


def test_rms_norm_output_mean_is_not_forced_to_zero():
    out = rms_norm([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0)
    assert sum(out) / 3 == pytest.approx(1.0)


def test_rms_norm_applies_gamma_per_channel():
    assert rms_norm([1.0, 1.0], [2.0, 5.0], 0.0) == pytest.approx([2.0, 5.0])


def test_rms_norm_eps_saves_an_all_zero_vector():
    """Без eps здесь было бы деление на ноль."""
    assert rms_norm([0.0, 0.0], [1.0, 1.0]) == pytest.approx([0.0, 0.0])


# -------------------------------------------------------------- rope_rotate
def test_rope_at_position_zero_is_identity():
    v = [1.0, 2.0, 3.0, 4.0]
    assert rope_rotate(v, 0) == pytest.approx(v)


def test_rope_preserves_the_norm_of_each_pair():
    """Поворот — это поворот: длина не меняется."""
    v = [1.0, 2.0, 3.0, 4.0]
    out = rope_rotate(v, 17, theta=100.0)
    assert dot(out, out) == pytest.approx(dot(v, v))


def test_rope_dot_product_depends_only_on_relative_position():
    """Главное свойство RoPE: q на позиции 7 и k на позиции 4 видят «3»."""
    q = [1.0, 2.0, 3.0, 4.0]
    k = [0.5, -1.0, 2.0, 0.3]
    far = dot(rope_rotate(q, 7), rope_rotate(k, 4))
    near = dot(rope_rotate(q, 3), rope_rotate(k, 0))
    assert far == pytest.approx(near, abs=1e-9)


def test_rope_actually_moves_the_vector():
    v = [1.0, 0.0]
    assert rope_rotate(v, 1, theta=1.0) == pytest.approx([math.cos(1.0), math.sin(1.0)])


def test_rope_rejects_an_odd_dimension():
    with pytest.raises(ValueError):
        rope_rotate([1.0, 2.0, 3.0], 1)


def test_rope_low_frequency_pairs_rotate_slower():
    """Старшая пара почти стоит на месте — она кодирует далёкие расстояния."""
    v = [1.0, 0.0, 1.0, 0.0]
    out = rope_rotate(v, 1, theta=10000.0)
    moved_first = abs(out[1])
    moved_last = abs(out[3])
    assert moved_first > moved_last


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    assert sum(softmax([1.0, 2.0, 3.0])) == APPROX(1.0)


def test_softmax_survives_huge_logits():
    assert softmax([0.0, 1000.0]) == pytest.approx([0.0, 1.0])


def test_softmax_of_equal_logits_is_uniform():
    assert softmax([5.0] * 4) == pytest.approx([0.25] * 4)


# --------------------------------------------------------------- swiglu_mlp
def test_swiglu_gate_at_zero_kills_the_branch():
    """silu(0) = 0, значит нулевой gate закрывает MLP независимо от up."""
    assert swiglu_mlp([1.0], [[0.0]], [[99.0]], [[1.0]]) == APPROX([0.0])


def test_swiglu_matches_the_hand_computed_value():
    # gate = 1, up = 2, silu(1) = 1 * sigmoid(1)
    expected = 1.0 * (1.0 / (1.0 + math.exp(-1.0))) * 2.0
    assert swiglu_mlp([1.0], [[1.0]], [[2.0]], [[1.0]]) == pytest.approx([expected])


def test_swiglu_survives_large_negative_gate():
    """Наивная sigmoid(z)=1/(1+exp(-z)) падает с OverflowError на z=-1000."""
    assert swiglu_mlp([1.0], [[-1000.0]], [[1.0]], [[1.0]]) == pytest.approx([0.0])


def test_swiglu_is_not_linear():
    """Гейтинг — вся суть SwiGLU: удвоение входа не удваивает выход."""
    one = swiglu_mlp([1.0], [[1.0]], [[1.0]], [[1.0]])[0]
    two = swiglu_mlp([2.0], [[1.0]], [[1.0]], [[1.0]])[0]
    assert two != pytest.approx(2 * one)


# -------------------------------------------------------------- top_k_route
def test_route_picks_the_highest_logits():
    indices, _ = top_k_route([0.0, 5.0, 1.0, 4.0], 2)
    assert indices == [1, 3]


def test_route_weights_sum_to_one():
    _, weights = top_k_route([0.0, 5.0, 1.0, 4.0], 2)
    assert sum(weights) == APPROX(1.0)


def test_route_normalizes_over_the_selected_experts_only():
    """Softmax по всем логитам дал бы сумму меньше единицы — это баг."""
    _, weights = top_k_route([0.0, 0.0, 100.0, 100.0], 2)
    assert weights == pytest.approx([0.5, 0.5])


def test_route_with_k_equal_to_everything_is_a_plain_softmax():
    logits = [1.0, 2.0, 3.0]
    indices, weights = top_k_route(logits, 3)
    assert indices == [0, 1, 2]
    assert weights == pytest.approx(softmax(logits))


def test_route_breaks_ties_by_lowest_index():
    indices, _ = top_k_route([1.0, 1.0, 1.0], 1)
    assert indices == [0]


def test_route_rejects_k_larger_than_the_expert_pool():
    with pytest.raises(ValueError):
        top_k_route([1.0, 2.0], 3)


# ---------------------------------------------------------------- moe_block
def test_moe_uses_only_the_selected_expert():
    """Невыбранный эксперт может быть каким угодно — выход тот же."""
    good = (identity(1), identity(1), identity(1))
    junk_a = ([[7.0]], [[7.0]], [[7.0]])
    junk_b = ([[-3.0]], [[11.0]], [[0.5]])
    a = moe_block([1.0], [good, junk_a], [10.0, 0.0], 1)
    b = moe_block([1.0], [good, junk_b], [10.0, 0.0], 1)
    assert a == pytest.approx(b)


def test_moe_with_all_experts_and_equal_logits_is_their_average():
    e0 = (identity(1), identity(1), [[1.0]])
    e1 = (identity(1), identity(1), [[3.0]])
    single_0 = swiglu_mlp([1.0], *e0)[0]
    single_1 = swiglu_mlp([1.0], *e1)[0]
    mixed = moe_block([1.0], [e0, e1], [0.0, 0.0], 2)[0]
    assert mixed == pytest.approx((single_0 + single_1) / 2)


def test_moe_with_k_one_equals_calling_that_expert_directly():
    e0 = (identity(1), identity(1), [[2.0]])
    e1 = (identity(1), identity(1), [[5.0]])
    assert moe_block([1.0], [e0, e1], [0.0, 9.0], 1) == pytest.approx(swiglu_mlp([1.0], *e1))


# -------------------------------------------------------------- param_count
def test_llama3_8b_total_matches_the_published_size():
    assert param_count(LLAMA3_8B)["total"] == 8_030_261_248


def test_components_add_up_to_the_total():
    counts = param_count(LLAMA3_8B)
    parts = sum(v for k, v in counts.items() if k != "total")
    assert parts == counts["total"]


def test_gqa_shrinks_the_attention_block():
    """32 Q-головы на 8 KV-голов дешевле, чем полная MHA."""
    mha = dict(LLAMA3_8B, num_key_value_heads=32)
    assert param_count(LLAMA3_8B)["attention"] < param_count(mha)["attention"]


def test_tied_embeddings_remove_the_output_head():
    tied = dict(LLAMA3_8B, tie_word_embeddings=True)
    counts = param_count(tied)
    assert counts["head"] == 0
    assert counts["total"] == param_count(LLAMA3_8B)["total"] - 128256 * 4096


def test_mlp_uses_three_matrices_not_two():
    """SwiGLU это gate + up + down; двух матриц не хватит."""
    counts = param_count(LLAMA3_8B)
    assert counts["mlp"] == 3 * 4096 * 14336 * 32


# ----------------------------------------------------------- kv_cache_bytes
def test_llama3_8b_kv_cache_at_full_context():
    assert kv_cache_bytes(LLAMA3_8B, 131072) == 17_179_869_184


def test_kv_cache_beats_the_weights_at_long_context():
    """17 ГБ кэша против 16 ГБ весов в BF16 — вот почему все ушли на GQA."""
    weights_bytes = param_count(LLAMA3_8B)["total"] * 2
    assert kv_cache_bytes(LLAMA3_8B, 131072) > weights_bytes


def test_gqa_gives_a_fourfold_kv_cache_reduction():
    mha = dict(LLAMA3_8B, num_key_value_heads=32)
    assert kv_cache_bytes(mha, 4096) == 4 * kv_cache_bytes(LLAMA3_8B, 4096)


def test_fp8_halves_the_kv_cache():
    bf16 = kv_cache_bytes(LLAMA3_8B, 8192, bytes_per_elem=2)
    fp8 = kv_cache_bytes(LLAMA3_8B, 8192, bytes_per_elem=1)
    assert fp8 * 2 == bf16


def test_kv_cache_is_linear_in_sequence_length():
    assert kv_cache_bytes(LLAMA3_8B, 2048) * 3 == kv_cache_bytes(LLAMA3_8B, 6144)
