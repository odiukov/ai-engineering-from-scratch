"""Тесты к уроку «BLIP-2 и Q-Former как мост между модальностями». Правь exercise.py."""

import math

import pytest

from exercise import (
    cross_attention,
    linear_project,
    pick_bridge,
    qformer_forward,
    scaled_dot_attention,
    softmax,
    top_patches_per_query,
    visual_token_budget,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)
ROUGH = lambda x: pytest.approx(x, abs=1e-4)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def ramp(n, dim, step=0.1):
    """Детерминированные векторы: никакого глобального random в тестах."""
    return [[math.sin(i * dim + d) * step for d in range(dim)] for i in range(n)]


# ------------------------------------------------------------------ softmax
def test_softmax_of_equal_scores_is_uniform():
    assert softmax([0.0, 0.0, 0.0, 0.0]) == APPROX([0.25] * 4)


def test_softmax_sums_to_one():
    assert sum(softmax([3.0, -1.0, 0.5, 7.0])) == APPROX(1.0)


def test_softmax_is_shift_invariant():
    """Вычитание максимума — не приближение, а точное тождество."""
    xs = [2.0, 1.0, 0.0]
    assert softmax([x + 100.0 for x in xs]) == APPROX(softmax(xs))


def test_softmax_survives_huge_scores():
    """Наивный math.exp(1000) падает с OverflowError."""
    got = softmax([1000.0, 999.0])
    assert sum(got) == APPROX(1.0)
    assert got[0] > got[1]


def test_softmax_rejects_empty_input():
    with pytest.raises(ValueError):
        softmax([])


# ------------------------------------------------------- scaled_dot_attention
def test_scaled_dot_attention_divides_by_sqrt_dim():
    """Без деления на sqrt(2) веса были бы [0.731, 0.269], а не [0.670, 0.330]."""
    context, weights = scaled_dot_attention(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
    )
    assert weights == ROUGH([0.6697615, 0.3302385])
    assert context == ROUGH([0.6697615, 0.3302385])


def test_identical_keys_give_uniform_attention():
    """Нечего различать — внимание размазывается ровно."""
    _, weights = scaled_dot_attention([1.0, 1.0], [[0.5, 0.5]] * 4, [[1.0]] * 4)
    assert weights == APPROX([0.25] * 4)


def test_one_dominant_key_makes_attention_nearly_one_hot():
    context, weights = scaled_dot_attention(
        [10.0, 0.0], [[10.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        [[7.0], [0.0], [0.0]],
    )
    assert weights[0] > 0.999
    assert context == ROUGH([7.0 * weights[0]])


def test_context_is_a_convex_combination_of_values():
    """Выход не может выйти за диапазон values ни по одной координате."""
    values = [[1.0, -5.0], [3.0, 2.0], [-2.0, 0.0]]
    context, _ = scaled_dot_attention([0.3, -0.7], ramp(3, 2), values)
    for d in range(2):
        column = [v[d] for v in values]
        assert min(column) <= context[d] <= max(column)


def test_scaled_dot_attention_rejects_mismatched_keys_and_values():
    with pytest.raises(ValueError):
        scaled_dot_attention([1.0], [[1.0], [2.0]], [[1.0]])


# ---------------------------------------------------------- cross_attention
def test_cross_attention_output_count_follows_the_queries():
    """Суть Q-Former: 8 запросов дают 8 токенов и при 16, и при 200 патчах."""
    queries = ramp(8, 4)
    for num_patches in (16, 200):
        outputs, attn = cross_attention(queries, ramp(num_patches, 4), ramp(num_patches, 6))
        assert len(outputs) == 8
        assert len(attn) == 8
        assert len(attn[0]) == num_patches


def test_cross_attention_output_dim_comes_from_values():
    outputs, _ = cross_attention(ramp(3, 4), ramp(10, 4), ramp(10, 7))
    assert all(len(o) == 7 for o in outputs)


def test_every_attention_row_sums_to_one():
    _, attn = cross_attention(ramp(4, 3), ramp(9, 3), ramp(9, 3))
    for row in attn:
        assert sum(row) == APPROX(1.0)


def test_identical_queries_produce_identical_outputs():
    q = [0.4, -0.2, 1.0]
    outputs, _ = cross_attention([q, [9.0, 9.0, 9.0], q], ramp(6, 3), ramp(6, 2))
    assert outputs[0] == APPROX(outputs[2])
    assert outputs[0] != APPROX(outputs[1])


def test_cross_attention_rejects_empty_query_set():
    with pytest.raises(ValueError):
        cross_attention([], ramp(4, 2), ramp(4, 2))


# ---------------------------------------------------------- linear_project
def test_linear_project_applies_the_matrix():
    W = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert flat(linear_project([[1.0, 2.0]], W)) == APPROX([1.0, 2.0, 3.0])


def test_linear_project_adds_bias():
    assert flat(linear_project([[2.0]], [[3.0]], bias=[-1.0])) == APPROX([5.0])


def test_linear_project_changes_the_dimension_not_the_count():
    """Мост меняет размерность токена, но не их число."""
    tokens = ramp(32, 8)
    out = linear_project(tokens, ramp(24, 8))
    assert len(out) == 32
    assert len(out[0]) == 24


def test_linear_project_rejects_wrong_token_length():
    with pytest.raises(ValueError):
        linear_project([[1.0, 2.0, 3.0]], [[1.0, 0.0]])


# --------------------------------------------------------- qformer_forward
def test_qformer_forward_shape_is_set_by_queries_and_projection():
    tokens = qformer_forward(ramp(64, 8), ramp(8, 8), ramp(24, 8))
    assert len(tokens) == 8
    assert len(tokens[0]) == 24


def test_qformer_forward_shape_does_not_depend_on_patch_count():
    """256 патчей или 1024 — на выходе всё те же 8 токенов."""
    queries, W = ramp(8, 8), ramp(24, 8)
    a = qformer_forward(ramp(64, 8), queries, W)
    b = qformer_forward(ramp(256, 8), queries, W)
    assert len(a) == len(b) == 8
    assert flat(a) != ROUGH(flat(b))  # содержимое всё-таки другое


def test_qformer_forward_equals_attention_then_projection():
    patches, queries, W = ramp(20, 6), ramp(4, 6), ramp(9, 6)
    outputs, _ = cross_attention(queries, patches, patches)
    assert flat(qformer_forward(patches, queries, W)) == APPROX(
        flat(linear_project(outputs, W))
    )


def test_qformer_forward_uses_patches_as_both_keys_and_values():
    """K и V — один и тот же замороженный выход ViT, отдельного V-источника нет."""
    patches = [[1.0, 0.0], [1.0, 0.0]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    tokens = qformer_forward(patches, [[1.0, 0.0]], W)
    assert tokens[0] == ROUGH([1.0, 0.0])


# ---------------------------------------------------- top_patches_per_query
def test_top_patches_are_sorted_by_descending_weight():
    assert top_patches_per_query([[0.1, 0.7, 0.2]], k=2) == [[1, 2]]


def test_top_patches_breaks_ties_by_lower_index():
    assert top_patches_per_query([[0.5, 0.5, 0.0]], k=2) == [[0, 1]]


def test_top_patches_handles_every_query_row():
    attn = [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]
    assert top_patches_per_query(attn, k=1) == [[0], [2]]


def test_top_patches_rejects_k_larger_than_the_patch_count():
    with pytest.raises(ValueError):
        top_patches_per_query([[0.5, 0.5]], k=3)


# --------------------------------------------------- visual_token_budget
def test_qformer_is_cheaper_than_mlp_when_queries_are_fewer():
    b = visual_token_budget(1, 256, 32)
    assert b["mlp"] == 256
    assert b["qformer"] == 32
    assert b["compression"] == APPROX(8.0)


def test_token_budget_scales_linearly_with_frame_count():
    """60 кадров видео: 34560 токенов через MLP против 1920 через Q-Former."""
    b = visual_token_budget(60, 576, 32)
    assert b["mlp"] == 34560
    assert b["qformer"] == 1920


def test_compression_is_one_when_queries_match_patches():
    b = visual_token_budget(4, 64, 64)
    assert b["compression"] == APPROX(1.0)
    assert b["mlp"] == b["qformer"]


def test_token_budget_rejects_zero_images():
    with pytest.raises(ValueError):
        visual_token_budget(0, 256, 32)


# ------------------------------------------------------------- pick_bridge
def test_pick_bridge_prefers_mlp_when_it_fits():
    assert pick_bridge(1, 256, 32, 4096) == "mlp"


def test_pick_bridge_falls_back_to_qformer_for_long_video():
    assert pick_bridge(60, 576, 32, 32768) == "qformer"


def test_pick_bridge_treats_an_exactly_full_context_as_fitting():
    assert pick_bridge(1, 256, 32, 256) == "mlp"
    assert pick_bridge(1, 256, 32, 255) == "qformer"


def test_pick_bridge_refuses_when_nothing_fits():
    with pytest.raises(ValueError):
        pick_bridge(1000, 576, 32, 1024)
