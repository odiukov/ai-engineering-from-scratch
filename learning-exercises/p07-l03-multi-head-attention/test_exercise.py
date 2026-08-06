"""Тесты к уроку «Multi-head attention». Правь exercise.py."""

import random

import pytest

from exercise import (
    combine_heads,
    head_attention,
    kv_cache_cells,
    matmul,
    multi_head_attention,
    repeat_kv_heads,
    softmax,
    split_heads,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def noise(rows, cols, seed):
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def column(M, j):
    return [row[j] for row in M]


# три токена, d_model = 4, значит при 2 головах d_head = 2
X3 = [
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.5, 0.5, 1.0, 1.0],
]


# ------------------------------------------------------------------ matmul
def test_matmul_row_by_column():
    assert flat(matmul([[1.0, 2.0]], [[3.0], [4.0]])) == APPROX([11.0])


def test_matmul_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        matmul([[1.0, 2.0, 3.0]], [[1.0], [2.0]])


# ----------------------------------------------------------------- softmax
def test_softmax_weights_sum_to_one():
    assert sum(softmax([2.0, -1.0, 0.5])) == pytest.approx(1.0)


def test_softmax_survives_huge_scores():
    """Без вычитания максимума math.exp(900) кидает OverflowError."""
    assert sum(softmax([900.0, 899.0])) == pytest.approx(1.0)


def test_softmax_rejects_an_empty_score_vector():
    with pytest.raises(ValueError):
        softmax([])


# ------------------------------------------------------------- split_heads
def test_split_heads_cuts_the_row_into_contiguous_slices():
    assert split_heads([[1, 2, 3, 4]], 2) == [[[1, 2]], [[3, 4]]]


def test_split_heads_keeps_every_token_as_a_row():
    heads = split_heads(X3, 2)
    assert len(heads) == 2
    assert all(len(head) == 3 for head in heads)
    assert all(len(row) == 2 for head in heads for row in head)


def test_split_heads_with_one_head_returns_the_matrix_unchanged():
    assert flat(split_heads(X3, 1)[0]) == APPROX(flat(X3))


def test_split_heads_rejects_a_width_that_does_not_divide_evenly():
    with pytest.raises(ValueError):
        split_heads([[1.0, 2.0, 3.0]], 2)


# ----------------------------------------------------------- combine_heads
def test_combine_heads_glues_slices_back():
    assert combine_heads([[[1, 2]], [[3, 4]]]) == [[1, 2, 3, 4]]


def test_split_then_combine_is_the_identity():
    """Round-trip — самая дешёвая страховка от перепутанного transpose."""
    assert flat(combine_heads(split_heads(X3, 2))) == APPROX(flat(X3))


def test_combine_heads_preserves_head_order():
    """Порядок голов важен: W_o применяется к конкретным столбцам."""
    heads = [[[1.0, 1.0]], [[9.0, 9.0]]]
    assert combine_heads(heads) == [[1.0, 1.0, 9.0, 9.0]]


# ---------------------------------------------------------- head_attention
def test_head_attention_with_equal_scores_averages_the_values():
    out, weights = head_attention([[0.0]], [[0.0], [0.0]], [[1.0], [3.0]])
    assert flat(out) == APPROX([2.0])
    assert flat(weights) == APPROX([0.5, 0.5])


def test_head_attention_scales_by_sqrt_of_head_width():
    """Сырые скоры [2, 0], d_head = 4, значит softmax считается от [1, 0]."""
    Q = [[1.0, 1.0, 1.0, 1.0]]
    K = [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    _, weights = head_attention(Q, K, [[1.0], [0.0]])
    assert flat(weights) == APPROX(softmax([1.0, 0.0]))


def test_head_attention_weights_sum_to_one_per_row():
    _, weights = head_attention(X3, X3, X3)
    assert [sum(row) for row in weights] == pytest.approx([1.0, 1.0, 1.0])


def test_head_attention_mask_gives_exactly_zero_weight():
    """Не «почти ноль»: -1e9 вместо маски оставил бы утечку."""
    mask = [[j <= i for j in range(3)] for i in range(3)]
    _, weights = head_attention(X3, X3, X3, mask=mask)
    assert weights[0] == APPROX([1.0, 0.0, 0.0])
    assert weights[1][2] == 0.0


def test_head_attention_rejects_a_fully_masked_row():
    with pytest.raises(ValueError):
        head_attention([[1.0, 0.0]], [[1.0, 0.0]], [[1.0]], mask=[[False]])


# --------------------------------------------------------- repeat_kv_heads
def test_repeat_kv_heads_expands_groups_in_order():
    A, B = [[1.0]], [[2.0]]
    assert repeat_kv_heads([A, B], 4) == [A, A, B, B]


def test_repeat_kv_heads_with_one_group_is_multi_query_attention():
    A = [[1.0]]
    assert repeat_kv_heads([A], 3) == [A, A, A]


def test_repeat_kv_heads_with_matching_counts_changes_nothing():
    """n_kv == n_heads — это обычный MHA, размножать нечего."""
    A, B = [[1.0]], [[2.0]]
    assert repeat_kv_heads([A, B], 2) == [A, B]


def test_repeat_kv_heads_rejects_uneven_groups():
    with pytest.raises(ValueError):
        repeat_kv_heads([[[1.0]], [[2.0]], [[3.0]]], 4)


# --------------------------------------------------- multi_head_attention
def test_one_head_with_identity_output_equals_plain_attention():
    """Multi-head — та же математика, применённая к кускам вектора."""
    Wq, Wk, Wv = noise(4, 4, 1), noise(4, 4, 2), noise(4, 4, 3)
    out, weights = multi_head_attention(X3, Wq, Wk, Wv, eye(4), n_heads=1)
    ref_out, ref_weights = head_attention(
        matmul(X3, Wq), matmul(X3, Wk), matmul(X3, Wv)
    )
    assert flat(out) == APPROX(flat(ref_out))
    assert len(weights) == 1
    assert flat(weights[0]) == APPROX(flat(ref_weights))


def test_multi_head_output_keeps_the_model_width():
    out, weights = multi_head_attention(
        X3, noise(4, 4, 4), noise(4, 4, 5), noise(4, 4, 6), noise(4, 4, 7), n_heads=2
    )
    assert len(out) == 3 and all(len(row) == 4 for row in out)
    assert len(weights) == 2
    assert all(len(w) == 3 and all(len(row) == 3 for row in w) for w in weights)


def test_every_head_produces_its_own_distribution():
    _, weights = multi_head_attention(
        X3, noise(4, 4, 8), noise(4, 4, 9), noise(4, 4, 10), eye(4), n_heads=2
    )
    assert all(sum(row) == pytest.approx(1.0) for w in weights for row in w)
    assert flat(weights[0]) != APPROX(flat(weights[1]))


def test_without_the_output_projection_heads_never_mix():
    """W_o = I: столбцы головы 0 не должны зависеть от значений головы 1."""
    Wq, Wk = eye(4), eye(4)
    Wv_a = eye(4)
    Wv_b = [row[:] for row in Wv_a]
    Wv_b[2][2] = 5.0  # трогаем столбец, который достаётся только голове 1
    out_a, _ = multi_head_attention(X3, Wq, Wk, Wv_a, eye(4), n_heads=2)
    out_b, _ = multi_head_attention(X3, Wq, Wk, Wv_b, eye(4), n_heads=2)
    assert column(out_a, 0) == APPROX(column(out_b, 0))
    assert column(out_a, 1) == APPROX(column(out_b, 1))
    assert column(out_a, 2) != APPROX(column(out_b, 2))


def test_the_output_projection_is_where_heads_mix():
    """С перемешивающей W_o та же правка головы 1 доходит до столбца 0."""
    Wq, Wk = eye(4), eye(4)
    Wv_a = eye(4)
    Wv_b = [row[:] for row in Wv_a]
    Wv_b[2][2] = 5.0
    Wo = [[1.0] * 4 for _ in range(4)]
    out_a, _ = multi_head_attention(X3, Wq, Wk, Wv_a, Wo, n_heads=2)
    out_b, _ = multi_head_attention(X3, Wq, Wk, Wv_b, Wo, n_heads=2)
    assert column(out_a, 0) != APPROX(column(out_b, 0))


def test_grouped_query_attention_with_full_kv_equals_plain_multi_head():
    Wq, Wk, Wv, Wo = noise(4, 4, 11), noise(4, 4, 12), noise(4, 4, 13), noise(4, 4, 14)
    plain, _ = multi_head_attention(X3, Wq, Wk, Wv, Wo, n_heads=2)
    gqa, _ = multi_head_attention(X3, Wq, Wk, Wv, Wo, n_heads=2, n_kv_heads=2)
    assert flat(plain) == APPROX(flat(gqa))


def test_multi_query_attention_uses_a_narrower_kv_projection():
    """Одна kv-голова: Wk и Wv имеют ширину d_head, а не d_model."""
    Wk = noise(4, 2, 15)
    Wv = noise(4, 2, 16)
    out, weights = multi_head_attention(
        X3, noise(4, 4, 17), Wk, Wv, eye(4), n_heads=2, n_kv_heads=1
    )
    assert len(out) == 3 and all(len(row) == 4 for row in out)
    assert len(weights) == 2


def test_multi_head_passes_the_mask_into_every_head():
    mask = [[j <= i for j in range(3)] for i in range(3)]
    _, weights = multi_head_attention(
        X3, eye(4), eye(4), eye(4), eye(4), n_heads=2, mask=mask
    )
    assert all(w[0] == APPROX([1.0, 0.0, 0.0]) for w in weights)
    assert all(w[1][2] == 0.0 for w in weights)


def test_multi_head_rejects_a_width_that_does_not_split_evenly():
    with pytest.raises(ValueError):
        multi_head_attention(X3, eye(4), eye(4), eye(4), eye(4), n_heads=3)


# ----------------------------------------------------------- kv_cache_cells
def test_kv_cache_counts_both_keys_and_values():
    assert kv_cache_cells(10, 8, 128) == 2 * 8 * 10 * 128


def test_kv_cache_scales_with_layers():
    assert kv_cache_cells(10, 8, 128, n_layers=32) == 32 * kv_cache_cells(10, 8, 128)


def test_grouped_query_attention_shrinks_the_cache_eightfold():
    """Llama 3 70B: 64 головы запросов против 8 kv — ровно 8x по кэшу."""
    full = kv_cache_cells(2048, 64, 128, n_layers=80)
    grouped = kv_cache_cells(2048, 8, 128, n_layers=80)
    assert full / grouped == pytest.approx(8.0)


def test_kv_cache_grows_linearly_with_context_not_quadratically():
    """Кэш линеен по длине — в отличие от матрицы внимания."""
    assert kv_cache_cells(4096, 8, 128) == 2 * kv_cache_cells(2048, 8, 128)
