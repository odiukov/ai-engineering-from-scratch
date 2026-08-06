"""Тесты к уроку «Self-attention с нуля». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    attention_scores,
    causal_mask,
    matmul,
    scaled_dot_product_attention,
    self_attention,
    softmax,
    transpose,
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


# «The cat sat» — три токена, четыре измерения
X3 = [
    [1.0, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.5],
    [0.5, 0.5, 1.0, 1.0],
]


# ----------------------------------------------------------------- softmax
def test_softmax_weights_sum_to_one():
    assert sum(softmax([2.0, -1.0, 0.5])) == pytest.approx(1.0)


def test_softmax_of_equal_scores_is_uniform():
    assert softmax([4.0, 4.0, 4.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_invariant_to_a_constant_shift():
    """Общий множитель сокращается — на этом и держится трюк с вычитанием max."""
    assert softmax([1.0, 2.0, 3.0]) == APPROX(softmax([101.0, 102.0, 103.0]))


def test_softmax_survives_huge_scores():
    """Без вычитания максимума math.exp(900) кидает OverflowError."""
    weights = softmax([900.0, 899.0])
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1]


def test_softmax_rejects_an_empty_score_vector():
    with pytest.raises(ValueError):
        softmax([])


# --------------------------------------------------------------- transpose
def test_transpose_swaps_rows_and_columns():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]


def test_transpose_twice_is_the_original():
    M = noise(4, 3, seed=1)
    assert flat(transpose(transpose(M))) == APPROX(flat(M))


def test_transpose_of_empty_is_empty():
    assert transpose([]) == []


# ------------------------------------------------------------------ matmul
def test_matmul_row_by_column():
    assert flat(matmul([[1.0, 2.0]], [[3.0], [4.0]])) == APPROX([11.0])


def test_matmul_with_identity_returns_the_original():
    M = noise(3, 3, seed=2)
    assert flat(matmul(M, eye(3))) == APPROX(flat(M))


def test_matmul_output_shape_is_rows_of_a_by_columns_of_b():
    out = matmul(noise(4, 5, seed=3), noise(5, 2, seed=4))
    assert len(out) == 4
    assert all(len(row) == 2 for row in out)


def test_matmul_rejects_mismatched_shapes():
    """Молчаливый обрез через zip спрятал бы ошибку размерности."""
    with pytest.raises(ValueError):
        matmul([[1.0, 2.0, 3.0]], [[1.0], [2.0]])


# --------------------------------------------------------- attention_scores
def test_attention_scores_divide_by_sqrt_dk():
    """Сырое скалярное произведение равно 2.0, dk равно 4, ответ 1.0."""
    Q = [[1.0, 1.0, 1.0, 1.0]]
    K = [[1.0, 1.0, 0.0, 0.0]]
    assert flat(attention_scores(Q, K)) == APPROX([2.0 / math.sqrt(4)])


def test_attention_scores_shape_is_queries_by_keys():
    scores = attention_scores(noise(3, 8, seed=5), noise(5, 8, seed=6))
    assert len(scores) == 3
    assert all(len(row) == 5 for row in scores)


def test_scaling_keeps_softmax_from_saturating():
    """Без деления на sqrt(dk) при dk=64 softmax схлопывается в one-hot.

    Берём один и тот же Q@K^T, сравниваем масштабированное распределение с
    немасштабированным: у сырого максимальный вес почти единица, а значит
    градиент почти нулевой.
    """
    dk = 64
    Q = [[1.0] * dk]
    K = [[1.0] * dk, [0.5] * dk]
    scaled = softmax(attention_scores(Q, K)[0])
    raw = softmax([s * math.sqrt(dk) for s in attention_scores(Q, K)[0]])
    assert max(raw) > 0.999
    assert max(scaled) < 0.99


def test_attention_scores_of_no_queries_is_empty():
    assert attention_scores([], [[1.0, 0.0]]) == []


# ------------------------------------------------------------- causal_mask
def test_causal_mask_is_lower_triangular():
    assert causal_mask(3) == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_causal_mask_always_lets_a_position_see_itself():
    mask = causal_mask(5)
    assert all(mask[i][i] for i in range(5))


def test_causal_mask_forbids_every_future_position():
    mask = causal_mask(5)
    assert all(not mask[i][j] for i in range(5) for j in range(i + 1, 5))


# ------------------------------------- scaled_dot_product_attention
def test_attention_with_equal_scores_averages_the_values():
    out, weights = scaled_dot_product_attention([[0.0]], [[0.0], [0.0]], [[1.0], [3.0]])
    assert flat(out) == APPROX([2.0])
    assert flat(weights) == APPROX([0.5, 0.5])


def test_attention_weights_of_each_row_sum_to_one():
    _, weights = scaled_dot_product_attention(X3, X3, X3)
    assert [sum(row) for row in weights] == pytest.approx([1.0, 1.0, 1.0])


def test_attention_output_is_a_convex_combination_of_the_values():
    """Каждая координата выхода обязана лежать внутри диапазона V по столбцу."""
    out, _ = scaled_dot_product_attention(X3, X3, X3)
    for row in out:
        for i, value in enumerate(row):
            column = [v[i] for v in X3]
            assert min(column) - 1e-9 <= value <= max(column) + 1e-9


def test_attention_with_identity_values_returns_the_weights_themselves():
    """V = I превращает выход в саму матрицу внимания — удобно для отладки."""
    out, weights = scaled_dot_product_attention(X3, X3, eye(3))
    assert flat(out) == APPROX(flat(weights))


def test_attention_collapses_onto_the_matching_key():
    """Огромный скор по одному ключу — выход равен соответствующему value."""
    Q = [[10.0, 0.0]]
    K = [[10.0, 0.0], [0.0, 10.0]]
    V = [[1.0, 2.0], [30.0, 40.0]]
    out, _ = scaled_dot_product_attention(Q, K, V)
    assert flat(out) == pytest.approx([1.0, 2.0], abs=1e-6)


def test_causal_masking_gives_the_future_exactly_zero_weight():
    """Не «почти ноль»: -1e9 вместо маски оставил бы утечку из будущего."""
    _, weights = scaled_dot_product_attention(X3, X3, X3, mask=causal_mask(3))
    assert weights[0][1] == 0.0
    assert weights[0][2] == 0.0
    assert weights[1][2] == 0.0


def test_the_first_position_under_a_causal_mask_sees_only_itself():
    out, weights = scaled_dot_product_attention(X3, X3, X3, mask=causal_mask(3))
    assert weights[0] == APPROX([1.0, 0.0, 0.0])
    assert out[0] == APPROX(X3[0])


def test_causal_masking_really_forbids_looking_ahead():
    """Меняем значения будущих позиций — прошлые выходы обязаны не дрогнуть."""
    mask = causal_mask(3)
    out_a, _ = scaled_dot_product_attention(X3, X3, X3, mask=mask)
    V_changed = [X3[0], [99.0, -99.0, 99.0, -99.0], [7.0, 7.0, 7.0, 7.0]]
    out_b, _ = scaled_dot_product_attention(X3, X3, V_changed, mask=mask)
    assert out_a[0] == APPROX(out_b[0])
    assert out_a[2] != APPROX(out_b[2])


def test_attention_rejects_a_row_where_the_mask_hides_everything():
    with pytest.raises(ValueError):
        scaled_dot_product_attention([[1.0, 0.0]], [[1.0, 0.0]], [[1.0]], mask=[[False]])


# ---------------------------------------------------------- self_attention
def test_self_attention_output_shape_matches_the_value_width():
    Wq = noise(4, 2, seed=7)
    Wk = noise(4, 2, seed=8)
    Wv = noise(4, 3, seed=9)
    out, weights = self_attention(X3, Wq, Wk, Wv)
    assert len(out) == 3 and all(len(row) == 3 for row in out)
    assert len(weights) == 3 and all(len(row) == 3 for row in weights)


def test_self_attention_uses_one_source_for_q_k_and_v():
    """Sanity: с единичными проекциями это ровно attention(X, X, X)."""
    I4 = eye(4)
    out, weights = self_attention(X3, I4, I4, I4)
    ref_out, ref_weights = scaled_dot_product_attention(X3, X3, X3)
    assert flat(out) == APPROX(flat(ref_out))
    assert flat(weights) == APPROX(flat(ref_weights))


def test_self_attention_is_blind_to_token_order():
    """Переставили строки X — строки выхода переставились так же, и всё.

    Отсюда и берётся урок 04: позицию во внимание надо вносить руками.
    """
    I4 = eye(4)
    order = [2, 0, 1]
    permuted = [X3[i] for i in order]
    out, _ = self_attention(X3, I4, I4, I4)
    out_permuted, _ = self_attention(permuted, I4, I4, I4)
    assert flat(out_permuted) == APPROX(flat([out[i] for i in order]))


def test_zero_query_projection_makes_attention_uniform():
    """Wq = 0 обнуляет все скоры, и выход становится средним по values."""
    zero = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    Wv = eye(4)
    out, weights = self_attention(X3, zero, noise(4, 2, seed=10), Wv)
    assert flat(weights) == APPROX(flat([[1 / 3] * 3] * 3))
    mean_row = [sum(v[i] for v in X3) / 3 for i in range(4)]
    assert out[0] == APPROX(mean_row)


def test_self_attention_passes_the_mask_through():
    I4 = eye(4)
    _, weights = self_attention(X3, I4, I4, I4, mask=causal_mask(3))
    assert weights[0] == APPROX([1.0, 0.0, 0.0])
    assert weights[1][2] == 0.0
