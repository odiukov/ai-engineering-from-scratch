"""Тесты к уроку «Механизм внимания». Правь exercise.py."""

import math

import pytest

from exercise import (
    additive_score,
    alignment_matrix,
    attend,
    dot_score,
    general_score,
    masked_softmax,
    multi_head_dot_attention,
    softmax,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


# три состояния энкодера, примерно «cat», «sat», «mat» из урока
H = [
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
]
IDENTITY3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# ----------------------------------------------------------------- softmax
def test_softmax_weights_sum_to_one():
    assert sum(softmax([2.0, -1.0, 0.5])) == pytest.approx(1.0)


def test_softmax_of_equal_scores_is_uniform():
    assert softmax([4.0, 4.0, 4.0]) == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_softmax_survives_huge_scores():
    """Ловушка: без вычитания максимума math.exp переполняется."""
    weights = softmax([900.0, 899.0])
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] > weights[1]


def test_softmax_rejects_an_empty_score_vector():
    with pytest.raises(ValueError):
        softmax([])


# ---------------------------------------------------------- masked_softmax
def test_masked_softmax_gives_padding_exactly_zero():
    """Не «почти ноль», а ровно 0.0 — иначе паддинг подмешается в контекст."""
    weights = masked_softmax([1.0, 1.0, 5.0], [True, True, False])
    assert weights[2] == 0.0
    assert weights[:2] == APPROX([0.5, 0.5])


def test_masked_softmax_still_sums_to_one():
    weights = masked_softmax([3.0, -2.0, 0.0, 7.0], [True, False, True, False])
    assert sum(weights) == pytest.approx(1.0)


def test_masked_softmax_with_everything_visible_equals_plain_softmax():
    scores = [1.0, 2.0, 3.0]
    assert masked_softmax(scores, [True] * 3) == APPROX(softmax(scores))


def test_masked_softmax_rejects_a_fully_hidden_row():
    with pytest.raises(ValueError):
        masked_softmax([1.0, 2.0], [False, False])


# --------------------------------------------------------------- dot_score
def test_dot_score_returns_one_score_per_encoder_position():
    assert len(dot_score([0.9, 0.1, 0.2], H)) == len(H)


def test_dot_score_is_highest_for_the_most_aligned_key():
    scores = dot_score([0.9, 0.1, 0.2], H)
    assert scores.index(max(scores)) == 0


def test_dot_score_rejects_a_query_of_the_wrong_width():
    """Жёсткое ограничение Luong dot: d_s должно равняться d_h."""
    with pytest.raises(ValueError):
        dot_score([1.0, 0.0], H)


# ----------------------------------------------------------- general_score
def test_general_score_with_identity_matrix_equals_dot_score():
    query = [0.9, 0.1, 0.2]
    assert general_score(query, H, IDENTITY3) == APPROX(dot_score(query, H))


def test_general_score_allows_different_query_and_key_widths():
    """В этом и смысл W: query размерности 2 против ключей размерности 3."""
    W = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert general_score([1.0, 2.0], H, W) == APPROX([1.0, 1.5, 1.9])


def test_general_score_returns_one_score_per_key():
    W = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert len(general_score([1.0, 2.0], H, W)) == len(H)


# ---------------------------------------------------------- additive_score
def test_additive_score_returns_one_score_per_key():
    W_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    U_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert len(additive_score([1.0, 0.0, 0.0], H, W_a, U_a, [1.0, 1.0])) == len(H)


def test_additive_score_is_bounded_by_the_size_of_v_a():
    """|tanh| < 1, поэтому |e_i| строго меньше суммы модулей v_a."""
    W_a = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]
    U_a = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]]
    v_a = [2.0, -3.0]
    scores = additive_score([5.0, 5.0, 5.0], H, W_a, U_a, v_a)
    assert all(abs(s) < sum(abs(v) for v in v_a) for s in scores)


def test_additive_score_with_a_zero_projection_makes_attention_uniform():
    """v_a = 0 схлопывает все скоры в ноль, значит внимание размазано ровно."""
    W_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    U_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    scores = additive_score([1.0, 2.0, 3.0], H, W_a, U_a, [0.0, 0.0])
    assert scores == APPROX([0.0, 0.0, 0.0])
    _, weights = attend(scores, H)
    assert weights == APPROX([1 / 3, 1 / 3, 1 / 3])


def test_additive_score_does_not_require_equal_query_and_key_widths():
    """Bahdanau живёт там, где Luong dot падает: d_s = 2, d_h = 3."""
    W_a = [[1.0, 0.0], [0.0, 1.0]]
    U_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    scores = additive_score([0.5, 0.5], H, W_a, U_a, [1.0, 1.0])
    assert len(scores) == len(H)


# ------------------------------------------------------------------ attend
def test_attend_weights_sum_to_one():
    _, weights = attend([2.0, -1.0, 0.5], H)
    assert sum(weights) == pytest.approx(1.0)


def test_attend_averages_the_values():
    context, weights = attend([0.0, 0.0], [[1.0], [3.0]])
    assert context == APPROX([2.0])
    assert weights == APPROX([0.5, 0.5])


def test_attend_context_is_a_convex_combination_of_the_values():
    """Каждая координата контекста обязана лежать внутри диапазона values."""
    context, _ = attend([1.0, -2.0, 0.3], H)
    for i, value in enumerate(context):
        column = [row[i] for row in H]
        assert min(column) <= value <= max(column)


def test_attend_collapses_onto_one_value_when_a_score_dominates():
    context, _ = attend([50.0, 0.0, 0.0], H)
    assert context == pytest.approx(H[0], abs=1e-9)


def test_attend_weights_the_values_not_the_keys():
    """Скоры считались против ключей, а суммируются values — это разные вещи."""
    scores = [1.0, 0.0]
    first, _ = attend(scores, [[1.0], [0.0]])
    second, _ = attend(scores, [[0.0], [1.0]])
    assert first != APPROX(second)


def test_attend_with_a_mask_ignores_the_padded_value_completely():
    context, weights = attend([0.0, 99.0], [[1.0], [7.0]], mask=[True, False])
    assert weights == APPROX([1.0, 0.0])
    assert context == APPROX([1.0])


# -------------------------------------------------------- alignment_matrix
def test_alignment_matrix_has_one_row_per_decoder_step():
    rows = alignment_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], H)
    assert len(rows) == 2
    assert all(len(row) == len(H) for row in rows)


def test_alignment_matrix_rows_each_sum_to_one():
    rows = alignment_matrix([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]], H)
    assert [sum(row) for row in rows] == pytest.approx([1.0, 1.0])


def test_alignment_matrix_peaks_on_the_diagonal_for_matching_queries():
    """Запрос, совпадающий с i-м состоянием энкодера, смотрит на позицию i."""
    encoder = IDENTITY3
    queries = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    rows = alignment_matrix(queries, encoder)
    assert [row.index(max(row)) for row in rows] == [0, 1, 2]


def test_alignment_matrix_of_no_decoder_steps_is_empty():
    assert alignment_matrix([], H) == []


# ------------------------------------------------ multi_head_dot_attention
def test_single_head_equals_plain_dot_attention():
    query = [0.9, 0.1, 0.2]
    context, heads = multi_head_dot_attention(query, H, H, 1)
    plain_context, plain_weights = attend(dot_score(query, H), H)
    assert context == APPROX(plain_context)
    assert len(heads) == 1
    assert heads[0] == APPROX(plain_weights)


def test_multi_head_context_keeps_the_original_width():
    context, _ = multi_head_dot_attention([1.0, 1.0, 0.0, 0.0], [[1.0] * 4] * 2, [[2.0] * 4] * 2, 2)
    assert len(context) == 4


def test_every_head_produces_its_own_distribution():
    _, heads = multi_head_dot_attention([1.0, 1.0], [[5.0, 0.0], [0.0, 5.0]], [[5.0, 0.0], [0.0, 5.0]], 2)
    assert len(heads) == 2
    assert all(sum(w) == pytest.approx(1.0) for w in heads)
    assert heads[0].index(max(heads[0])) == 0
    assert heads[1].index(max(heads[1])) == 1


def test_multi_head_rejects_a_width_that_does_not_split_evenly():
    with pytest.raises(ValueError):
        multi_head_dot_attention([1.0, 0.0, 0.0], H, H, 2)


def test_multi_head_uses_math_not_magic():
    """Голова из одного измерения — просто softmax по одной координате."""
    context, heads = multi_head_dot_attention([1.0], [[0.0], [math.log(4)]], [[0.0], [4.0]], 1)
    assert heads[0][1] > heads[0][0]
    assert 0.0 < context[0] < 4.0
