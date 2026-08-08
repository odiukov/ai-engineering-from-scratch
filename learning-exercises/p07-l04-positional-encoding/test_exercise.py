"""Тесты к уроку «Позиционное кодирование». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    add_positional_encoding,
    alibi_bias,
    alibi_slopes,
    apply_rope,
    rope_dot,
    scale_rope_base,
    sinusoidal_encoding,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [value for row in matrix for value in row]


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def local_softmax(scores):
    """Локальный хелпер: softmax этот урок не проходит, но проверить надо."""
    shift = max(scores)
    exps = [math.exp(s - shift) for s in scores]
    return [e / sum(exps) for e in exps]


def vector(d, seed):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(d)]


# ------------------------------------------------------ sinusoidal_encoding
def test_sinusoidal_shape_is_positions_by_width():
    pe = sinusoidal_encoding(5, 8)
    assert len(pe) == 5
    assert all(len(row) == 8 for row in pe)


def test_sinusoidal_position_zero_is_alternating_zero_and_one():
    """sin(0) = 0, cos(0) = 1 — узнаваемая первая строка."""
    assert sinusoidal_encoding(1, 6)[0] == APPROX([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


def test_sinusoidal_even_columns_are_sin_and_odd_are_cos():
    pe = sinusoidal_encoding(4, 2)
    assert [row[0] for row in pe] == APPROX([math.sin(p) for p in range(4)])
    assert [row[1] for row in pe] == APPROX([math.cos(p) for p in range(4)])


def test_sinusoidal_values_stay_inside_minus_one_to_one():
    pe = sinusoidal_encoding(64, 16)
    assert all(-1.0 <= value <= 1.0 for value in flat(pe))


def test_sinusoidal_high_index_pairs_change_slowly():
    """Последняя пара — самая низкая частота: соседние позиции почти совпадают."""
    pe = sinusoidal_encoding(2, 32)
    fast = abs(pe[1][0] - pe[0][0])
    slow = abs(pe[1][30] - pe[0][30])
    assert slow < 1e-3
    assert fast > 0.5


def test_sinusoidal_rejects_an_odd_width():
    """Каждой частоте нужна пара sin/cos, поэтому d обязано быть чётным."""
    with pytest.raises(ValueError):
        sinusoidal_encoding(4, 5)


# --------------------------------------------------- add_positional_encoding
def test_add_positional_encoding_keeps_the_shape():
    X = [[0.5] * 4 for _ in range(3)]
    out = add_positional_encoding(X)
    assert len(out) == 3 and all(len(row) == 4 for row in out)


def test_add_positional_encoding_separates_identical_tokens():
    """Два одинаковых эмбеддинга на разных позициях перестают совпадать.

    Именно этой возможности у внимания без позиционного сигнала нет вовсе.
    """
    X = [[1.0, 2.0], [1.0, 2.0]]
    out = add_positional_encoding(X)
    assert out[0] != APPROX(out[1])


def test_add_positional_encoding_adds_and_does_not_replace():
    X = [[3.0, 4.0]]
    assert flat(add_positional_encoding(X)) == APPROX([3.0, 5.0])


def test_add_positional_encoding_does_not_mutate_the_input():
    X = [[1.0, 2.0], [3.0, 4.0]]
    add_positional_encoding(X)
    assert X == [[1.0, 2.0], [3.0, 4.0]]


def test_add_positional_encoding_of_empty_is_empty():
    assert add_positional_encoding([]) == []


# --------------------------------------------------------------- apply_rope
def test_rope_at_position_zero_is_the_identity():
    x = vector(8, seed=1)
    assert apply_rope(x, 0) == APPROX(x)


def test_rope_rotates_the_first_pair_by_the_position_itself():
    """theta_0 = base^0 = 1, значит первая пара крутится ровно на pos радиан."""
    assert apply_rope([1.0, 0.0], 1) == APPROX([math.cos(1.0), math.sin(1.0)])


def test_rope_preserves_the_norm():
    """Это поворот, а не растяжение: длина вектора обязана сохраниться."""
    x = vector(16, seed=2)
    assert norm(apply_rope(x, 37)) == pytest.approx(norm(x), abs=1e-9)


def test_rope_moves_the_vector_at_a_nonzero_position():
    x = vector(8, seed=3)
    assert apply_rope(x, 4) != APPROX(x)


def test_rope_rejects_an_odd_width():
    with pytest.raises(ValueError):
        apply_rope([1.0, 2.0, 3.0], 1)


def test_a_larger_base_barely_rotates_the_slow_dimensions():
    """base — это ручка: чем он больше, тем медленнее крутятся хвостовые пары."""
    x = [1.0] * 8
    slow = apply_rope(x, 100, base=10 ** 9)
    fast = apply_rope(x, 100, base=10)
    assert slow[6:] == pytest.approx(x[6:], abs=1e-3)
    assert fast[6:] != pytest.approx(x[6:], abs=1e-3)


# ----------------------------------------------------------------- rope_dot
def test_rope_dot_at_equal_positions_is_the_plain_dot_product():
    """Одинаковые повороты взаимно сокращаются на любой позиции."""
    q, k = vector(8, seed=4), vector(8, seed=5)
    assert rope_dot(q, k, 0, 0) == pytest.approx(dot(q, k), abs=1e-9)
    assert rope_dot(q, k, 17, 17) == pytest.approx(dot(q, k), abs=1e-9)


def test_rope_dot_depends_only_on_the_relative_distance():
    """Сердце RoPE: сдвинь обе позиции на одно и то же — скор не изменится."""
    q, k = vector(8, seed=6), vector(8, seed=7)
    near = rope_dot(q, k, 3, 7)
    shifted = rope_dot(q, k, 103, 107)
    assert near == pytest.approx(shifted, abs=1e-9)


def test_rope_dot_changes_when_the_gap_changes():
    """Относительное расстояние в скор всё-таки входит, иначе смысла нет."""
    q, k = vector(8, seed=8), vector(8, seed=9)
    assert rope_dot(q, k, 0, 1) != pytest.approx(rope_dot(q, k, 0, 5), abs=1e-6)


def test_rope_dot_is_symmetric_in_the_sign_of_the_gap_only_through_rotation():
    """Скор с зазором +2 и -2 в общем случае разный: поворот не симметричен."""
    q, k = vector(8, seed=10), vector(8, seed=11)
    assert rope_dot(q, k, 5, 3) != pytest.approx(rope_dot(q, k, 3, 5), abs=1e-6)


# ----------------------------------------------------------- scale_rope_base
def test_scaling_by_one_leaves_the_base_alone():
    assert scale_rope_base(10000, 1.0, 128) == pytest.approx(10000.0)


def test_scaling_grows_the_base():
    assert scale_rope_base(10000, 32.0, 128) == pytest.approx(338098.0, rel=1e-3)


def test_ntk_factor_sixteen_grows_the_base_without_claiming_a_yarn_recipe():
    """8K -> 128K задаёт factor=16, но одна эта формула ещё не является YaRN."""
    assert scale_rope_base(10000, 4.0, 128) < scale_rope_base(10000, 16.0, 128)


def test_scale_rope_base_rejects_a_head_width_of_two_or_less():
    """Показатель d/(d-2) при d = 2 делит на ноль."""
    with pytest.raises(ValueError):
        scale_rope_base(10000, 8.0, 2)


# ------------------------------------------------------------- alibi_slopes
def test_alibi_slopes_are_the_powers_of_two_from_the_paper():
    assert alibi_slopes(8) == APPROX([2 ** -(h + 1) for h in range(8)])


def test_alibi_slopes_decrease_across_heads():
    """Голова 0 смотрит близко, последняя — далеко."""
    slopes = alibi_slopes(16)
    assert all(slopes[h] > slopes[h + 1] for h in range(15))


def test_the_last_alibi_slope_is_always_one_over_256():
    assert alibi_slopes(4)[-1] == pytest.approx(2 ** -8)
    assert alibi_slopes(32)[-1] == pytest.approx(2 ** -8)


def test_alibi_slopes_rejects_zero_heads():
    with pytest.raises(ValueError):
        alibi_slopes(0)


# --------------------------------------------------------------- alibi_bias
def test_alibi_bias_shape_is_heads_by_seq_by_seq():
    bias = alibi_bias(4, 6)
    assert len(bias) == 4
    assert all(len(head) == 6 and all(len(row) == 6 for row in head) for head in bias)


def test_alibi_bias_diagonal_is_zero():
    """Расстояние до себя нулевое, штрафа нет."""
    bias = alibi_bias(2, 5)
    assert all(head[i][i] == 0.0 for head in bias for i in range(5))


def test_alibi_bias_is_symmetric_and_grows_with_distance():
    head = alibi_bias(1, 5)[0]
    assert head[0][4] == pytest.approx(head[4][0])
    assert head[0][4] < head[0][2] < head[0][1] < head[0][0]


def test_the_first_head_penalizes_distance_hardest():
    bias = alibi_bias(8, 4)
    assert bias[0][0][3] < bias[7][0][3]


def test_alibi_bias_pulls_attention_toward_the_neighbours():
    """Скоры плоские, весь порядок приносит только штраф за расстояние."""
    head = alibi_bias(1, 5)[0]
    weights = local_softmax([0.0 + b for b in head[4]])
    assert weights.index(max(weights)) == 4
    assert weights[0] < weights[1] < weights[2] < weights[3] < weights[4]
