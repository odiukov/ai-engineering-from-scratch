"""Тесты к уроку «KV-cache, Flash Attention и оптимизация инференса». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    KVCache,
    attention_full,
    decode_cached,
    decode_naive,
    kv_cache_bytes,
    softmax,
    tiled_softmax_dot,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не сравнивает вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def toy_qkv(n, d, seed):
    """Детерминированные Q, K, V: тесты обязаны быть воспроизводимы."""
    rng = random.Random(seed)
    Q = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    return Q, K, V


def project(state):
    """Детерминированная «проекция» токена в (k, v). Чистая функция."""
    return [state, state * state], [state + 1.0, 1.0 - state]


# ---------------------------------------------------------------- softmax
def test_softmax_sums_to_one():
    assert sum(softmax([3.0, -1.0, 0.5, 7.0])) == APPROX(1.0)


def test_softmax_of_equal_scores_is_uniform():
    assert softmax([2.0] * 4) == pytest.approx([0.25] * 4)


def test_softmax_is_shift_invariant():
    """Прибавили константу ко всем логитам — распределение то же самое."""
    a = softmax([1.0, 2.0, 3.0])
    b = softmax([101.0, 102.0, 103.0])
    assert a == pytest.approx(b)


def test_softmax_survives_huge_logits():
    """Наивный math.exp(1000) бросает OverflowError — сдвиг на максимум спасает."""
    out = softmax([1000.0, 999.0, 990.0])
    assert sum(out) == APPROX(1.0)
    assert out[0] > out[1] > out[2]


# --------------------------------------------------------- attention_full
def test_attention_of_identical_keys_averages_values():
    out = attention_full([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [[2.0], [4.0]])
    assert out == pytest.approx([3.0])


def test_attention_of_dominant_key_returns_its_value():
    """Один логит намного больше остальных — softmax вырождается в argmax."""
    out = attention_full([50.0, 0.0], [[1.0, 0.0], [-1.0, 0.0]], [[2.0], [4.0]])
    assert out == pytest.approx([2.0], abs=1e-9)


def test_attention_output_is_convex_combination_of_values():
    """Выход внимания всегда лежит внутри выпуклой оболочки значений."""
    Q, K, V = toy_qkv(6, 4, seed=1)
    out = attention_full(Q[0], K, V)
    for j in range(len(out)):
        column = [v[j] for v in V]
        assert min(column) - 1e-9 <= out[j] <= max(column) + 1e-9


def test_attention_scales_scores_by_inverse_sqrt_d():
    """Без деления на sqrt(d) веса были бы другими — проверяем ручным счётом."""
    q, Ks, Vs = [1.0, 1.0, 1.0, 1.0], [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0], [0.0]]
    # score_0 = 1/sqrt(4) = 0.5, score_1 = 0
    w0 = math.exp(0.5) / (math.exp(0.5) + 1.0)
    assert attention_full(q, Ks, Vs) == pytest.approx([w0])


# ----------------------------------------------------- tiled_softmax_dot
def test_tiled_softmax_matches_full_attention():
    """Главное свойство Flash Attention: результат точный, а не приближённый."""
    Q, K, V = toy_qkv(17, 8, seed=2)
    reference = attention_full(Q[-1], K, V)
    assert tiled_softmax_dot(Q[-1], K, V, tile=4) == pytest.approx(reference, abs=1e-12)


def test_tiled_softmax_is_independent_of_tile_size():
    """Размер блока — параметр железа, а не математики. Ответ не меняется."""
    Q, K, V = toy_qkv(17, 8, seed=3)
    outs = [tiled_softmax_dot(Q[0], K, V, tile=t) for t in (1, 2, 3, 5, 17, 64)]
    for other in outs[1:]:
        assert other == pytest.approx(outs[0], abs=1e-12)


def test_tiled_softmax_survives_huge_logits():
    """Логиты порядка 1000: без бегущего максимума здесь OverflowError."""
    q = [200.0, 0.0, 0.0, 0.0]
    K = [[float(i), 0.0, 0.0, 0.0] for i in range(1, 11)]
    V = [[float(i)] for i in range(1, 11)]
    out = tiled_softmax_dot(q, K, V, tile=3)
    assert out == pytest.approx([10.0], abs=1e-9)


def test_tiled_softmax_matches_full_attention_on_huge_logits():
    q = [200.0, 0.0, 0.0, 0.0]
    K = [[float(i), 0.0, 0.0, 0.0] for i in range(1, 11)]
    V = [[float(i), float(-i)] for i in range(1, 11)]
    assert tiled_softmax_dot(q, K, V, tile=4) == pytest.approx(
        attention_full(q, K, V), abs=1e-12
    )


def test_tiled_softmax_single_key_returns_that_value():
    assert tiled_softmax_dot([1.0, 2.0], [[3.0, 4.0]], [[7.0, 8.0]], tile=4) == pytest.approx(
        [7.0, 8.0]
    )


# ---------------------------------------------------------------- KVCache
def test_kv_cache_starts_empty():
    assert len(KVCache()) == 0


def test_kv_cache_grows_by_one_per_token():
    c = KVCache()
    for i in range(5):
        c.append([float(i)], [float(-i)])
        assert len(c) == i + 1


def test_kv_cache_read_returns_tokens_in_order():
    c = KVCache()
    c.append([1.0], [2.0])
    c.append([3.0], [4.0])
    Ks, Vs = c.read()
    assert flat(Ks) == pytest.approx([1.0, 3.0])
    assert flat(Vs) == pytest.approx([2.0, 4.0])


# ------------------------------------------------- decode_naive / _cached
def test_cached_decoder_gives_exactly_the_same_output_as_recomputing():
    """Главный тест урока: кэш ничего не приближает, выходы бит в бит те же."""
    states = [0.1 * i for i in range(1, 13)]
    Q, _, _ = toy_qkv(12, 2, seed=4)
    naive_out, _ = decode_naive(states, project, Q)
    cached_out, _ = decode_cached(states, project, Q)
    assert flat(cached_out) == pytest.approx(flat(naive_out), abs=1e-12)


def test_naive_decoder_projects_quadratically_many_times():
    states = [0.1 * i for i in range(1, 101)]
    Q, _, _ = toy_qkv(100, 2, seed=5)
    _, calls = decode_naive(states, project, Q)
    assert calls == 100 * 101 // 2 == 5050


def test_cached_decoder_projects_once_per_token():
    states = [0.1 * i for i in range(1, 101)]
    Q, _, _ = toy_qkv(100, 2, seed=5)
    _, calls = decode_cached(states, project, Q)
    assert calls == 100


def test_cache_saving_grows_with_sequence_length():
    """На 10 токенах выигрыш 5.5x, на 100 — 50.5x. Он линеен по N."""
    ratios = []
    for n in (10, 100):
        states = [0.1 * i for i in range(1, n + 1)]
        Q, _, _ = toy_qkv(n, 2, seed=6)
        ratios.append(decode_naive(states, project, Q)[1] / decode_cached(states, project, Q)[1])
    assert ratios == pytest.approx([5.5, 50.5])


def test_decoder_is_causal_first_output_sees_only_first_token():
    """Выход первого шага не зависит от того, что придёт потом."""
    Q, _, _ = toy_qkv(4, 2, seed=7)
    short, _ = decode_cached([1.0], project, Q[:1])
    long, _ = decode_cached([1.0, 2.0, 3.0, 4.0], project, Q)
    assert long[0] == pytest.approx(short[0])


# --------------------------------------------------------- kv_cache_bytes
def test_kv_cache_bytes_counts_k_and_v_separately():
    """Двойка в формуле — это K и V, а не «на всякий случай»."""
    assert kv_cache_bytes(1, 1, 1, 128, 2) == 512


def test_kv_cache_bytes_for_llama3_70b_at_32k():
    """80 слоёв, GQA с 8 KV-головами, d_head=128, fp16."""
    assert kv_cache_bytes(32768, 80, 8, 128, 2) == 10_737_418_240


def test_gqa_shrinks_cache_proportionally_to_kv_heads():
    """MHA с 64 головами против GQA с 8 — ровно восьмикратная экономия."""
    mha = kv_cache_bytes(32768, 80, 64, 128, 2)
    gqa = kv_cache_bytes(32768, 80, 8, 128, 2)
    assert mha / gqa == APPROX(8.0)


def test_fp8_cache_is_half_of_fp16():
    assert kv_cache_bytes(4096, 32, 8, 128, 1) * 2 == kv_cache_bytes(4096, 32, 8, 128, 2)


def test_kv_cache_grows_linearly_with_context():
    a = kv_cache_bytes(2048, 32, 8, 128)
    b = kv_cache_bytes(131072, 32, 8, 128)
    assert b / a == APPROX(64.0)
