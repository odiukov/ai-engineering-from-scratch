"""Тесты к уроку «Оптимизация инференса». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    attention,
    batching_steps,
    generate_no_cache,
    generate_with_cache,
    kv_cache_bytes,
    matvec,
    softmax,
    speculative_speedup,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

IDENTITY = [[1.0, 0.0], [0.0, 1.0]]

_rng = random.Random(20260807)
D = 4
W_K = [[_rng.gauss(0.0, 0.5) for _ in range(D)] for _ in range(D)]
W_V = [[_rng.gauss(0.0, 0.5) for _ in range(D)] for _ in range(D)]
TOKENS = [[_rng.gauss(0.0, 1.0) for _ in range(D)] for _ in range(6)]

LLAMA70B = dict(num_layers=80, num_kv_heads=8, head_dim=128)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу."""
    return [v for row in M for v in row]


# ---------------------------------------------------------------- softmax
def test_softmax_of_equal_scores_is_uniform():
    assert softmax([0.0, 0.0]) == pytest.approx([0.5, 0.5])


def test_softmax_sums_to_one():
    assert sum(softmax([3.0, -1.0, 0.5, 7.0])) == APPROX(1.0)


def test_softmax_survives_huge_scores():
    """Без вычитания максимума exp(1000) даёт OverflowError."""
    assert softmax([1000.0, 0.0]) == pytest.approx([1.0, 0.0])


# ----------------------------------------------------------------- matvec
def test_matvec_through_the_identity_returns_the_vector():
    assert matvec([1.0, 2.0], IDENTITY) == pytest.approx([1.0, 2.0])


def test_matvec_sums_the_rows_it_is_told_to():
    assert matvec([1.0, 1.0], [[1.0, 2.0], [3.0, 4.0]]) == pytest.approx([4.0, 6.0])


def test_matvec_reads_rows_not_columns():
    """Ловушка размерностей: перепутанные индексы дают транспонированный ответ."""
    assert matvec([1.0, 0.0], [[1.0, 2.0], [3.0, 4.0]]) == pytest.approx([1.0, 2.0])


def test_matvec_can_change_the_dimension():
    assert matvec([1.0, 2.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]) == pytest.approx(
        [1.0, 2.0, 2.0]
    )


# -------------------------------------------------------------- attention
def test_attention_to_a_single_token_returns_its_value():
    assert attention([1.0, 0.0], [[1.0, 0.0]], [[5.0, 7.0]]) == pytest.approx([5.0, 7.0])


def test_a_zero_query_averages_every_value_equally():
    out = attention([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 4.0]])
    assert out == pytest.approx([1.0, 2.0])


def test_attention_leans_toward_the_matching_key():
    """Запрос совпал с первым ключом — его value весит больше."""
    out = attention([3.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0], [0.0]])
    assert out[0] > 0.5


def test_attention_output_is_a_convex_mix_of_the_values():
    """Веса неотрицательны и в сумме дают 1, значит выход не выходит за пределы values."""
    out = attention([0.7, -0.2], [[1.0, 0.0], [0.0, 1.0]], [[2.0], [8.0]])
    assert 2.0 < out[0] < 8.0


def test_attention_divides_the_scores_by_sqrt_of_the_dimension():
    """Забудешь sqrt(d) — вес окажется 1/(1+e^-16) вместо 1/(1+e^-8), молча и мимо."""
    out = attention([2.0, 2.0, 2.0, 2.0], [[2.0] * 4, [0.0] * 4], [[1.0], [0.0]])
    assert out[0] == pytest.approx(1 / (1 + math.exp(-8)), abs=1e-12)


# ------------------------------------------------------- KV-кэш: коррект.
def test_kv_cache_gives_exactly_the_same_outputs_as_a_full_recompute():
    """Главный контракт урока: кэш — чистая экономия, а не приближение."""
    slow = generate_no_cache(TOKENS, W_K, W_V)
    fast = generate_with_cache(TOKENS, W_K, W_V)
    assert flat(fast["outputs"]) == pytest.approx(flat(slow["outputs"]), abs=1e-12)


def test_kv_cache_matches_on_a_single_token_too():
    slow = generate_no_cache(TOKENS[:1], W_K, W_V)
    fast = generate_with_cache(TOKENS[:1], W_K, W_V)
    assert flat(fast["outputs"]) == pytest.approx(flat(slow["outputs"]))


def test_generation_is_causal_and_the_first_output_never_changes():
    """Дописали токенов в конец — первый выход обязан остаться прежним."""
    short = generate_with_cache(TOKENS[:2], W_K, W_V)["outputs"]
    long = generate_with_cache(TOKENS, W_K, W_V)["outputs"]
    assert long[0] == pytest.approx(short[0])
    assert long[1] == pytest.approx(short[1])


def test_generation_produces_one_output_per_token():
    assert len(generate_with_cache(TOKENS, W_K, W_V)["outputs"]) == len(TOKENS)


# ---------------------------------------------------- KV-кэш: экономия
def test_recomputing_projections_grows_quadratically():
    n = len(TOKENS)
    assert generate_no_cache(TOKENS, W_K, W_V)["projections"] == n * (n + 1)


def test_the_cache_projects_every_token_exactly_once():
    assert generate_with_cache(TOKENS, W_K, W_V)["projections"] == 2 * len(TOKENS)


def test_the_saving_grows_with_the_sequence_length():
    """На двух токенах экономия скромная, на шести — уже втрое."""
    short = generate_no_cache(TOKENS[:2], W_K, W_V)["projections"] / generate_with_cache(
        TOKENS[:2], W_K, W_V
    )["projections"]
    long = generate_no_cache(TOKENS, W_K, W_V)["projections"] / generate_with_cache(
        TOKENS, W_K, W_V
    )["projections"]
    assert long > short > 1.0


# --------------------------------------------------------- kv_cache_bytes
def test_llama_70b_spends_320_kilobytes_per_token():
    assert kv_cache_bytes(seq_len=1, **LLAMA70B) == 327680


def test_a_128k_context_eats_40_gigabytes():
    assert kv_cache_bytes(seq_len=131072, **LLAMA70B) / 1024 ** 3 == APPROX(40.0)


def test_kv_cache_grows_linearly_with_the_context():
    assert kv_cache_bytes(seq_len=8192, **LLAMA70B) == 2 * kv_cache_bytes(
        seq_len=4096, **LLAMA70B
    )


def test_gqa_is_where_the_eightfold_saving_comes_from():
    """64 головы запроса против 8 голов KV — считать надо по вторым."""
    gqa = kv_cache_bytes(80, 8, 128, 4096)
    full_mha = kv_cache_bytes(80, 64, 128, 4096)
    assert full_mha == 8 * gqa


def test_int8_kv_cache_halves_the_memory():
    fp16 = kv_cache_bytes(seq_len=4096, dtype_bytes=2, **LLAMA70B)
    int8 = kv_cache_bytes(seq_len=4096, dtype_bytes=1, **LLAMA70B)
    assert int8 * 2 == fp16


# --------------------------------------------------------- batching_steps
def test_uniform_lengths_give_continuous_batching_nothing():
    """Честный результат: разнобоя нет — выигрыша нет."""
    out = batching_steps([10, 10, 10, 10], 2)
    assert out["static"] == out["continuous"] == 20
    assert out["speedup"] == APPROX(1.0)


def test_continuous_batching_refills_the_freed_slot():
    out = batching_steps([50, 10, 10, 10], 2)
    assert out["static"] == 60
    assert out["continuous"] == 50
    assert out["speedup"] == pytest.approx(1.2)


def test_continuous_batching_is_never_worse_than_static():
    """Свойство, а не совпадение: проверяем на десятке случайных нагрузок."""
    rng = random.Random(1)
    for _ in range(10):
        lens = [rng.randint(1, 200) for _ in range(20)]
        out = batching_steps(lens, 4)
        assert out["continuous"] <= out["static"]


def test_the_gain_grows_with_the_spread_of_output_lengths():
    """Именно разброс длин — источник выигрыша, а не число запросов."""
    even = batching_steps([30] * 8, 2)["speedup"]
    spread = batching_steps([200, 5, 5, 5, 200, 5, 5, 5], 2)["speedup"]
    assert spread > even


def test_batching_an_empty_queue_takes_no_steps():
    assert batching_steps([], 4)["static"] == 0


# ---------------------------------------------------- speculative_speedup
def test_a_never_accepted_draft_only_costs_extra():
    """Ловушка: черновик оплачен, толку ноль — ускорение меньше единицы."""
    out = speculative_speedup(5, 0.0)
    assert out["expected_accepted"] == APPROX(0.0)
    assert out["tokens_per_round"] == APPROX(1.0)
    assert out["speedup"] == pytest.approx(10 / 15)


def test_a_perfect_draft_accepts_every_candidate():
    out = speculative_speedup(5, 1.0)
    assert out["expected_accepted"] == APPROX(5.0)
    assert out["speedup"] == pytest.approx(4.0)


def test_expected_accepted_is_the_geometric_sum():
    """Первый отказ обрывает цепочку, поэтому p + p^2 + ... + p^K, а не K*p."""
    p = 0.8
    expected = sum(p ** k for k in range(1, 5))
    assert speculative_speedup(4, p)["expected_accepted"] == pytest.approx(expected)


def test_speedup_grows_with_the_acceptance_rate():
    rates = [0.2, 0.5, 0.78, 0.9]
    speedups = [speculative_speedup(5, r)["speedup"] for r in rates]
    assert all(a < b for a, b in zip(speedups, speedups[1:]))


def test_a_realistic_draft_target_pair_beats_sequential_decoding():
    """8B, черновик к 70B: доля принятия 0.78 — это заявленные в уроке 2-3x."""
    assert speculative_speedup(5, 0.78)["speedup"] > 1.5


def test_a_cheaper_draft_model_is_worth_more():
    cheap = speculative_speedup(5, 0.8, draft_cost=0.5)["speedup"]
    pricey = speculative_speedup(5, 0.8, draft_cost=4.0)["speedup"]
    assert cheap > pricey
