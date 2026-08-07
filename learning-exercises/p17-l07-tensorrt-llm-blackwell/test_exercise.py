"""Тесты к уроку «Компиляция под железо: FP8 и NVFP4 на Blackwell». Правь exercise.py."""

import pytest

from exercise import (
    MODEL_SHAPES,
    STACKS,
    NoStackFitsError,
    choose_stack,
    cost_per_million_tokens,
    decode_tokens_per_s,
    gpus_needed,
    hbm_footprint_gb,
    kv_cache_gb,
    stack_speedup,
    weights_gb,
)

APPROX = lambda x: pytest.approx(x, rel=1e-9)

L70 = MODEL_SHAPES["llama-70b"]
MOE120 = MODEL_SHAPES["gpt-oss-120b"]


# ------------------------------------------------------------- weights_gb
def test_bf16_weights_are_two_bytes_per_parameter():
    assert weights_gb(70, 16) == APPROX(140.0)


def test_nvfp4_weights_are_a_quarter_of_bf16():
    assert weights_gb(70, 4) == APPROX(weights_gb(70, 16) / 4)


def test_weight_bytes_are_linear_in_parameter_count():
    assert weights_gb(140, 8) == APPROX(2 * weights_gb(70, 8))


def test_zero_bit_weights_are_a_call_error_not_a_free_model():
    with pytest.raises(ValueError):
        weights_gb(70, 0)


# ------------------------------------------------------------ kv_cache_gb
def test_kv_cache_of_a_single_2k_session_on_a_70b_shape():
    assert kv_cache_gb(80, 8, 128, 2048, 1, 16) == APPROX(0.67108864)


def test_kv_cache_grows_linearly_with_concurrency():
    one = kv_cache_gb(80, 8, 128, 2048, 1, 16)
    many = kv_cache_gb(80, 8, 128, 2048, 128, 16)
    assert many == APPROX(128 * one)


def test_fp8_kv_cache_is_half_of_fp16():
    assert kv_cache_gb(80, 8, 128, 2048, 64, 8) == APPROX(
        kv_cache_gb(80, 8, 128, 2048, 64, 16) / 2
    )


def test_kv_cache_outgrows_int4_weights_at_production_batch():
    """«Модель теперь 35 GB» — а KV-кэш на 128 сессиях уже больше весов."""
    kv = kv_cache_gb(80, 8, 128, 2048, 128, 16)
    assert kv > weights_gb(70, 4)


# ------------------------------------------------------ hbm_footprint_gb
def test_footprint_total_is_the_sum_of_its_parts():
    m = hbm_footprint_gb(L70, 4, 8, 2048, 128)
    assert m["total"] == APPROX(m["weights"] + m["kv"] + m["activations"])


def test_quantizing_weights_does_not_shrink_the_kv_cache():
    bf16 = hbm_footprint_gb(L70, 16, 16, 2048, 128)
    awq = hbm_footprint_gb(L70, 4, 16, 2048, 128)
    assert awq["weights"] == APPROX(bf16["weights"] / 4)
    assert awq["kv"] == APPROX(bf16["kv"])


def test_total_footprint_shrinks_far_less_than_fourfold():
    """Веса ужались вчетверо, а весь бюджет — меньше чем вдвое."""
    bf16 = hbm_footprint_gb(L70, 16, 16, 2048, 128)
    awq = hbm_footprint_gb(L70, 4, 16, 2048, 128)
    assert bf16["total"] / awq["total"] < 2.0


def test_kv_dominates_the_budget_at_long_context():
    m = hbm_footprint_gb(L70, 4, 8, 8192, 128)
    assert m["kv"] > m["weights"]


# ------------------------------------------------------------ gpus_needed
def test_gpus_needed_rounds_up():
    assert gpus_needed(230.0, 80) == 3


def test_exactly_full_gpu_is_still_one_gpu():
    assert gpus_needed(80.0, 80) == 1


def test_empty_model_needs_no_gpu():
    assert gpus_needed(0.0, 80) == 0


def test_gpu_of_zero_capacity_is_a_call_error():
    with pytest.raises(ValueError):
        gpus_needed(10.0, 0)


# --------------------------------------------------- decode_tokens_per_s
def test_decode_ceiling_on_h100_bf16():
    assert decode_tokens_per_s(70, 16, 3.35) == APPROX(3.35e12 / 140e9)


def test_halving_weight_bits_doubles_decode_speed():
    """Decode упирается в байты, а не во флопсы: вдвое меньше байт — вдвое быстрее."""
    assert decode_tokens_per_s(70, 8, 3.35) == APPROX(2 * decode_tokens_per_s(70, 16, 3.35))


def test_decode_speed_is_proportional_to_bandwidth():
    assert decode_tokens_per_s(70, 8, 8.0) == APPROX(
        decode_tokens_per_s(70, 8, 4.0) * 2
    )


def test_moe_decode_depends_on_active_parameters_only():
    """120B MoE с 36B активных читает столько же байт, сколько плотная 36B."""
    moe = decode_tokens_per_s(MOE120["active_b"], 4, 8.0)
    dense_36b = decode_tokens_per_s(36, 4, 8.0)
    assert moe == APPROX(dense_36b)


def test_zero_active_parameters_is_a_call_error():
    with pytest.raises(ValueError):
        decode_tokens_per_s(0, 8, 3.35)


# ---------------------------------------------------------- stack_speedup
def test_speedup_factors_multiply():
    assert stack_speedup([2.0, 1.8]) == APPROX(3.6)


def test_empty_stack_gives_speedup_of_one():
    assert stack_speedup([]) == APPROX(1.0)


def test_efficiency_below_one_lands_the_paper_number():
    """~14x на бумаге превращается в ~7x на реальном трафике."""
    paper = stack_speedup([2.39, 2.0, 1.8, 1.6])
    real = stack_speedup([2.39, 2.0, 1.8, 1.6], efficiency=0.5)
    assert paper > 13.0
    assert real == APPROX(paper / 2)


# -------------------------------------------------- cost_per_million_tokens
def test_cost_of_a_million_tokens_worked_example():
    assert cost_per_million_tokens(1000.0, 3.60) == APPROX(1.0)


def test_doubling_throughput_halves_the_price():
    assert cost_per_million_tokens(2000.0, 3.60) == APPROX(
        cost_per_million_tokens(1000.0, 3.60) / 2
    )


def test_zero_throughput_is_a_call_error_not_zero_cost():
    with pytest.raises(ValueError):
        cost_per_million_tokens(0.0, 3.60)


# ------------------------------------------------------------ choose_stack
def test_chat_workload_picks_the_blackwell_four_bit_stack():
    best = choose_stack(STACKS, L70, 2048, 128)
    assert best["name"] == "GB200 NVL72 + TRT-LLM + Dynamo"


def test_reasoning_workload_refuses_four_bit_weights():
    best = choose_stack(STACKS, L70, 2048, 128, workload="reasoning")
    assert best["name"] == "H200 + FP8 + vLLM"


def test_quality_constraint_costs_real_money():
    """Запрет на 4 бита — это не абстракция, а счёт в разы больше."""
    chat = choose_stack(STACKS, L70, 2048, 128)
    reasoning = choose_stack(STACKS, L70, 2048, 128, workload="reasoning")
    assert reasoning["cost_per_million"] > 5 * chat["cost_per_million"]


def test_single_gpu_limit_leaves_no_stack_for_a_405b_model():
    with pytest.raises(NoStackFitsError):
        choose_stack(STACKS, MODEL_SHAPES["llama-405b"], 2048, 128, max_gpus=1)
