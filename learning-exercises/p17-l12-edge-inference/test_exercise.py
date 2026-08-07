"""Тесты к уроку «Инференс на устройстве: ANE, Hexagon, WebGPU, Jetson». Правь exercise.py."""

import pytest

from exercise import (
    EDGE_MODELS,
    EDGE_TARGETS,
    LPDDR5_PJ_PER_BYTE,
    EdgeBudgetError,
    decode_ceiling_tps,
    energy_per_token_j,
    fits_on_device,
    kv_cache_gb,
    max_context_tokens,
    roofline_regime,
    roofline_times,
    weights_gb,
)

APPROX = lambda x: pytest.approx(x, rel=1e-9)

LLAMA8B = EDGE_MODELS["llama-3.1-8b"]
IPHONE = EDGE_TARGETS["iphone-16"]
SNAPDRAGON = EDGE_TARGETS["snapdragon-8-gen-3"]
H100 = EDGE_TARGETS["h100"]


# ------------------------------------------------------------- weights_gb
def test_bf16_weights_of_an_8b_model():
    assert weights_gb(8.0, 16) == APPROX(16.0)


def test_q4_weights_are_a_quarter_of_bf16():
    assert weights_gb(8.0, 4) == APPROX(4.0)


def test_zero_bits_is_a_call_error():
    with pytest.raises(ValueError):
        weights_gb(8.0, 0)


# ------------------------------------------------------ decode_ceiling_tps
def test_seven_b_q4_on_mobile_dram_tops_out_at_fourteen_tokens():
    assert decode_ceiling_tps(3.5, 50.0) == pytest.approx(14.28, abs=0.01)


def test_the_same_model_on_hbm3_is_two_orders_faster():
    assert decode_ceiling_tps(3.5, 3000.0) == pytest.approx(857.1, abs=0.1)


def test_the_datacenter_edge_gap_is_the_bandwidth_gap():
    """30-50x между HBM и мобильной DRAM — это ровно отношение пропускных способностей."""
    model = weights_gb(LLAMA8B["params_b"], 4)
    gap = decode_ceiling_tps(model, H100["bandwidth_gb_s"]) / decode_ceiling_tps(
        model, SNAPDRAGON["bandwidth_gb_s"])
    assert 30 <= gap <= 50
    assert gap == APPROX(H100["bandwidth_gb_s"] / SNAPDRAGON["bandwidth_gb_s"])


def test_halving_the_weight_bits_doubles_the_ceiling():
    q4 = decode_ceiling_tps(weights_gb(8.0, 4), 60.0)
    q8 = decode_ceiling_tps(weights_gb(8.0, 8), 60.0)
    assert q4 == APPROX(2 * q8)


def test_a_model_of_zero_size_is_a_call_error():
    with pytest.raises(ValueError):
        decode_ceiling_tps(0.0, 60.0)


# ------------------------------------------------------------ kv_cache_gb
def test_kv_cache_of_a_4k_session():
    assert kv_cache_gb(4096, 32, 8, 128, 16) == APPROX(0.536870912)


def test_kv_cache_is_linear_in_context_length():
    assert kv_cache_gb(32768, 32, 8, 128, 16) == APPROX(
        8 * kv_cache_gb(4096, 32, 8, 128, 16)
    )


def test_a_32k_kv_cache_outgrows_the_q4_model_itself():
    """Именно поэтому длинный контекст на телефоне — датацентровая фича."""
    assert kv_cache_gb(32768, 32, 8, 128, 16) > weights_gb(8.0, 4)


# --------------------------------------------------------- roofline_times
def test_decode_reads_the_whole_model_for_a_single_token():
    times = roofline_times(1, LLAMA8B, IPHONE, 4)
    assert times["memory_s"] == APPROX(4.0 / 60.0)


def test_memory_time_does_not_depend_on_how_many_tokens_are_in_the_pass():
    one = roofline_times(1, LLAMA8B, IPHONE, 4)["memory_s"]
    many = roofline_times(512, LLAMA8B, IPHONE, 4)["memory_s"]
    assert one == APPROX(many)


def test_compute_time_is_linear_in_tokens():
    one = roofline_times(1, LLAMA8B, IPHONE, 4)["compute_s"]
    many = roofline_times(512, LLAMA8B, IPHONE, 4)["compute_s"]
    assert many == APPROX(512 * one)


def test_a_pass_with_less_than_one_token_is_a_call_error():
    with pytest.raises(ValueError):
        roofline_times(0, LLAMA8B, IPHONE, 4)


# -------------------------------------------------------- roofline_regime
def test_decode_is_memory_bound():
    assert roofline_regime(1, LLAMA8B, IPHONE, 4) == "memory"


def test_prefill_is_compute_bound():
    assert roofline_regime(512, LLAMA8B, IPHONE, 4) == "compute"


def test_the_regime_flips_somewhere_between_decode_and_prefill():
    """Точка перелома — арифметическая интенсивность, а не «маленький/большой запрос»."""
    regimes = [roofline_regime(n, LLAMA8B, IPHONE, 4) for n in (1, 8, 64, 512, 4096)]
    assert regimes[0] == "memory"
    assert regimes[-1] == "compute"
    assert regimes.index("compute") > 0


def test_decode_stays_memory_bound_even_on_a_datacenter_gpu():
    """Роль памяти в decode — не особенность edge, а свойство самого decode."""
    assert roofline_regime(1, LLAMA8B, H100, 4) == "memory"


def test_more_tops_never_speeds_up_a_memory_bound_pass():
    slow = roofline_times(1, LLAMA8B, IPHONE, 4)
    fast_npu = dict(IPHONE, tops=IPHONE["tops"] * 100)
    boosted = roofline_times(1, LLAMA8B, fast_npu, 4)
    assert boosted["memory_s"] == APPROX(slow["memory_s"])
    assert roofline_regime(1, LLAMA8B, fast_npu, 4) == "memory"


# ------------------------------------------------------ max_context_tokens
def test_context_that_fits_into_one_and_a_half_gigabytes():
    assert max_context_tokens(1.5, LLAMA8B, 16) == 11444


def test_fp8_kv_cache_doubles_the_context_window():
    assert max_context_tokens(1.5, LLAMA8B, 8) == 2 * max_context_tokens(1.5, LLAMA8B, 16)


def test_context_is_rounded_down_to_whole_tokens():
    tokens = max_context_tokens(0.001, LLAMA8B, 16)
    assert tokens == int(tokens)
    assert kv_cache_gb(tokens, 32, 8, 128, 16) <= 0.001


def test_negative_free_memory_is_a_call_error():
    with pytest.raises(ValueError):
        max_context_tokens(-1.0, LLAMA8B, 16)


# --------------------------------------------------------- fits_on_device
def test_q4_model_with_a_4k_window_fits_an_iphone():
    budget = fits_on_device(IPHONE, LLAMA8B, 4, 4096, 16)
    assert budget["free_gb"] == pytest.approx(0.963, abs=0.001)


def test_the_budget_adds_up_to_the_ram_minus_the_os():
    budget = fits_on_device(IPHONE, LLAMA8B, 4, 4096, 16)
    total = budget["model_gb"] + budget["kv_gb"] + budget["free_gb"]
    assert total == APPROX(IPHONE["ram_gb"] - IPHONE["os_overhead_gb"])


def test_a_32k_window_blows_the_phone_budget():
    with pytest.raises(EdgeBudgetError):
        fits_on_device(IPHONE, LLAMA8B, 4, 32768, 16)


def test_only_aggressive_kv_quantization_rescues_the_long_window():
    """32K на телефоне: FP16 и FP8 KV не помещаются, помещается только Q4 KV."""
    with pytest.raises(EdgeBudgetError):
        fits_on_device(IPHONE, LLAMA8B, 4, 32768, 16)
    with pytest.raises(EdgeBudgetError):
        fits_on_device(IPHONE, LLAMA8B, 4, 32768, 8)
    assert fits_on_device(IPHONE, LLAMA8B, 4, 32768, 4)["free_gb"] > 0


def test_memory_runs_out_long_before_compute_does():
    """8B в BF16 телефон не поднимет, хотя посчитать один токен ему нечего делать."""
    with pytest.raises(EdgeBudgetError):
        fits_on_device(IPHONE, LLAMA8B, 16, 4096, 16)
    assert roofline_times(1, LLAMA8B, IPHONE, 16)["compute_s"] < 0.001


def test_a_smaller_model_is_what_actually_unlocks_the_phone():
    budget = fits_on_device(IPHONE, EDGE_MODELS["llama-3.2-3b"], 4, 8192, 16)
    assert budget["free_gb"] > 2.5


# ------------------------------------------------------ energy_per_token_j
def test_energy_of_one_token_on_lpddr5():
    assert energy_per_token_j(LLAMA8B, 4, LPDDR5_PJ_PER_BYTE) == APPROX(0.16)


def test_energy_halves_together_with_the_weight_bits():
    q4 = energy_per_token_j(LLAMA8B, 4, LPDDR5_PJ_PER_BYTE)
    q8 = energy_per_token_j(LLAMA8B, 8, LPDDR5_PJ_PER_BYTE)
    assert q8 == APPROX(2 * q4)


def test_fixed_overhead_does_not_scale_with_the_model():
    plain = energy_per_token_j(LLAMA8B, 4, LPDDR5_PJ_PER_BYTE)
    with_screen = energy_per_token_j(LLAMA8B, 4, LPDDR5_PJ_PER_BYTE, overhead_j=0.5)
    assert with_screen == APPROX(plain + 0.5)


def test_a_phone_battery_is_worth_a_few_hundred_thousand_tokens():
    joules = IPHONE["battery_wh"] * 3600.0
    tokens = joules / energy_per_token_j(LLAMA8B, 4, LPDDR5_PJ_PER_BYTE)
    assert 300_000 < tokens < 400_000


def test_negative_energy_per_byte_is_a_call_error():
    with pytest.raises(ValueError):
        energy_per_token_j(LLAMA8B, 4, -1.0)
