"""Тесты к уроку «Квантование в проде: AWQ, GPTQ, GGUF, FP8, NVFP4». Правь exercise.py."""

import pytest

from exercise import (
    FORMATS,
    FormatUnsupportedError,
    blockwise_roundtrip,
    dequantize,
    format_memory_gb,
    pick_format,
    quant_params,
    quantization_error,
    quantize,
    roundtrip,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ----------------------------------------------------------- quant_params
def test_scale_covers_the_whole_range_with_the_available_codes():
    assert quant_params([0.0, 15.0], 4) == (APPROX(1.0), 0)


def test_zero_point_shifts_a_signed_range_onto_unsigned_codes():
    assert quant_params([-8.0, 7.0], 4) == (APPROX(1.0), 8)


def test_range_is_extended_to_keep_zero_representable():
    """Все веса положительные, а ноль всё равно обязан попасть в сетку."""
    scale, zero_point = quant_params([10.0, 20.0], 4)
    assert zero_point == 0
    assert scale == APPROX(20.0 / 15)


def test_more_bits_give_a_finer_step():
    scale4, _ = quant_params([-1.0, 1.0], 4)
    scale8, _ = quant_params([-1.0, 1.0], 8)
    assert scale8 < scale4 / 10


def test_constant_tensor_does_not_produce_a_zero_scale():
    """Иначе первое же деление на scale уронит квантование."""
    scale, zero_point = quant_params([3.0, 3.0, 3.0], 4)
    assert scale != 0


# --------------------------------------------------------------- quantize
def test_codes_land_on_the_expected_grid_points():
    assert quantize([-8.0, 0.0, 7.0], 1.0, 8, 4) == [0, 8, 15]


def test_out_of_range_values_saturate_instead_of_wrapping():
    """Переполнение 4-битного кода превратило бы большой вес в мусор."""
    assert quantize([-99.0, 99.0], 1.0, 8, 4) == [0, 15]


def test_codes_never_leave_the_bit_width():
    codes = quantize([-100.0, -3.0, 0.0, 3.0, 100.0], 0.5, 8, 4)
    assert all(0 <= c <= 15 for c in codes)


# ------------------------------------------------------------- dequantize
def test_dequantize_undoes_quantize_on_grid_values():
    assert dequantize([0, 8, 15], 1.0, 8) == APPROX([-8.0, 0.0, 7.0])


def test_dequantize_is_linear_in_the_code():
    values = dequantize([0, 1, 2, 3], 0.25, 0)
    assert values == APPROX([0.0, 0.25, 0.5, 0.75])


# --------------------------------------------------------------- roundtrip
def test_values_on_the_grid_survive_untouched():
    assert roundtrip([-8.0, 0.0, 7.0], 4) == APPROX([-8.0, 0.0, 7.0])


def test_roundtrip_keeps_the_order_of_the_weights():
    """Разрешение теряется, но порядок величин — нет: сортировка не ломается."""
    values = [-3.0, -0.5, 0.0, 0.2, 1.0, 4.5]
    out = roundtrip(values, 4)
    assert all(a <= b for a, b in zip(out, out[1:]))


def test_roundtrip_never_leaves_the_range_by_more_than_half_a_step():
    """Округление может чуть вынести вес за исходный диапазон, но только на полшага."""
    values = [-2.0, 0.3, 1.7]
    half_step = quantization_error(values, 4)["scale"] / 2
    out = roundtrip(values, 4)
    assert min(out) >= min(values) - half_step - 1e-9
    assert max(out) <= max(values) + half_step + 1e-9


def test_roundtrip_of_zeros_is_zeros():
    assert roundtrip([0.0, 0.0, 0.0], 4) == APPROX([0.0, 0.0, 0.0])


# ------------------------------------------------------- quantization_error
def test_error_never_exceeds_half_a_step():
    values = [-1.0 + i * 0.07 for i in range(30)]
    err = quantization_error(values, 4)
    assert err["max_abs"] <= err["scale"] / 2 + 1e-9


def test_int8_beats_int4_by_more_than_an_order_of_magnitude():
    values = [-1.0 + i * 0.013 for i in range(150)]
    assert quantization_error(values, 8)["max_abs"] * 10 < quantization_error(values, 4)["max_abs"]


def test_a_single_outlier_destroys_the_resolution_of_everything_else():
    """Ровно за это AWQ и защищает salient-веса: шкалу задаёт хвост распределения."""
    normal = [0.0, 0.25, 0.5, 0.75, 1.0]
    with_outlier = normal + [100.0]
    assert quantization_error(normal, 4)["max_abs"] < 0.05
    assert quantization_error(with_outlier, 4)["max_abs"] > 0.9


def test_mean_error_is_never_worse_than_the_max():
    values = [0.0, 1.0, 100.0]
    err = quantization_error(values, 4)
    assert err["mean_abs"] <= err["max_abs"]


# --------------------------------------------------- blockwise_roundtrip
def test_per_block_scale_saves_the_small_weights_from_the_outlier():
    """Микромасштабирование NVFP4: выброс сидит в своём блоке и не портит соседей."""
    values = [0.0, 1.0, 0.0, 100.0]
    assert blockwise_roundtrip(values, 4, 2) == APPROX(values)
    assert roundtrip(values, 4)[1] == APPROX(0.0)


def test_blockwise_error_is_never_worse_than_per_tensor():
    values = [0.02 * i for i in range(16)] + [50.0]
    per_tensor = max(abs(a - v) for a, v in zip(roundtrip(values, 4), values))
    blocked = max(abs(a - v) for a, v in zip(blockwise_roundtrip(values, 4, 8), values))
    assert blocked < per_tensor


def test_one_giant_block_is_the_same_as_no_blocks_at_all():
    values = [0.0, 1.0, 0.0, 100.0]
    assert blockwise_roundtrip(values, 4, len(values)) == APPROX(roundtrip(values, 4))


def test_zero_block_size_is_a_call_error():
    with pytest.raises(ValueError):
        blockwise_roundtrip([1.0, 2.0], 4, 0)


# ------------------------------------------------------- format_memory_gb
def test_int4_weights_are_a_quarter_of_bf16():
    bf16 = format_memory_gb(70, 16, 16, 1, 2048)
    awq = format_memory_gb(70, 4, 16, 1, 2048)
    assert bf16["weights"] == APPROX(140.0)
    assert awq["weights"] == APPROX(35.0)


def test_kv_cache_at_production_batch_is_larger_than_int4_weights():
    """«Модель теперь 35 GB» — а KV-кэш на 128 сессиях по 2k уже больше 80 GB."""
    m = format_memory_gb(70, 4, 16, 128, 2048)
    assert m["kv"] > m["weights"]
    assert m["kv"] == pytest.approx(85.9, abs=0.1)


def test_quantizing_weights_does_not_move_the_kv_cache_at_all():
    awq = format_memory_gb(70, 4, 16, 128, 2048)
    bf16 = format_memory_gb(70, 16, 16, 128, 2048)
    assert awq["kv"] == APPROX(bf16["kv"])


def test_awq_alone_does_not_fit_a_70b_on_one_h100():
    """Урок обещает ~60 GB. Посчитанный целиком бюджет в 80 GB не влезает."""
    assert format_memory_gb(70, 4, 16, 128, 2048)["total"] > 80


def test_fp8_kv_is_what_actually_buys_the_headroom():
    with_fp16_kv = format_memory_gb(70, 4, 16, 128, 2048)["total"]
    with_fp8_kv = format_memory_gb(70, 4, 8, 128, 2048)["total"]
    assert with_fp16_kv - with_fp8_kv == pytest.approx(42.9, abs=0.1)


def test_kv_cache_grows_with_concurrency_and_context_together():
    base = format_memory_gb(70, 4, 16, 128, 2048)["kv"]
    wider = format_memory_gb(70, 4, 16, 256, 8192)["kv"]
    assert wider == APPROX(base * 8)


# ------------------------------------------------------------ pick_format
def test_edge_target_gets_gguf():
    assert pick_format("edge") == "GGUF Q4_K_M"


def test_multi_lora_on_hopper_can_use_awq():
    assert pick_format("hopper", needs_lora=True) == "AWQ-Int4"


def test_forced_awq_with_lora_is_supported():
    assert pick_format("hopper", needs_lora=True, forced="AWQ-Int4") == "AWQ-Int4"


def test_reasoning_workload_gets_fp8_even_on_blackwell():
    assert pick_format("blackwell", reasoning_heavy=True) == "FP8"


def test_blackwell_default_is_the_four_bit_microscaling_format():
    assert pick_format("blackwell") == "NVFP4 + FP8 KV"


def test_lora_on_top_of_nvfp4_is_rejected_by_name():
    with pytest.raises(FormatUnsupportedError):
        pick_format("blackwell", needs_lora=True, forced="NVFP4 + FP8 KV")


def test_gguf_path_does_not_claim_multi_lora_serving():
    with pytest.raises(FormatUnsupportedError):
        pick_format("edge", needs_lora=True, forced="GGUF Q4_K_M")


def test_unknown_format_name_is_a_value_error():
    with pytest.raises(ValueError):
        pick_format("hopper", forced="AWQ-Int2")


def test_every_picked_format_is_a_known_one():
    picks = [
        pick_format("cpu"),
        pick_format("edge", reasoning_heavy=True),
        pick_format("hopper"),
        pick_format("hopper", reasoning_heavy=True),
        pick_format("blackwell"),
        pick_format("blackwell", needs_lora=True),
    ]
    assert all(p in FORMATS for p in picks)
