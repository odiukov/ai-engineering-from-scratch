"""Тесты к уроку «Квантование: как уместить модель». Правь exercise.py."""

import random

import pytest

from exercise import (
    dequantize,
    dequantize_asymmetric,
    dequantize_per_channel,
    model_memory_gb,
    quantization_error,
    quantize_asymmetric,
    quantize_per_channel,
    quantize_symmetric,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу."""
    return [v for row in M for v in row]


def roundtrip(values, num_bits=8):
    q, scale = quantize_symmetric(values, num_bits)
    return dequantize(q, scale)


# Матрица с одной строкой-выбросом: ровно тот случай, ради которого
# придумали поканальный масштаб.
OUTLIER_MATRIX = [
    [0.010, -0.020, 0.015, -0.005],
    [0.008, 0.012, -0.018, 0.004],
    [1.000, -2.000, 1.500, -0.500],
]

_rng = random.Random(20260807)
WEIGHTS = [_rng.gauss(0.0, 0.02) for _ in range(400)]


# ------------------------------------------------------ quantize_symmetric
def test_quantize_symmetric_uses_the_full_int8_range():
    q, scale = quantize_symmetric([0.0, 1.0, -1.0], 8)
    assert q == [0, 127, -127]
    assert scale == APPROX(1 / 127)


def test_quantize_symmetric_survives_an_all_zero_tensor():
    """Слой из одних нулей бывает; деления на ноль быть не должно."""
    assert quantize_symmetric([0.0, 0.0], 8) == ([0, 0], 1.0)


def test_quantize_symmetric_scales_by_qmax_not_qmin():
    """Ловушка диапазона: 4 бита — это [-8, 7]. Масштаб считаем по 7."""
    q, scale = quantize_symmetric([2.0, -2.0], 4)
    assert q == [7, -7]
    assert scale == APPROX(2 / 7)


def test_quantize_symmetric_never_exceeds_the_integer_range():
    """4 бита — это [-8, 7]. Наружу не должно вылезти ни одно значение."""
    q, _ = quantize_symmetric(WEIGHTS, 4)
    assert all(-8 <= v <= 7 for v in q)


def test_quantization_preserves_the_order_of_magnitudes():
    """Порядок сохраняется: отсортированный вход даёт неубывающие целые."""
    values = sorted(WEIGHTS)
    q, _ = quantize_symmetric(values, 8)
    assert all(a <= b for a, b in zip(q, q[1:]))


def test_quantization_preserves_signs():
    q, _ = quantize_symmetric(WEIGHTS, 8)
    assert all((v > 0) == (x > 0) or x == 0 for v, x in zip(q, WEIGHTS) if abs(x) > 1e-4)


# ------------------------------------------------------------- dequantize
def test_dequantize_is_a_single_multiplication_by_the_scale():
    assert dequantize([0, 127, -64], 0.5) == pytest.approx([0.0, 63.5, -32.0])


def test_roundtrip_error_never_exceeds_half_a_step():
    """Ошибка округления ограничена scale/2 — это вся теория погрешности."""
    _, scale = quantize_symmetric(WEIGHTS, 8)
    restored = roundtrip(WEIGHTS, 8)
    assert max(abs(a - b) for a, b in zip(WEIGHTS, restored)) <= scale / 2 + 1e-12


def test_roundtrip_keeps_the_extreme_value_intact():
    """Самый большой по модулю вес восстанавливается точно: по нему считали scale."""
    values = [0.3, -0.7, 0.1]
    assert roundtrip(values, 8)[1] == pytest.approx(-0.7, abs=1e-9)


def test_more_bits_mean_less_damage():
    """Кривая качества монотонна — на этом строят выбор разрядности."""
    errors = [
        quantization_error(WEIGHTS, roundtrip(WEIGHTS, bits))["mse"]
        for bits in (2, 3, 4, 8)
    ]
    assert all(a > b for a, b in zip(errors, errors[1:]))


def test_each_extra_bit_adds_about_six_decibels():
    """Правило «бит = 6 dB» — то, чем на глаз прикидывают потерю качества."""
    snr4 = quantization_error(WEIGHTS, roundtrip(WEIGHTS, 4))["snr_db"]
    snr8 = quantization_error(WEIGHTS, roundtrip(WEIGHTS, 8))["snr_db"]
    assert 4 * 6 - 6 < snr8 - snr4 < 4 * 6 + 6


# ----------------------------------------------------- quantize_asymmetric
def test_quantize_asymmetric_stretches_the_range_over_all_levels():
    q, scale, zero_point = quantize_asymmetric([0.0, 1.0], 8)
    assert (q, zero_point) == ([0, 255], 0)
    assert scale == APPROX(1 / 255)


def test_asymmetric_roundtrip_error_never_exceeds_half_a_step():
    values = [0.0, 0.3, 0.7, 1.0, 0.55]
    q, scale, zp = quantize_asymmetric(values, 8)
    restored = dequantize_asymmetric(q, scale, zp)
    assert max(abs(a - b) for a, b in zip(values, restored)) <= scale / 2 + 1e-12


def test_asymmetric_beats_symmetric_on_non_negative_data():
    """Активации после ReLU: симметричная схема дарит половину диапазона впустую."""
    values = [i / 20 + 1.0 for i in range(21)]

    q_sym, scale_sym = quantize_symmetric(values, 4)
    err_sym = quantization_error(values, dequantize(q_sym, scale_sym))["mse"]

    q_asym, scale_asym, zp = quantize_asymmetric(values, 4)
    err_asym = quantization_error(values, dequantize_asymmetric(q_asym, scale_asym, zp))["mse"]

    assert err_asym < err_sym


def test_asymmetric_does_not_collapse_positive_data_to_a_constant():
    """Ловушка zero_point: без расширения диапазона до нуля все q упрутся в qmax."""
    values = [1.0 + i / 10 for i in range(11)]
    q, _, _ = quantize_asymmetric(values, 8)
    assert len(set(q)) > 1


def test_asymmetric_represents_zero_exactly():
    """Ноль обязан лечь ровно в zero_point и восстановиться без ошибки."""
    values = [-1.0, 0.0, 2.0]
    q, scale, zp = quantize_asymmetric(values, 8)
    assert q[1] == zp
    assert dequantize_asymmetric(q, scale, zp)[1] == APPROX(0.0)


def test_asymmetric_survives_an_all_zero_tensor():
    assert quantize_asymmetric([0.0, 0.0, 0.0], 8) == ([0, 0, 0], 1.0, 0)


# ---------------------------------------------------- quantize_per_channel
def test_per_channel_gives_every_row_its_own_scale():
    q, scales = quantize_per_channel([[1.0, -1.0], [100.0, -100.0]], 8)
    assert flat(q) == [127, -127, 127, -127]
    assert scales == pytest.approx([1 / 127, 100 / 127])


def test_per_channel_roundtrip_restores_the_matrix():
    q, scales = quantize_per_channel(OUTLIER_MATRIX, 8)
    restored = dequantize_per_channel(q, scales)
    assert flat(restored) == pytest.approx(flat(OUTLIER_MATRIX), abs=2e-2)


def test_per_channel_beats_per_tensor_when_one_row_is_an_outlier():
    """Главный аргумент урока: один общий scale убивает мелкие строки."""
    values = flat(OUTLIER_MATRIX)

    q_pt, scale_pt = quantize_symmetric(values, 4)
    err_pt = quantization_error(values, dequantize(q_pt, scale_pt))["mse"]

    q_pc, scales_pc = quantize_per_channel(OUTLIER_MATRIX, 4)
    err_pc = quantization_error(values, flat(dequantize_per_channel(q_pc, scales_pc)))["mse"]

    assert err_pc < err_pt


def test_per_tensor_quantization_wipes_out_the_small_rows():
    """При 4 битах общий scale обнуляет все веса тихой строки целиком."""
    values = flat(OUTLIER_MATRIX)
    q_pt, _ = quantize_symmetric(values, 4)
    assert q_pt[:4] == [0, 0, 0, 0]


def test_per_channel_does_not_reuse_the_first_scale_everywhere():
    """Ловушка: применил scales[0] ко всей матрице — получил per-tensor и не заметил."""
    q, scales = quantize_per_channel(OUTLIER_MATRIX, 8)
    restored = dequantize_per_channel(q, scales)
    assert restored[2][1] == pytest.approx(-2.0, abs=1e-2)


# ---------------------------------------------------- quantization_error
def test_quantization_error_of_a_perfect_copy():
    err = quantization_error([1.0, 2.0, -3.0], [1.0, 2.0, -3.0])
    assert err["mse"] == APPROX(0.0)
    assert err["rmse"] == APPROX(0.0)
    assert err["cosine_similarity"] == APPROX(1.0)


def test_quantization_error_does_not_divide_by_zero_on_a_perfect_copy():
    """mse == 0 в знаменателе SNR — типичный ZeroDivisionError."""
    assert quantization_error([1.0, 2.0], [1.0, 2.0])["snr_db"] > 100


def test_rmse_is_the_square_root_of_mse():
    err = quantization_error([1.0, 2.0, -3.0], [1.1, 1.7, -3.4])
    assert err["rmse"] ** 2 == pytest.approx(err["mse"])


def test_cosine_similarity_ignores_a_uniform_rescale():
    """Косинус смотрит на направление: сжатый вдвое вектор всё ещё коллинеарен."""
    original = [1.0, 2.0, 3.0]
    halved = [0.5, 1.0, 1.5]
    err = quantization_error(original, halved)
    assert err["cosine_similarity"] == pytest.approx(1.0)
    assert err["mse"] > 0


def test_max_error_is_the_worst_single_weight():
    err = quantization_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.5])
    assert err["max_error"] == pytest.approx(0.5)


def test_snr_falls_as_the_reconstruction_gets_worse():
    good = quantization_error(WEIGHTS, roundtrip(WEIGHTS, 8))["snr_db"]
    bad = quantization_error(WEIGHTS, roundtrip(WEIGHTS, 2))["snr_db"]
    assert good > bad


# ------------------------------------------------------- model_memory_gb
def test_llama_70b_in_fp16_needs_about_130_gigabytes():
    assert model_memory_gb(70, 16) == pytest.approx(130.385, abs=0.01)


def test_int4_quarters_the_memory_of_fp16():
    assert model_memory_gb(70, 4) == pytest.approx(model_memory_gb(70, 16) / 4)


def test_int4_puts_a_70b_model_on_a_single_48gb_card():
    """Ровно тот вывод, ради которого квантование и существует."""
    assert model_memory_gb(70, 16) > 48 > model_memory_gb(70, 4)


def test_memory_is_linear_in_parameter_count():
    assert model_memory_gb(14, 8) == pytest.approx(2 * model_memory_gb(7, 8))
