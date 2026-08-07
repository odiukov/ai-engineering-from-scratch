"""Тесты к уроку «Real-time вывод на edge». Правь exercise.py."""

import random

import pytest

from exercise import (
    conv2d_flops,
    dequantize_int8,
    drop_warmup,
    latency_stats,
    linear_flops,
    model_flops,
    quantize_int8,
    throughput_fps,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------- drop_warmup
def test_drop_warmup_removes_exactly_the_first_k_measurements():
    assert drop_warmup([5.0, 4.9, 1.2, 1.1, 1.3], 2) == [1.2, 1.1, 1.3]


def test_drop_warmup_with_zero_keeps_everything():
    assert drop_warmup([1.0, 2.0], 0) == [1.0, 2.0]


def test_drop_warmup_does_not_mutate_the_input():
    """Замеры могут понадобиться ещё раз — портить их нельзя."""
    times = [5.0, 4.9, 1.2, 1.1]
    drop_warmup(times, 2)
    assert times == [5.0, 4.9, 1.2, 1.1]


def test_drop_warmup_returns_a_new_list_even_when_it_drops_nothing():
    times = [1.0, 2.0]
    assert drop_warmup(times, 0) is not times


def test_drop_warmup_refuses_to_leave_an_empty_sample():
    with pytest.raises(ValueError):
        drop_warmup([1.0, 2.0], 2)


# ----------------------------------------------------------- latency_stats
def test_latency_stats_uses_nearest_rank():
    stats = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["p50_ms"] == APPROX(2.0)
    assert stats["p95_ms"] == APPROX(4.0)
    assert stats["p99_ms"] == APPROX(4.0)
    assert stats["mean_ms"] == APPROX(2.5)


def test_latency_stats_of_a_single_measurement_is_that_measurement():
    stats = latency_stats([7.0])
    assert stats == {"p50_ms": 7.0, "p95_ms": 7.0, "p99_ms": 7.0, "mean_ms": 7.0}


def test_percentiles_are_monotonically_ordered():
    rng = random.Random(0)
    times = [rng.expovariate(0.2) for _ in range(500)]
    stats = latency_stats(times)
    assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]


def test_latency_stats_does_not_depend_on_measurement_order():
    rng = random.Random(1)
    times = [rng.uniform(1.0, 20.0) for _ in range(200)]
    shuffled = list(times)
    rng.shuffle(shuffled)
    assert latency_stats(shuffled) == latency_stats(times)


def test_p99_of_a_hundred_samples_is_not_the_maximum():
    """Ловушка урока: times[int(n * 0.99)] на n=100 даёт индекс 99 — максимум.

    Настоящий nearest-rank p99 стоит на индексе 98. На тяжёлом хвосте разница
    между «почти худшим» и «худшим» кадром и есть весь смысл отчёта.
    """
    times = [float(i) for i in range(100)]
    stats = latency_stats(times)
    assert stats["p99_ms"] == APPROX(98.0)
    assert stats["p95_ms"] == APPROX(94.0)
    assert stats["p50_ms"] == APPROX(49.0)


def test_latency_stats_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        latency_stats([])


# ---------------------------------------------------------- throughput_fps
def test_ten_milliseconds_is_a_hundred_frames_per_second():
    assert throughput_fps(10.0) == APPROX(100.0)


def test_doubling_the_batch_doubles_throughput_at_the_same_latency():
    """Латентность и пропускная способность — разные бюджеты."""
    assert throughput_fps(10.0, batch_size=4) == APPROX(
        4 * throughput_fps(10.0, batch_size=1)
    )


def test_throughput_refuses_zero_latency():
    with pytest.raises(ValueError):
        throughput_fps(0.0)


# ------------------------------------------------------------ conv2d_flops
def test_first_resnet_conv_flops():
    assert conv2d_flops(3, 64, 3, 224, 224) == 173408256


def test_depthwise_conv_is_c_in_times_cheaper():
    """Весь MobileNet держится на этом множителе."""
    full = conv2d_flops(64, 64, 3, 56, 56)
    depthwise = conv2d_flops(64, 64, 3, 56, 56, groups=64)
    assert full == 64 * depthwise


def test_flops_scale_linearly_with_output_resolution():
    small = conv2d_flops(16, 32, 3, 28, 28)
    big = conv2d_flops(16, 32, 3, 56, 56)
    assert big == 4 * small


def test_conv_flops_reject_channels_not_divisible_by_groups():
    with pytest.raises(ValueError):
        conv2d_flops(10, 32, 3, 8, 8, groups=4)


# ------------------------------------------------------------ linear_flops
def test_classifier_head_flops():
    assert linear_flops(1024, 1000) == 2048000


def test_linear_flops_are_symmetric_in_their_dimensions():
    assert linear_flops(512, 10) == linear_flops(10, 512)


# ------------------------------------------------------------- model_flops
def test_model_flops_sum_the_layers():
    layers = [
        {"type": "conv", "c_in": 3, "c_out": 8, "k": 3, "h_out": 32, "w_out": 32},
        {"type": "linear", "in_features": 8, "out_features": 10},
    ]
    assert model_flops(layers) == conv2d_flops(3, 8, 3, 32, 32) + linear_flops(8, 10)


def test_model_flops_do_not_depend_on_layer_order():
    layers = [
        {"type": "conv", "c_in": 3, "c_out": 8, "k": 3, "h_out": 32, "w_out": 32},
        {"type": "linear", "in_features": 8, "out_features": 10},
    ]
    assert model_flops(layers) == model_flops(list(reversed(layers)))


def test_empty_model_costs_nothing():
    assert model_flops([]) == 0


def test_groups_default_to_one_when_the_layer_omits_them():
    dense = {"type": "conv", "c_in": 16, "c_out": 16, "k": 3, "h_out": 8, "w_out": 8}
    assert model_flops([dense]) == conv2d_flops(16, 16, 3, 8, 8, groups=1)


def test_unknown_layer_type_is_loud_not_silent():
    """Молча пропущенный слой занизит бюджет, и модель не влезет в устройство."""
    with pytest.raises(ValueError):
        model_flops([{"type": "attention", "dim": 64}])


# ------------------------------------------------- quantize / dequantize int8
def test_quantize_maps_the_range_onto_the_full_int8_grid():
    qs, scale, zero_point = quantize_int8([0.0, 1.0, 2.0, 3.0])
    assert qs == [-128, -43, 42, 127]
    assert scale == APPROX(3.0 / 255)
    assert zero_point == -128


def test_round_trip_error_never_exceeds_one_scale_step():
    rng = random.Random(7)
    values = [rng.uniform(-4.0, 4.0) for _ in range(300)]
    qs, scale, zero_point = quantize_int8(values)
    restored = dequantize_int8(qs, scale, zero_point)
    assert all(abs(a - b) <= scale for a, b in zip(values, restored))


def test_round_trip_survives_an_all_positive_tensor():
    """Выход ReLU не содержит отрицательных чисел — именно тут асимметричная
    квантизация выигрывает у симметричной."""
    values = [2.0, 4.0, 6.0, 5.5, 0.25]
    qs, scale, zero_point = quantize_int8(values)
    restored = dequantize_int8(qs, scale, zero_point)
    assert all(abs(a - b) <= scale for a, b in zip(values, restored))


def test_zero_is_representable_exactly():
    """Паддинг и ReLU — это буквально нули; смещённый ноль течёт по всей карте."""
    values = [-1.5, 0.0, 3.0]
    qs, scale, zero_point = quantize_int8(values)
    restored = dequantize_int8(qs, scale, zero_point)
    assert qs[1] == zero_point
    assert restored[1] == APPROX(0.0)


def test_every_quantized_value_fits_in_a_signed_byte():
    rng = random.Random(3)
    values = [rng.gauss(0.0, 50.0) for _ in range(200)]
    qs, _, _ = quantize_int8(values)
    assert all(isinstance(q, int) and -128 <= q <= 127 for q in qs)


def test_quantization_preserves_order():
    values = [-3.0, -0.5, 0.0, 0.5, 3.0]
    qs, _, _ = quantize_int8(values)
    assert qs == sorted(qs)


def test_an_all_zero_tensor_does_not_divide_by_zero():
    qs, scale, zero_point = quantize_int8([0.0, 0.0, 0.0])
    assert scale > 0
    assert dequantize_int8(qs, scale, zero_point) == APPROX([0.0, 0.0, 0.0])


def test_quantize_rejects_an_empty_tensor():
    with pytest.raises(ValueError):
        quantize_int8([])


def test_int8_weighs_a_quarter_of_fp32():
    """Ровно та экономия, ради которой всё затевалось: 4x по памяти."""
    values = [float(i) for i in range(1000)]
    qs, _, _ = quantize_int8(values)
    fp32_bytes = len(values) * 4
    int8_bytes = len(qs) * 1
    assert fp32_bytes == 4 * int8_bytes
