"""Тесты к уроку «Нейронные аудиокодеки». Правь exercise.py."""

import math
import random

import pytest

from exercise import (
    codec_cost,
    nearest_code,
    quantize_layer,
    reconstruction_mse,
    rvq_decode,
    rvq_encode,
    split_semantic_acoustic,
    uniform_codebook,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


def signal(n=200, seed=0):
    rng = random.Random(seed)
    return [0.7 * math.sin(2 * math.pi * i / 40) + 0.2 * rng.gauss(0, 1.0) for i in range(n)]


def cascade(n_layers, size=9, scale=1.2):
    """Каскад codebook'ов: каждый следующий в 4 раза мельче предыдущего."""
    return [uniform_codebook(size, scale / (4 ** k)) for k in range(n_layers)]


# ----------------------------------------------------------- nearest_code
def test_nearest_code_returns_index_not_value():
    assert nearest_code([-1.0, 0.0, 1.0], 0.4) == 1


def test_nearest_code_picks_the_closer_side():
    assert nearest_code([-1.0, 0.0, 1.0], 0.6) == 2


def test_nearest_code_breaks_ties_toward_lower_index():
    """Ничья должна решаться детерминированно, иначе кодер не воспроизводим."""
    assert nearest_code([0.0, 1.0], 0.5) == 0


def test_nearest_code_rejects_empty_codebook():
    with pytest.raises(ValueError):
        nearest_code([], 0.0)


# ------------------------------------------------------- uniform_codebook
def test_uniform_codebook_spans_the_requested_range():
    assert uniform_codebook(3, 1.0) == APPROX([-1.0, 0.0, 1.0])


def test_uniform_codebook_is_symmetric_around_zero():
    cb = uniform_codebook(5, 2.0)
    assert [-c for c in reversed(cb)] == APPROX(cb)


def test_uniform_codebook_of_odd_size_contains_zero():
    """Ноль в codebook — гарантия, что остаток не вырастет после квантования."""
    assert min(abs(c) for c in uniform_codebook(9, 1.0)) == APPROX(0.0)


def test_uniform_codebook_of_size_one_is_just_zero():
    assert uniform_codebook(1, 7.0) == APPROX([0.0])


def test_uniform_codebook_rejects_zero_size():
    with pytest.raises(ValueError):
        uniform_codebook(0, 1.0)


# -------------------------------------------------------- quantize_layer
def test_quantize_layer_returns_indices_and_residuals():
    indices, residuals = quantize_layer([0.4, -0.9], [-1.0, 0.0, 1.0])
    assert indices == [1, 0]
    assert residuals == APPROX([0.4, 0.1])


def test_quantize_layer_residual_is_never_larger_than_the_input():
    """Если в codebook есть 0.0, слой может только уменьшить остаток."""
    values = signal(120, seed=1)
    _, residuals = quantize_layer(values, uniform_codebook(9, 1.2))
    assert all(abs(r) <= abs(v) + 1e-12 for v, r in zip(values, residuals))


def test_quantize_layer_keeps_the_length():
    indices, residuals = quantize_layer(signal(37), uniform_codebook(9, 1.2))
    assert len(indices) == len(residuals) == 37


# ------------------------------------------------------------ rvq_encode
def test_rvq_encode_returns_one_index_list_per_codebook():
    codes = rvq_encode(signal(50), cascade(4))
    assert len(codes) == 4
    assert all(len(layer) == 50 for layer in codes)


def test_rvq_encode_indices_stay_inside_the_codebooks():
    books = cascade(3, size=9)
    codes = rvq_encode(signal(50), books)
    assert all(0 <= i < len(cb) for layer, cb in zip(codes, books) for i in layer)


def test_rvq_encode_second_layer_sees_the_residual_not_the_signal():
    """0.4 -> код 0.0 на первом слое, остаток 0.4 -> код 0.5 на втором."""
    assert rvq_encode([0.4], [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]) == [[1], [2]]


# ------------------------------------------------------------ rvq_decode
def test_rvq_decode_sums_the_chosen_codes():
    books = [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]
    assert rvq_decode([[1], [2]], books, 1) == APPROX([0.5])


def test_rvq_decode_of_empty_cascade_is_silence():
    assert rvq_decode([], [], 3) == APPROX([0.0, 0.0, 0.0])


def test_rvq_roundtrip_approximates_the_signal():
    values = signal(150, seed=2)
    books = cascade(5)
    recon = rvq_decode(rvq_encode(values, books), books, len(values))
    assert reconstruction_mse(values, recon) < 1e-3


def test_rvq_error_shrinks_with_every_extra_codebook():
    """Главное свойство RVQ: каждый слой уменьшает ошибку, а не переписывает её."""
    values = signal(150, seed=3)
    errors = []
    for n in (1, 2, 3, 4, 5):
        books = cascade(n)
        recon = rvq_decode(rvq_encode(values, books), books, len(values))
        errors.append(reconstruction_mse(values, recon))
    assert all(a > b for a, b in zip(errors, errors[1:]))


def test_rvq_prefix_of_the_cascade_still_decodes():
    """Оборвали каскад на половине — сигнал грубее, но не сломан."""
    values = signal(80, seed=4)
    books = cascade(4)
    codes = rvq_encode(values, books)
    rough = rvq_decode(codes[:2], books[:2], len(values))
    full = rvq_decode(codes, books, len(values))
    assert reconstruction_mse(values, rough) > reconstruction_mse(values, full)


# --------------------------------------------------- reconstruction_mse
def test_reconstruction_mse_of_identical_signals_is_zero():
    assert reconstruction_mse([1.0, 2.0], [1.0, 2.0]) == APPROX(0.0)


def test_reconstruction_mse_counts_squared_error():
    assert reconstruction_mse([0.0, 0.0], [1.0, -1.0]) == APPROX(1.0)


def test_reconstruction_mse_rejects_length_mismatch():
    """Молчаливый zip спрятал бы потерянные декодером отсчёты."""
    with pytest.raises(ValueError):
        reconstruction_mse([1.0, 2.0], [1.0])


# ------------------------------------------------------------ codec_cost
def test_codec_cost_of_mimi_ten_seconds():
    got = codec_cost(10, 12.5, 8, 1024)
    assert got == pytest.approx({"frames": 125.0, "tokens": 1000.0, "bitrate_bps": 1000.0})


def test_codec_cost_of_encodec_at_six_kbps():
    assert codec_cost(1, 75.0, 8, 1024)["bitrate_bps"] == APPROX(6000.0)


def test_codec_cost_low_frame_rate_means_shorter_lm_sequence():
    """12.5 Hz против 75 Hz — вот почему Mimi удобна для авторегрессии."""
    assert codec_cost(10, 12.5, 8, 1024)["tokens"] < codec_cost(10, 75.0, 8, 1024)["tokens"]


def test_codec_cost_doubling_codebooks_doubles_bitrate():
    a = codec_cost(1, 12.5, 8, 1024)["bitrate_bps"]
    b = codec_cost(1, 12.5, 16, 1024)["bitrate_bps"]
    assert b == APPROX(2 * a)


def test_codec_cost_rejects_degenerate_codebook():
    with pytest.raises(ValueError):
        codec_cost(1, 12.5, 8, 1)


# -------------------------------------------------- split_semantic_acoustic
def test_split_semantic_acoustic_takes_codebook_zero_as_semantic():
    semantic, acoustic = split_semantic_acoustic([[5, 1, 2], [7, 3, 4]])
    assert semantic == [5, 7]
    assert flat(acoustic) == [1, 2, 3, 4]


def test_split_semantic_acoustic_keeps_all_remaining_codebooks():
    frames = [[i] + [i * 10 + k for k in range(7)] for i in range(4)]
    semantic, acoustic = split_semantic_acoustic(frames)
    assert len(semantic) == 4
    assert all(len(row) == 7 for row in acoustic)


def test_split_semantic_acoustic_is_lossless():
    frames = [[1, 2, 3], [4, 5, 6]]
    semantic, acoustic = split_semantic_acoustic(frames)
    assert [[s] + a for s, a in zip(semantic, acoustic)] == frames


def test_split_semantic_acoustic_does_not_alias_the_input():
    """Acoustic-часть должна быть копией: правка вернувшегося списка не
    имеет права портить исходные фреймы."""
    frames = [[1, 2, 3]]
    _, acoustic = split_semantic_acoustic(frames)
    acoustic[0][0] = 999
    assert frames == [[1, 2, 3]]


def test_split_semantic_acoustic_rejects_empty_frame():
    with pytest.raises(ValueError):
        split_semantic_acoustic([[1, 2], []])
