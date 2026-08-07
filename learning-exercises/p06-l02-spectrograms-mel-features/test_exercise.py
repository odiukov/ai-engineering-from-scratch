"""Тесты к уроку «Спектрограммы, мел-шкала и признаки аудио». Правь exercise.py."""

import math

import pytest

from exercise import (
    frame_signal,
    hann,
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
    rfft_magnitudes,
    stft_magnitude,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


def _sine(freq_hz, sr, n, amp=1.0):
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]


def _chirp(n, sr, f0, f1):
    """Частота плавно едет от f0 к f1 — классический тест спектрограммы."""
    out, phase = [], 0.0
    for i in range(n):
        f = f0 + (f1 - f0) * i / n
        phase += 2 * math.pi * f / sr
        out.append(math.sin(phase))
    return out


# --------------------------------------------------------------------- hann
def test_hann_has_the_requested_length():
    assert len(hann(400)) == 400


def test_hann_vanishes_at_both_edges():
    """Ловушка: с n в знаменателе вместо n-1 правый край не дотянет до нуля."""
    w = hann(64)
    assert w[0] == APPROX(0.0)
    assert w[-1] == APPROX(0.0)


def test_hann_peaks_at_one_in_the_middle():
    assert max(hann(65)) == pytest.approx(1.0, abs=1e-12)


def test_hann_is_symmetric():
    w = hann(33)
    assert w == pytest.approx(w[::-1], abs=1e-12)


# ------------------------------------------------------------- frame_signal
def test_frame_count_follows_the_formula():
    assert len(frame_signal(list(range(1000)), 64, 32)) == 30


def test_every_frame_has_the_requested_length():
    frames = frame_signal(list(range(1000)), 64, 32)
    assert all(len(f) == 64 for f in frames)


def test_consecutive_frames_are_shifted_by_hop():
    frames = frame_signal([1, 2, 3, 4, 5], 3, 1)
    assert frames == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def test_signal_shorter_than_one_frame_yields_nothing():
    """Хвост не добивается нулями: неполный кадр просто выбрасывается."""
    assert frame_signal([1, 2], 5, 1) == []


def test_hop_equal_to_frame_length_tiles_without_overlap():
    frames = frame_signal([1, 2, 3, 4], 2, 2)
    assert frames == [[1, 2], [3, 4]]


# ---------------------------------------------------------- rfft_magnitudes
def test_rfft_returns_half_the_bins_plus_one():
    """Вещественный сигнал: вторая половина спектра зеркальна, её не считают."""
    assert len(rfft_magnitudes([0.0] * 64)) == 33


def test_dc_bin_is_the_absolute_sum_of_the_frame():
    assert rfft_magnitudes([1.0, 2.0, 3.0, 4.0])[0] == APPROX(10.0)


def test_alternating_signal_lands_on_the_nyquist_bin():
    m = rfft_magnitudes([1.0, -1.0, 1.0, -1.0])
    assert m.index(max(m)) == 2


def test_sine_peaks_at_its_own_bin():
    """Тон на частоте k*sr/N даёт максимум ровно в бине k."""
    sr, n = 16000, 64
    m = rfft_magnitudes(_sine(2000, sr, n))  # 2000 / (16000/64) = бин 8
    assert m.index(max(m)) == 8


# ----------------------------------------------------------- stft_magnitude
def test_stft_shape_is_frames_by_bins():
    spec = stft_magnitude([0.0] * 1000, 64, 32)
    assert len(spec) == 30
    assert all(len(row) == 33 for row in spec)


def test_stft_applies_the_window_to_every_frame():
    """Постоянный сигнал: без окна DC-бин был бы 64, с окном Ханна — 31.5."""
    spec = stft_magnitude([1.0] * 64, 64, 64)
    assert spec[0][0] == pytest.approx(31.5, abs=1e-9)


def test_stationary_tone_has_the_same_peak_in_every_frame():
    spec = stft_magnitude(_sine(2000, 16000, 512), 64, 64)
    peaks = [row.index(max(row)) for row in spec]
    assert len(set(peaks)) == 1


def test_chirp_peak_moves_up_the_frequency_axis():
    """Смысл спектрограммы: она видит, как частота меняется во времени."""
    spec = stft_magnitude(_chirp(512, 16000, 500, 7000), 64, 64)
    peaks = [row.index(max(row)) for row in spec]
    assert peaks[0] < peaks[-1]


# ------------------------------------------------------------ мел-шкала
def test_zero_hertz_is_zero_mel():
    assert hz_to_mel(0) == APPROX(0.0)


def test_mel_conversions_are_inverse_to_each_other():
    for f in (0.0, 100.0, 700.0, 4000.0, 8000.0):
        assert mel_to_hz(hz_to_mel(f)) == pytest.approx(f, abs=1e-6)


def test_mel_is_strictly_increasing():
    mels = [hz_to_mel(f) for f in (0, 100, 500, 1000, 4000, 8000)]
    assert all(a < b for a, b in zip(mels, mels[1:]))


def test_mel_is_nearly_linear_low_and_compressive_high():
    """Удвоение частоты внизу почти удваивает мелы, наверху — уже нет.

    Это и есть логарифмическое восприятие высоты, ради которого всё затевалось.
    """
    low_ratio = hz_to_mel(200) / hz_to_mel(100)
    high_ratio = hz_to_mel(4000) / hz_to_mel(2000)
    assert 1.8 < low_ratio < 2.0
    assert high_ratio < low_ratio


def test_equal_mel_steps_cover_more_hertz_higher_up():
    low_span = mel_to_hz(1000) - mel_to_hz(900)
    high_span = mel_to_hz(2000) - mel_to_hz(1900)
    assert high_span > low_span


# ----------------------------------------------------------- mel_filterbank
def test_filterbank_shape_is_mels_by_bins():
    fb = mel_filterbank(4, 64, 16000)
    assert len(fb) == 4
    assert all(len(row) == 33 for row in fb)


def test_filterbank_is_non_negative():
    fb = mel_filterbank(8, 64, 16000)
    assert min(flat(fb)) >= 0.0


def test_every_filter_is_a_triangle_peaking_at_one():
    for row in mel_filterbank(4, 64, 16000):
        peak = row.index(max(row))
        assert row[peak] == pytest.approx(1.0, abs=1e-9)
        assert all(a <= b + 1e-12 for a, b in zip(row[:peak], row[1 : peak + 1]))
        assert all(a >= b - 1e-12 for a, b in zip(row[peak:], row[peak + 1 :]))


def test_filters_are_ordered_by_frequency():
    peaks = [row.index(max(row)) for row in mel_filterbank(4, 64, 16000)]
    assert all(a < b for a, b in zip(peaks, peaks[1:]))


def test_filters_get_wider_towards_high_frequencies():
    """Точки расставлены по мелам, поэтому в герцах верхние фильтры шире.

    Если фильтры вышли одинаковой ширины — значит, шаг сделан по герцам,
    и мел-шкалы в этой матрице нет.
    """
    widths = [sum(1 for v in row if v > 0) for row in mel_filterbank(4, 64, 16000)]
    assert all(a < b for a, b in zip(widths, widths[1:]))


# ------------------------------------------------------ log_mel_spectrogram
def test_log_mel_shape_is_frames_by_mels():
    fb = mel_filterbank(4, 64, 16000)
    spec = stft_magnitude(_sine(2000, 16000, 512), 64, 32)
    lm = log_mel_spectrogram(spec, fb)
    assert len(lm) == len(spec)
    assert all(len(row) == 4 for row in lm)


def test_silence_maps_to_the_log_of_eps():
    """Ловушка: без пола eps здесь был бы log(0) = -inf и NaN в градиентах."""
    fb = mel_filterbank(2, 8, 16000)
    lm = log_mel_spectrogram([[0.0] * 5], fb, eps=1e-10)
    assert flat(lm) == pytest.approx([math.log(1e-10)] * 2)


def test_doubling_the_amplitude_adds_log_two():
    """Фильтрбанк линеен, логарифм превращает множитель в слагаемое."""
    fb = mel_filterbank(4, 64, 16000)
    spec = stft_magnitude(_sine(2000, 16000, 256), 64, 64)
    louder = [[2 * v for v in row] for row in spec]
    a = flat(log_mel_spectrogram(spec, fb))
    b = flat(log_mel_spectrogram(louder, fb))
    assert [y - x for x, y in zip(a, b)] == pytest.approx(
        [math.log(2)] * len(a), abs=1e-9
    )


def test_projection_is_a_weighted_sum_of_bins():
    fb = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]
    lm = log_mel_spectrogram([[math.e, 2.0, 2.0]], fb)
    assert lm[0] == pytest.approx([1.0, math.log(2.0)], abs=1e-9)
