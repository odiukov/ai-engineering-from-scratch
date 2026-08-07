"""Тесты к уроку «Основы аудио: волна, дискретизация, преобразование Фурье».

Правь exercise.py."""

import math

import pytest

from exercise import (
    alias_frequency,
    bin_to_hz,
    dft_magnitudes,
    dominant_frequency,
    float_to_pcm16,
    nyquist,
    pcm16_to_float,
    sine,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# --------------------------------------------------------------------- sine
def test_sine_length_is_sample_rate_times_seconds():
    assert len(sine(440, 16000, 1.0)) == 16000
    assert len(sine(440, 16000, 0.25)) == 4000


def test_sine_starts_at_zero():
    assert sine(440, 16000, 0.01)[0] == APPROX(0.0)


def test_sine_never_exceeds_its_amplitude():
    x = sine(1000, 16000, 0.05, amp=0.3)
    assert max(x) <= 0.3 + 1e-12
    assert min(x) >= -0.3 - 1e-12


def test_sine_repeats_after_one_period():
    """Частота 100 Гц при sr=16000 — период ровно 160 отсчётов."""
    x = sine(100, 16000, 0.05)
    assert x[7] == pytest.approx(x[7 + 160], abs=1e-9)


def test_sine_frequency_is_per_second_not_per_clip():
    """Ловушка: делить на sr, а не на len(x). Иначе клип в полсекунды
    даст вдвое меньше периодов, чем просили."""
    half = sine(4, 100, 0.5)
    # 4 Гц за полсекунды — два полных периода на 50 отсчётах
    assert half[0] == APPROX(0.0)
    assert half[25] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ nyquist
def test_nyquist_is_half_the_sample_rate():
    assert nyquist(16000) == APPROX(8000.0)


def test_telephony_nyquist_is_only_four_kilohertz():
    """8 кГц режет спектр на 4 кГц — там и теряются согласные."""
    assert nyquist(8000) == APPROX(4000.0)


# ---------------------------------------------------------- alias_frequency
def test_frequency_below_nyquist_is_unchanged():
    assert alias_frequency(3000, 16000) == APPROX(3000.0)


def test_frequency_above_nyquist_folds_down():
    """Классика: 7 кГц при sr=10 кГц превращается в 3 кГц."""
    assert alias_frequency(7000, 10000) == APPROX(3000.0)


def test_frequency_equal_to_sample_rate_looks_like_dc():
    assert alias_frequency(10000, 10000) == APPROX(0.0)


def test_alias_never_exceeds_nyquist():
    """Что бы ни подали на вход, наблюдаемая частота лежит в [0, sr/2]."""
    for f in (100, 4999, 5001, 12345, 48000):
        assert 0.0 <= alias_frequency(f, 10000) <= 5000.0


def test_aliasing_shows_up_as_a_real_dft_peak():
    """Не арифметика, а физика: пик спектра стоит там, куда свернулась частота."""
    sr = 10000
    x = sine(7000, sr, 0.01)  # 100 отсчётов, шаг бина 100 Гц
    assert dominant_frequency(x, sr) == APPROX(alias_frequency(7000, sr))


# ---------------------------------------------------------------- bin_to_hz
def test_bin_zero_is_dc():
    assert bin_to_hz(0, 16000, 1024) == APPROX(0.0)


def test_bin_step_is_sample_rate_over_n_fft():
    assert bin_to_hz(1, 16000, 1024) == APPROX(15.625)


def test_middle_bin_sits_at_nyquist():
    assert bin_to_hz(512, 16000, 1024) == APPROX(nyquist(16000))


def test_longer_window_gives_finer_resolution():
    """Удвоил окно — вдвое мельче шаг по частоте. Это и есть компромисс STFT."""
    assert bin_to_hz(1, 16000, 800) > bin_to_hz(1, 16000, 1600)


# ----------------------------------------------------------- dft_magnitudes
def test_dft_of_constant_puts_all_energy_in_bin_zero():
    m = dft_magnitudes([1.0, 1.0, 1.0, 1.0])
    assert m == pytest.approx([4.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_dft_of_alternating_signal_peaks_at_nyquist_bin():
    m = dft_magnitudes([1.0, -1.0, 1.0, -1.0])
    assert m.index(max(m)) == 2


def test_dft_of_a_sine_peaks_exactly_at_its_own_bin():
    """Главное свойство DFT: тон на частоте k*sr/N даёт пик ровно в бине k."""
    sr, n = 16000, 160
    x = sine(1500, sr, n / sr)  # 1500 Гц = бин 15 при шаге 100 Гц
    m = dft_magnitudes(x)
    assert m.index(max(m[: n // 2 + 1])) == 15


def test_dft_magnitudes_are_mirror_symmetric_for_real_input():
    """|X[k]| == |X[N-k]|: вторая половина спектра не несёт новой информации."""
    x = sine(1300, 16000, 0.005)
    m = dft_magnitudes(x)
    n = len(x)
    for k in range(1, n // 2):
        assert m[k] == pytest.approx(m[n - k], abs=1e-6)


def test_parseval_conserves_energy():
    """Теорема Парсеваля: сумма |X[k]|^2 равна N * сумме x[n]^2.

    Преобразование Фурье — это поворот системы координат, энергия сигнала
    от поворота не меняется."""
    x = [0.3, -0.1, 0.7, 0.2, -0.5, 0.9, 0.0, -0.4]
    m = dft_magnitudes(x)
    time_energy = sum(v * v for v in x)
    freq_energy = sum(v * v for v in m)
    assert freq_energy == pytest.approx(len(x) * time_energy, rel=1e-9)


# ------------------------------------------------------ dominant_frequency
def test_dominant_frequency_of_a_pure_tone():
    sr = 16000
    assert dominant_frequency(sine(3000, sr, 0.01), sr) == APPROX(3000.0)


def test_dominant_frequency_ignores_the_mirror_half():
    """Ловушка: argmax по всему спектру вернул бы sr - f вместо f."""
    sr = 16000
    assert dominant_frequency(sine(6000, sr, 0.01), sr) == APPROX(6000.0)


def test_dominant_frequency_picks_the_loudest_component_of_a_mix():
    sr, secs = 16000, 0.01
    quiet = sine(1000, sr, secs, amp=0.1)
    loud = sine(4000, sr, secs, amp=0.9)
    mix = [a + b for a, b in zip(quiet, loud)]
    assert dominant_frequency(mix, sr) == APPROX(4000.0)


# ------------------------------------------------------------- PCM/float
def test_float_to_pcm16_maps_full_scale():
    assert float_to_pcm16([0.0, 1.0, -1.0]) == [0, 32767, -32767]


def test_float_to_pcm16_clips_instead_of_overflowing():
    """1.5 * 32767 не влезает в int16 — обрезаем до максимума."""
    assert float_to_pcm16([1.5, -2.0]) == [32767, -32767]


def test_pcm16_to_float_returns_the_unit_range():
    assert pcm16_to_float([0, 32767, -32767]) == pytest.approx([0.0, 1.0, -1.0])


def test_pcm16_round_trip_is_accurate_to_one_step():
    """16 бит — это 1/32767 шага. Сигнал должен пережить запись и чтение."""
    x = sine(440, 16000, 0.01)
    back = pcm16_to_float(float_to_pcm16(x))
    assert back == pytest.approx(x, abs=1.0 / 32767)
