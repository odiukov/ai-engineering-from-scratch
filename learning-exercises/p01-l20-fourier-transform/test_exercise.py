"""Тесты к уроку «Преобразование Фурье». Правь exercise.py."""

import math

import pytest

from exercise import (
    c_add,
    c_mul,
    circular_convolution,
    dft,
    fft,
    idft,
    magnitude_spectrum,
    twiddle,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем пары в плоский."""
    return [v for pair in M for v in pair]


def tone(N, k, amplitude=1.0):
    """Чистая синусоида: ровно k периодов на N отсчётов."""
    return [amplitude * math.sin(2 * math.pi * k * n / N) for n in range(N)]


# -------------------------------------------------------------------- c_add
def test_c_add_adds_parts_separately():
    assert c_add((1.0, 2.0), (3.0, -5.0)) == APPROX((4.0, -3.0))


def test_c_add_with_zero_returns_the_same_number():
    assert c_add((7.0, 1.0), (0.0, 0.0)) == APPROX((7.0, 1.0))


def test_c_add_is_commutative():
    assert c_add((2.0, -3.0), (5.0, 8.0)) == APPROX(c_add((5.0, 8.0), (2.0, -3.0)))


# -------------------------------------------------------------------- c_mul
def test_c_mul_expands_the_brackets():
    assert c_mul((1.0, 2.0), (3.0, 4.0)) == APPROX((-5.0, 10.0))


def test_c_mul_i_times_i_is_minus_one():
    """Ловушка: i*i = -1, поэтому в вещественной части минус, а не плюс."""
    assert c_mul((0.0, 1.0), (0.0, 1.0)) == APPROX((-1.0, 0.0))


def test_c_mul_by_one_is_identity():
    assert c_mul((5.0, 3.0), (1.0, 0.0)) == APPROX((5.0, 3.0))


def test_c_mul_multiplies_magnitudes():
    """Смысловое свойство: |a*b| = |a| * |b|, умножение растягивает длину."""
    a, b = (3.0, 4.0), (0.0, 2.0)
    p = c_mul(a, b)
    assert math.hypot(*p) == pytest.approx(math.hypot(*a) * math.hypot(*b), abs=1e-9)


# ------------------------------------------------------------------ twiddle
def test_twiddle_zero_is_one():
    assert twiddle(0, 4) == APPROX((1.0, 0.0))


def test_twiddle_quarter_turn_is_minus_i():
    """Прямое преобразование крутит ПО часовой: мнимая часть отрицательна."""
    assert twiddle(1, 4) == APPROX((0.0, -1.0))


def test_twiddle_sign_flips_only_the_imaginary_part():
    """Ловушка знака: sign=+1 — это то же самое зеркально, для обратного DFT."""
    forward = twiddle(3, 8)
    backward = twiddle(3, 8, sign=1)
    assert backward == APPROX((forward[0], -forward[1]))


def test_twiddle_stays_on_the_unit_circle():
    for k in range(7):
        re, im = twiddle(k, 7)
        assert re * re + im * im == pytest.approx(1.0, abs=1e-9)


def test_twiddle_wraps_after_a_full_period():
    """k и k+N дают один и тот же поворот — это и есть периодичность спектра."""
    assert twiddle(11, 8) == APPROX(twiddle(3, 8))


# ---------------------------------------------------------------------- dft
def test_dft_of_constant_puts_all_energy_in_bin_zero():
    """Постоянный сигнал — нулевая частота: вся энергия в DC-бине."""
    assert flat(dft([1.0, 1.0, 1.0, 1.0])) == APPROX([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_dft_of_impulse_is_flat():
    """Импульс содержит все частоты одинаково."""
    assert flat(dft([1.0, 0.0, 0.0, 0.0])) == APPROX([1.0, 0.0] * 4)


def test_dft_of_pure_tone_peaks_at_k_and_N_minus_k():
    """Синусоида частоты k даёт два пика: в бине k и в зеркальном N-k."""
    N, k = 16, 3
    mags = magnitude_spectrum(dft(tone(N, k)))
    assert mags[k] == pytest.approx(N / 2, abs=1e-9)
    assert mags[N - k] == pytest.approx(N / 2, abs=1e-9)
    others = [m for i, m in enumerate(mags) if i not in (k, N - k)]
    assert max(others) < 1e-9


def test_dft_of_real_signal_is_conjugate_symmetric():
    """X[k] сопряжён с X[N-k] — половина спектра лишняя."""
    x = [1.0, 4.0, -2.0, 0.5, 3.0, -1.5]
    X = dft(x)
    N = len(x)
    for k in range(1, N):
        assert X[k] == APPROX((X[N - k][0], -X[N - k][1]))


def test_dft_is_linear():
    """DFT(2x + 3y) = 2*DFT(x) + 3*DFT(y)."""
    x = [1.0, 2.0, 0.0, -1.0]
    y = [0.5, -3.0, 2.0, 1.0]
    mixed = dft([2 * a + 3 * b for a, b in zip(x, y)])
    parts = [
        c_add((2 * xa, 2 * xb), (3 * ya, 3 * yb))
        for (xa, xb), (ya, yb) in zip(dft(x), dft(y))
    ]
    assert flat(mixed) == APPROX(flat(parts))


# --------------------------------------------------------------------- idft
def test_idft_inverts_dft():
    """Преобразование обратимо без потерь: это просто смена базиса."""
    x = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0]
    back = [re for re, _ in idft(dft(x))]
    assert back == APPROX(x)


def test_idft_normalizes_by_N():
    """Ловушка 1/N: без деления сигнал вернётся в N раз громче."""
    assert flat(idft([(4.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])) == APPROX(
        [1.0, 0.0] * 4
    )


def test_idft_of_flat_spectrum_is_an_impulse():
    assert flat(idft([(1.0, 0.0)] * 4)) == APPROX([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_idft_uses_the_opposite_sign_and_does_not_reverse_time():
    """С неверным знаком сигнал восстановится зеркально по времени."""
    x = [0.0, 1.0, 2.0, 3.0]
    back = [re for re, _ in idft(dft(x))]
    assert back[1] == pytest.approx(1.0, abs=1e-9)
    assert back[3] == pytest.approx(3.0, abs=1e-9)


# -------------------------------------------------------- magnitude_spectrum
def test_magnitude_spectrum_is_the_hypotenuse():
    assert magnitude_spectrum([(3.0, 4.0), (0.0, 0.0)]) == APPROX([5.0, 0.0])


def test_magnitude_spectrum_ignores_the_sign_of_the_imaginary_part():
    """Модуль не различает фазу: сопряжённые числа дают одну амплитуду."""
    assert magnitude_spectrum([(0.0, -2.0)]) == APPROX(magnitude_spectrum([(0.0, 2.0)]))


def test_magnitude_spectrum_of_a_two_tone_signal_has_two_pairs_of_peaks():
    N = 32
    x = [a + b for a, b in zip(tone(N, 2), tone(N, 5, amplitude=0.5))]
    mags = magnitude_spectrum(dft(x))
    assert mags[2] == pytest.approx(N / 2, abs=1e-9)
    assert mags[5] == pytest.approx(N / 4, abs=1e-9)
    assert mags[N - 2] == pytest.approx(N / 2, abs=1e-9)


def test_parseval_energy_is_conserved():
    """Сумма квадратов во времени = (1/N) * сумма квадратов в частоте."""
    x = [1.0, -2.0, 3.5, 0.0, 4.0, -1.0, 2.0, 0.5]
    time_energy = sum(v * v for v in x)
    freq_energy = sum(m * m for m in magnitude_spectrum(dft(x))) / len(x)
    assert freq_energy == pytest.approx(time_energy, abs=1e-9)


# ---------------------------------------------------------------------- fft
def test_fft_matches_dft_on_a_power_of_two():
    x = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, 2.0, 6.0]
    assert flat(fft(x)) == APPROX(flat(dft(x)))


def test_fft_of_constant_puts_all_energy_in_bin_zero():
    assert flat(fft([1.0] * 8)) == APPROX([8.0, 0.0] + [0.0] * 14)


def test_fft_of_a_single_sample_is_that_sample():
    assert flat(fft([2.5])) == APPROX([2.5, 0.0])


def test_fft_still_matches_dft_on_a_non_power_of_two_length():
    """Ловушка длины: разбиение пополам не проходит — но ответ обязан быть верным."""
    for x in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0] * 7):
        assert flat(fft(x)) == APPROX(flat(dft(x)))


def test_fft_round_trip_through_idft():
    x = [0.5, 1.5, -2.0, 3.0, 0.0, -1.0, 2.5, 4.0]
    assert [re for re, _ in idft(fft(x))] == APPROX(x)


# ------------------------------------------------------ circular_convolution
def test_convolution_with_delta_is_identity():
    assert circular_convolution([1, 2, 3, 4], [1, 0, 0, 0]) == APPROX([1.0, 2.0, 3.0, 4.0])


def test_convolution_with_shifted_delta_rotates_the_signal():
    """Свёртка ЦИКЛИЧЕСКАЯ: хвост заворачивается в начало."""
    assert circular_convolution([1, 2, 3, 4], [0, 1, 0, 0]) == APPROX([4.0, 1.0, 2.0, 3.0])


def test_convolution_theorem_matches_a_direct_loop():
    """Главная проверка: частотный путь даёт то же, что честный двойной цикл."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    h = [0.5, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    N = len(x)
    direct = [sum(x[m] * h[(n - m) % N] for m in range(N)) for n in range(N)]
    assert circular_convolution(x, h) == APPROX(direct)


def test_convolution_is_commutative():
    x = [1.0, 2.0, 3.0, 4.0]
    h = [1.0, 1.0, 0.0, 0.0]
    assert circular_convolution(x, h) == APPROX(circular_convolution(h, x))


def test_convolution_pads_the_shorter_input_with_zeros():
    assert circular_convolution([1, 2, 3, 4], [1, 1]) == APPROX([5.0, 3.0, 5.0, 7.0])
