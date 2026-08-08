"""Тесты к уроку «Audio-language модели: от Whisper до Audio Flamingo 3».

Правь exercise.py.
"""

import math

import pytest

from exercise import (
    ACOUSTIC_NEEDS,
    dft_magnitude,
    frame_signal,
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
    pick_pipeline,
    qformer_attend,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем матрицу в один."""
    return [v for row in M for v in row]


def sine(n, cycles, amp=1.0):
    return [amp * math.cos(2 * math.pi * cycles * i / n) for i in range(n)]


# ---------------------------------------------------------------- hz_to_mel
def test_hz_to_mel_of_zero_is_zero():
    assert hz_to_mel(0) == APPROX(0.0)


def test_hz_to_mel_matches_the_htk_formula():
    assert hz_to_mel(700) == pytest.approx(2595.0 * math.log10(2.0), abs=1e-9)


def test_hz_to_mel_compresses_high_frequencies():
    """Первая тысяча герц занимает в мелах больше, чем восьмая, — в этом вся шкала."""
    low = hz_to_mel(1000) - hz_to_mel(0)
    high = hz_to_mel(8000) - hz_to_mel(7000)
    assert low > 3 * high


# ---------------------------------------------------------------- mel_to_hz
def test_mel_to_hz_of_zero_is_zero():
    assert mel_to_hz(0) == APPROX(0.0)


def test_mel_to_hz_inverts_hz_to_mel():
    for f in (50.0, 440.0, 3000.0, 8000.0):
        assert mel_to_hz(hz_to_mel(f)) == pytest.approx(f, abs=1e-6)


def test_mel_to_hz_grows_faster_than_linearly():
    """Равные шаги в мелах дают всё более широкие шаги в герцах."""
    step_low = mel_to_hz(1000) - mel_to_hz(500)
    step_high = mel_to_hz(2500) - mel_to_hz(2000)
    assert step_high > step_low


# ----------------------------------------------------------- mel_filterbank
def test_mel_filterbank_shape():
    fbank = mel_filterbank(64, 20, 16000)
    assert len(fbank) == 20
    assert all(len(row) == 64 for row in fbank)


def test_mel_filterbank_weights_are_nonnegative_and_capped_at_one():
    fbank = mel_filterbank(64, 8, 16000)
    assert all(0.0 <= w <= 1.0 + 1e-12 for w in flat(fbank))


def test_mel_filterbank_centers_move_up_in_frequency():
    """Фильтр m должен смотреть выше, чем фильтр m-1."""
    fbank = mel_filterbank(64, 8, 16000)
    peaks = [row.index(max(row)) for row in fbank]
    assert peaks == sorted(peaks)
    assert len(set(peaks)) == len(peaks)


def test_mel_filterbank_widens_towards_high_frequencies():
    """Мел-варп: низкие фильтры узкие, высокие широкие. У линейного банка ширины равны."""
    fbank = mel_filterbank(64, 8, 16000)
    widths = [sum(1 for w in row if w > 0) for row in fbank]
    assert widths[-1] > 3 * widths[0]


# ------------------------------------------------------------- frame_signal
def test_frame_signal_counts_frames_of_one_second():
    assert len(frame_signal([0.0] * 16000, 16000)) == 98


def test_frame_signal_frames_have_window_length():
    frames = frame_signal([0.0] * 16000, 16000, win_ms=25, hop_ms=10)
    assert all(len(f) == 400 for f in frames)


def test_frame_signal_drops_the_tail_shorter_than_a_window():
    assert frame_signal([0.0] * 100, 16000) == []


def test_frame_signal_frames_overlap_by_the_hop():
    """Соседние кадры сдвинуты ровно на hop, а не на win."""
    x = list(range(1000))
    frames = frame_signal(x, 1000, win_ms=100, hop_ms=10)
    assert frames[0][0] == 0
    assert frames[1][0] == 10


def test_frame_signal_thirty_seconds_is_2998_not_3000():
    """Whisper печатает 3000 из-за padding до ровных 30 секунд, формула даёт 2998."""
    assert len(frame_signal([0.0] * 480000, 16000)) == 2998


# ------------------------------------------------------------ dft_magnitude
def test_dft_magnitude_dc_bin_is_the_sum():
    assert dft_magnitude([1.0, 2.0, 3.0, 4.0], 1)[0] == pytest.approx(10.0, abs=1e-9)


def test_dft_magnitude_of_a_constant_has_only_dc():
    mags = dft_magnitude([2.0] * 8, 5)
    assert mags[0] == pytest.approx(16.0, abs=1e-9)
    assert all(m == pytest.approx(0.0, abs=1e-9) for m in mags[1:])


def test_dft_magnitude_peaks_at_the_tone_bin():
    """Косинус с тремя периодами на кадр даёт всплеск ровно в третьем бине."""
    mags = dft_magnitude(sine(16, 3), 9)
    assert mags.index(max(mags)) == 3
    assert mags[3] == pytest.approx(8.0, abs=1e-9)


def test_dft_magnitude_ignores_a_time_shift():
    """Сдвиг во времени меняет фазу, но не модуль — спектрограмма на этом и стоит."""
    x = sine(16, 3)
    shifted = x[5:] + x[:5]
    assert dft_magnitude(shifted, 9) == pytest.approx(dft_magnitude(x, 9), abs=1e-9)


def test_dft_bins_span_to_nyquist_instead_of_taking_the_low_prefix():
    """With five outputs, a 3/8-rate tone belongs in bin 3, not outside the spectrum."""
    mags = dft_magnitude(sine(16, 6), 5)
    assert mags.index(max(mags)) == 3
    assert mags[3] == pytest.approx(8.0, abs=1e-9)


# ------------------------------------------------------ log_mel_spectrogram
def test_log_mel_spectrogram_shape():
    spec = log_mel_spectrogram([0.0] * 400, 8000, n_mels=4, n_bins=16)
    assert len(spec) == 3
    assert all(len(row) == 4 for row in spec)


def test_log_mel_spectrogram_of_silence_is_all_zeros():
    """log(1 + 0) = 0. Наивный log(0) дал бы -inf и утащил бы за собой всё."""
    spec = log_mel_spectrogram([0.0] * 400, 8000, n_mels=4, n_bins=16)
    assert flat(spec) == pytest.approx([0.0] * 12, abs=1e-12)


def test_log_mel_spectrogram_is_nonnegative():
    spec = log_mel_spectrogram(sine(400, 20, amp=0.5), 8000, n_mels=4, n_bins=16)
    assert all(v >= 0.0 for v in flat(spec))


def test_log_mel_spectrogram_compresses_dynamic_range():
    """Громче — больше, но НЕ вдвое: логарифм сжимает динамический диапазон."""
    quiet = log_mel_spectrogram(sine(400, 20, amp=1.0), 8000, n_mels=4, n_bins=16)
    loud = log_mel_spectrogram(sine(400, 20, amp=2.0), 8000, n_mels=4, n_bins=16)
    q, l = flat(quiet), flat(loud)
    assert max(l) > max(q)
    assert max(l) < 2 * max(q)


def test_high_frequency_tone_reaches_the_highest_mel_band():
    """The spectral grid must reach Nyquist or a 3.5 kHz tone is mapped as low audio."""
    sr = 8000
    low = [math.sin(2 * math.pi * 500 * i / sr) for i in range(200)]
    high = [math.sin(2 * math.pi * 3500 * i / sr) for i in range(200)]
    low_mel = log_mel_spectrogram(low, sr, n_mels=4, n_bins=16, hop_ms=25)[0]
    high_mel = log_mel_spectrogram(high, sr, n_mels=4, n_bins=16, hop_ms=25)[0]
    assert high_mel[-1] > low_mel[-1]


# ---------------------------------------------------------- qformer_attend
def test_qformer_attend_emits_one_token_per_query():
    frames = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 1.0]]
    out = qformer_attend([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], frames)
    assert len(out) == 2
    assert all(len(t) == 3 for t in out)


def test_qformer_attend_collapses_identical_frames_to_that_frame():
    """Веса софтмакса складываются в единицу, поэтому выход — сам кадр."""
    out = qformer_attend([[1.0, 0.0], [-5.0, 3.0]], [[3.0, 3.0]] * 4)
    assert flat(out) == pytest.approx([3.0, 3.0, 3.0, 3.0], abs=1e-9)


def test_qformer_attend_output_stays_inside_the_frames_hull():
    """Выход — выпуклая комбинация кадров, вылезти за их минимум/максимум он не может."""
    frames = [[1.0, -2.0], [4.0, 0.5], [-3.0, 7.0]]
    out = qformer_attend([[0.3, -1.1], [2.0, 2.0]], frames)
    for token in out:
        for k, v in enumerate(token):
            column = [f[k] for f in frames]
            assert min(column) - 1e-12 <= v <= max(column) + 1e-12


def test_qformer_attend_does_not_care_about_frame_order():
    """Аттеншен — сумма по кадрам, порядок кадров на результат не влияет."""
    frames = [[1.0, -2.0], [4.0, 0.5], [-3.0, 7.0]]
    q = [[0.7, 0.2]]
    assert flat(qformer_attend(q, frames)) == pytest.approx(
        flat(qformer_attend(q, list(reversed(frames)))), abs=1e-9
    )


def test_qformer_attend_survives_huge_scores_and_picks_one_frame():
    """Наивный math.exp(score) здесь падает с OverflowError."""
    out = qformer_attend([[1e4, 0.0]], [[10.0, 0.0], [0.0, 10.0]])
    assert out[0] == pytest.approx([10.0, 0.0], abs=1e-9)


# ------------------------------------------------------------ pick_pipeline
def test_pick_pipeline_text_only_task_is_cascaded():
    assert pick_pipeline(["transcription", "summarization"]) == "cascaded"


def test_pick_pipeline_no_requirements_is_cascaded():
    assert pick_pipeline([]) == "cascaded"


def test_pick_pipeline_one_acoustic_need_forces_end_to_end():
    """Одного акустического требования достаточно: транскрипт эмоцию не переносит."""
    for need in sorted(ACOUSTIC_NEEDS):
        assert pick_pipeline(["transcription", need]) == "end-to-end"


def test_pick_pipeline_rejects_an_unknown_requirement():
    """Опечатка не должна тихо превращаться в 'cascaded'."""
    with pytest.raises(ValueError):
        pick_pipeline(["emotoin"])
