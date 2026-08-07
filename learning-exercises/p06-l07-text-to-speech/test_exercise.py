"""Тесты к уроку «Синтез речи (TTS)». Правь exercise.py."""

import pytest

from exercise import (
    character_error_rate,
    clip_waveform,
    grapheme_to_phoneme,
    length_regulate,
    normalize_text,
    predict_durations,
    resample_linear,
    vocode,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in matrix for x in row]


# ---------------------------------------------------------- normalize_text
def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("Hello, WORLD!", {}) == ["hello", "world"]


def test_normalize_text_expands_abbreviations():
    got = normalize_text("Dr. Smith, 6 pm.", {"dr": "doctor", "pm": "p m"})
    assert got == ["doctor", "smith", "6", "p", "m"]


def test_normalize_text_drops_standalone_punctuation():
    """Токен из одной точки после чистки пуст — его не должно быть в списке."""
    assert normalize_text("a . b", {}) == ["a", "b"]


def test_normalize_text_keeps_apostrophe_inside_word():
    """Пунктуация снимается только с краёв: don't остаётся одним словом."""
    assert normalize_text("don't", {}) == ["don't"]


# --------------------------------------------------- grapheme_to_phoneme
def test_g2p_uses_the_lexicon():
    lex = {"cat": ["K", "AE", "T"]}
    assert grapheme_to_phoneme(["cat"], lex, lambda w: ["?"]) == ["K", "AE", "T"]


def test_g2p_falls_back_on_unknown_word():
    assert grapheme_to_phoneme(["ghu"], {}, lambda w: list(w.upper())) == ["G", "H", "U"]


def test_g2p_does_not_call_fallback_for_known_words():
    """Ловушка: словарное произношение всегда точнее предсказанного."""

    def fallback(word):
        raise AssertionError("fallback не должен вызываться для слова из lexicon")

    assert grapheme_to_phoneme(["cat"], {"cat": ["K"]}, fallback) == ["K"]


def test_g2p_result_is_flat_and_ordered():
    lex = {"a": ["AH"], "b": ["B", "IY"]}
    assert grapheme_to_phoneme(["b", "a"], lex, lambda w: []) == ["B", "IY", "AH"]


# ------------------------------------------------------ predict_durations
def test_predict_durations_worked_example():
    assert predict_durations(["K", "AE", "T"], {"AE": 116.0}) == [5, 11, 5]


def test_predict_durations_never_drops_a_phoneme():
    """Ловушка: floor(3/10) = 0, и фонема исчезла бы из речи."""
    assert predict_durations(["K"], {"K": 3.0}) == [1]


def test_predict_durations_uses_default_for_unknown_phoneme():
    assert predict_durations(["ZZ"], {}, default_ms=100.0, frame_ms=10.0) == [10]


def test_predict_durations_rounds_down_not_up():
    assert predict_durations(["K"], {"K": 99.0}, frame_ms=10.0) == [9]


# -------------------------------------------------------- length_regulate
def test_length_regulate_worked_example():
    got = length_regulate([[1.0], [2.0]], [2, 3])
    assert flat(got) == APPROX([1.0, 1.0, 2.0, 2.0, 2.0])


def test_length_regulate_output_length_is_sum_of_durations():
    """Главное свойство: сколько кадров назначили — столько и получили."""
    durations = [1, 4, 2, 7]
    vectors = [[float(i), 0.0] for i in range(4)]
    assert len(length_regulate(vectors, durations)) == sum(durations)


def test_length_regulate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        length_regulate([[1.0], [2.0]], [3])


def test_length_regulate_copies_each_frame():
    """Ловушка: если класть один и тот же список, правка кадра поменяет все."""
    frames = length_regulate([[1.0]], [3])
    frames[0][0] = 99.0
    assert frames[1][0] == APPROX(1.0)


# ------------------------------------------------------------------ vocode
def test_vocode_length_is_frames_times_hop():
    assert len(vocode([(0.5, 440.0), (0.5, 880.0)], 128, 16000)) == 256


def test_vocode_zero_frequency_is_silence():
    assert vocode([(1.0, 0.0)], 4, 8000) == APPROX([0.0, 0.0, 0.0, 0.0])


def test_vocode_respects_amplitude():
    wav = vocode([(0.25, 1000.0)], 200, 16000)
    assert max(abs(x) for x in wav) <= 0.25 + 1e-12


def test_vocode_keeps_phase_continuous_across_frames():
    """Ловушка: сброс фазы на границе кадра даёт слышимый щелчок.

    Два одинаковых кадра — это одна непрерывная синусоида. Скачок на стыке
    не может быть больше обычного шага между соседними сэмплами.
    """
    hop = 64
    wav = vocode([(1.0, 500.0), (1.0, 500.0)], hop, 16000)
    jump = abs(wav[hop] - wav[hop - 1])
    inside = max(abs(wav[i] - wav[i - 1]) for i in range(1, hop))
    assert jump <= inside + 1e-9


# ----------------------------------------------------------- clip_waveform
def test_clip_waveform_clamps_and_counts():
    wav, n = clip_waveform([0.5, 1.4, -2.0])
    assert wav == APPROX([0.5, 1.0, -1.0])
    assert n == 2


def test_clip_waveform_counts_zero_when_nothing_overshoots():
    assert clip_waveform([0.1, -0.1])[1] == 0


def test_clip_waveform_does_not_count_the_boundary_value():
    """Ровно 1.0 — это не перегрузка, резать нечего."""
    assert clip_waveform([1.0, -1.0])[1] == 0


def test_clip_waveform_honours_custom_bounds():
    wav, n = clip_waveform([0.5, -0.5], lo=0.0, hi=1.0)
    assert wav == APPROX([0.5, 0.0])
    assert n == 1


# --------------------------------------------------------- resample_linear
def test_resample_linear_downsamples():
    assert resample_linear([0.0, 1.0, 2.0, 3.0], 4, 2) == APPROX([0.0, 2.0])


def test_resample_linear_upsamples_with_interpolation():
    assert resample_linear([0.0, 1.0], 1, 2) == APPROX([0.0, 0.5, 1.0, 1.0])


def test_resample_linear_is_identity_at_equal_rates():
    wav = [0.0, 0.3, -0.7, 1.0]
    assert resample_linear(wav, 16000, 16000) == APPROX(wav)


def test_resample_linear_keeps_the_duration_in_seconds():
    """24 kHz → 16 kHz: сэмплов на треть меньше, но звучит столько же."""
    wav = [0.0] * 2400
    assert len(resample_linear(wav, 24000, 16000)) == 1600


def test_resample_linear_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        resample_linear([0.0, 1.0], 16000, 0)


# ---------------------------------------------------- character_error_rate
def test_cer_of_identical_strings_is_zero():
    assert character_error_rate("cat", "cat") == APPROX(0.0)


def test_cer_counts_one_substitution():
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)


def test_cer_of_empty_hypothesis_is_one():
    assert character_error_rate("cat", "") == APPROX(1.0)


def test_cer_of_two_empty_strings_is_zero():
    assert character_error_rate("", "") == APPROX(0.0)


def test_cer_can_exceed_one_when_tts_rambles():
    """Синтез наговорил лишнего — вставок больше, чем символов в эталоне."""
    assert character_error_rate("hi", "hi there friend") > 1.0


