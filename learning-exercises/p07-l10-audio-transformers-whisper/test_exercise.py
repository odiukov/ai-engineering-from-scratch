"""Тесты к уроку «Whisper: аудио как последовательность кадров». Правь exercise.py."""

import math

import pytest

from exercise import (
    FRAME_SIZE,
    conv_stem_length,
    frame_energy,
    frame_signal,
    n_frames,
    pad_or_clip,
    parse_whisper_prompt,
    sine_wave,
    whisper_prompt,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in M for x in row]


# -------------------------------------------------------------- sine_wave
def test_one_second_at_16_khz_is_16000_samples():
    assert len(sine_wave(440, 1.0)) == 16000


def test_sine_wave_starts_at_zero():
    assert sine_wave(440, 0.01)[0] == APPROX(0.0)


def test_sine_wave_stays_within_the_unit_range():
    assert all(-1.0 <= v <= 1.0 for v in sine_wave(1000, 0.05))


def test_sine_wave_repeats_after_one_period():
    """400 Гц при 16 кГц — это ровно 40 отсчётов на период."""
    x = sine_wave(400, 0.02)
    assert all(x[i] == pytest.approx(x[i + 40], abs=1e-12) for i in range(40))


# --------------------------------------------------------------- n_frames
def test_one_second_gives_about_a_hundred_frames():
    """98, а не 100: последние окна вылезли бы за конец сигнала."""
    assert n_frames(16000) == 98


def test_thirty_seconds_gives_almost_three_thousand_frames():
    """2998 против 3000 у Whisper — остаток добирается паддингом."""
    assert n_frames(30 * 16000) == 2998


def test_a_signal_shorter_than_one_window_gives_no_frames():
    assert n_frames(399) == 0


def test_a_signal_of_exactly_one_window_gives_one_frame():
    assert n_frames(400) == 1


def test_every_extra_hop_of_samples_adds_one_frame():
    assert n_frames(16000 + 160) == n_frames(16000) + 1


# ----------------------------------------------------------- frame_signal
def test_frame_signal_cuts_the_expected_number_of_frames():
    assert len(frame_signal(sine_wave(440, 1.0))) == n_frames(16000)


def test_every_frame_is_full_length():
    frames = frame_signal(sine_wave(440, 0.05))
    assert all(len(f) == 400 for f in frames)


def test_neighbouring_frames_overlap_by_frame_size_minus_hop():
    """При настройках Whisper 240 из 400 отсчётов повторяются в следующем окне."""
    frames = frame_signal(sine_wave(440, 0.05))
    assert frames[0][160:] == APPROX(frames[1][:240])


def test_frame_signal_matches_the_hand_written_example():
    assert frame_signal([1, 2, 3, 4, 5], frame_size=3, hop=1) == [
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    ]


def test_frame_signal_drops_the_incomplete_tail():
    """Из шести отсчётов при окне 4 и шаге 4 выходит одно окно, не два."""
    assert frame_signal([1, 2, 3, 4, 5, 6], frame_size=4, hop=4) == [[1, 2, 3, 4]]


# ----------------------------------------------------------- frame_energy
def test_silence_does_not_blow_up_into_minus_infinity():
    """log(0) это -inf; добавка 1e-9 держит тишину в районе -20.7."""
    assert frame_energy([0.0, 0.0]) == pytest.approx(math.log(1e-9), abs=1e-6)


def test_frame_energy_is_the_log_of_the_sum_of_squares():
    assert frame_energy([1.0, 1.0]) == pytest.approx(math.log(2.0), abs=1e-6)


def test_doubling_the_amplitude_adds_log_four():
    """Энергия квадратична по амплитуде, а логарифм превращает это в сдвиг."""
    quiet = frame_energy([0.5, -0.5, 0.25])
    loud = frame_energy([1.0, -1.0, 0.5])
    assert loud - quiet == pytest.approx(math.log(4.0), abs=1e-6)


# ------------------------------------------------------------ pad_or_clip
def test_pad_or_clip_reaches_the_target_exactly():
    assert len(pad_or_clip([[1.0], [2.0]], 3000)) == 3000


def test_padding_is_silence():
    padded = pad_or_clip([[1.0, 2.0]], 3)
    assert flat(padded[1:]) == APPROX([0.0, 0.0, 0.0, 0.0])


def test_padding_keeps_the_frame_width():
    padded = pad_or_clip([[1.0, 2.0, 3.0]], 5)
    assert all(len(f) == 3 for f in padded)


def test_empty_audio_is_padded_with_full_width_whisper_frames():
    """Без исходного кадра ширина всё равно известна из настройки Whisper."""
    padded = pad_or_clip([], 2)
    assert len(padded) == 2
    assert all(len(frame) == FRAME_SIZE for frame in padded)
    assert flat(padded) == APPROX([0.0] * (2 * FRAME_SIZE))


def test_a_long_recording_is_clipped_to_the_prefix():
    """Whisper видит только первые 30 секунд, остальное — забота чанкинга."""
    frames = [[float(i)] for i in range(10)]
    assert flat(pad_or_clip(frames, 3)) == APPROX([0.0, 1.0, 2.0])


def test_pad_or_clip_does_not_mutate_the_input():
    frames = [[1.0], [2.0]]
    pad_or_clip(frames, 5)
    assert frames == [[1.0], [2.0]]


# ------------------------------------------------------- conv_stem_length
def test_a_stride_one_conv_keeps_the_length():
    assert conv_stem_length(3000, stride=1) == 3000


def test_a_stride_two_conv_halves_the_length():
    assert conv_stem_length(3000, stride=2) == 1500


def test_the_whisper_stem_turns_3000_frames_into_1500_tokens():
    """Первый слой со шагом 1, второй со шагом 2. Два слоя со шагом 2 дали
    бы 750 — это уже не Whisper."""
    after_first = conv_stem_length(3000, stride=1)
    assert conv_stem_length(after_first, stride=2) == 1500


def test_halving_the_length_cuts_attention_cost_fourfold():
    """Ради этого стем и существует: attention квадратичен по длине."""
    short = conv_stem_length(3000, stride=2)
    assert (3000 / short) ** 2 == pytest.approx(4.0, abs=0.01)


def test_conv_stem_length_rejects_a_nonpositive_stride():
    with pytest.raises(ValueError):
        conv_stem_length(3000, stride=0)


# --------------------------------------------------------- whisper_prompt
def test_default_prompt_is_english_transcription_with_timestamps():
    assert whisper_prompt() == [
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
    ]


def test_switching_to_translation_changes_a_single_token():
    transcribe = whisper_prompt("fr", "transcribe")
    translate = whisper_prompt("fr", "translate")
    assert sum(1 for a, b in zip(transcribe, translate) if a != b) == 1


def test_the_notimestamps_token_appears_only_when_timestamps_are_off():
    """Логика обратная имени флага: токен ДОБАВЛЯЕТСЯ, когда таймкоды не нужны."""
    assert "<|notimestamps|>" in whisper_prompt(timestamps=False)
    assert "<|notimestamps|>" not in whisper_prompt(timestamps=True)


def test_an_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        whisper_prompt("en", "summarize")


# --------------------------------------------------- parse_whisper_prompt
def test_parse_reads_language_and_task():
    parsed = parse_whisper_prompt(["<|startoftranscript|>", "<|de|>", "<|translate|>"])
    assert parsed == {"lang": "de", "task": "translate", "timestamps": True}


def test_prompt_and_parse_are_inverse_for_every_combination():
    for lang in ("en", "fr", "ja"):
        for task in ("transcribe", "translate"):
            for timestamps in (True, False):
                parsed = parse_whisper_prompt(whisper_prompt(lang, task, timestamps))
                assert parsed == {"lang": lang, "task": task, "timestamps": timestamps}


def test_parse_rejects_a_prompt_without_the_start_token():
    with pytest.raises(ValueError):
        parse_whisper_prompt(["<|en|>", "<|transcribe|>", "<|0.00|>"])


def test_parse_rejects_a_prompt_that_is_too_short():
    with pytest.raises(ValueError):
        parse_whisper_prompt(["<|startoftranscript|>", "<|en|>"])
