"""Тесты к уроку «Whisper: архитектура и дообучение». Правь exercise.py."""

import pytest

from exercise import (
    build_prompt,
    chunk_schedule,
    frame_budget,
    lora_parameter_count,
    merge_chunk_transcripts,
    normalize_log_mel,
    pad_or_trim,
    parse_prompt,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(M):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [v for row in M for v in row]


# --------------------------------------------------------------- pad_or_trim
def test_output_always_has_the_target_length():
    for n in (0, 3, 5, 9):
        assert len(pad_or_trim([1.0] * n, 5)) == 5


def test_short_clip_is_padded_with_zeros_at_the_end():
    assert pad_or_trim([1.0, 2.0, 3.0], 5) == APPROX([1.0, 2.0, 3.0, 0.0, 0.0])


def test_long_clip_is_cut_not_resampled():
    """Ловушка: сжать час в 30 секунд — значит поднять высоту голоса.

    Первые отсчёты обязаны остаться ровно теми же.
    """
    signal = [float(i) for i in range(100)]
    assert pad_or_trim(signal, 4) == APPROX([0.0, 1.0, 2.0, 3.0])


def test_pad_or_trim_does_not_mutate_the_input():
    signal = [1.0, 2.0]
    pad_or_trim(signal, 6)
    assert signal == [1.0, 2.0]


# -------------------------------------------------------------- frame_budget
def test_thirty_seconds_is_three_thousand_mel_frames():
    assert frame_budget(30.0) == (3000, 1500)


def test_conv_stride_halves_the_encoder_length():
    n_mel, n_enc = frame_budget(30.0, conv_stride=2)
    assert n_enc == n_mel // 2


def test_frame_count_is_proportional_to_duration():
    assert frame_budget(1.0)[0] * 30 == frame_budget(30.0)[0]


def test_smaller_hop_gives_more_frames():
    """Ловушка: кадры считаются по шагу hop, а не по длине окна."""
    assert frame_budget(1.0, hop=80)[0] == 2 * frame_budget(1.0, hop=160)[0]


# --------------------------------------------------------- normalize_log_mel
def test_preprocessing_takes_log10_then_applies_the_fixed_affine_map():
    assert flat(normalize_log_mel([[1.0, 1e-4]])) == APPROX([1.0, 0.0])


def test_dynamic_range_is_clamped_to_eight_log10_units_per_clip():
    """Whisper отрезает всё ниже пика клипа минус 8, а не использует mean/std."""
    assert flat(normalize_log_mel([[1.0, 1e-8, 1e-10, 0.0]])) == APPROX(
        [1.0, -1.0, -1.0, -1.0]
    )


def test_clamp_tracks_each_clip_peak_instead_of_corpus_statistics():
    quiet = flat(normalize_log_mel([[1e-2, 1e-12]]))
    loud = flat(normalize_log_mel([[1.0, 1e-12]]))
    assert quiet == APPROX([0.5, -1.5])
    assert loud == APPROX([1.0, -1.0])


def test_empty_spectrogram_stays_empty():
    assert normalize_log_mel([]) == []


def test_normalize_does_not_mutate_the_input():
    mel_power = [[1.0, 1e-4]]
    normalize_log_mel(mel_power)
    assert flat(mel_power) == APPROX([1.0, 1e-4])


# -------------------------------------------------------------- build_prompt
def test_prompt_starts_with_start_of_transcript():
    assert build_prompt("en")[0] == "<|startoftranscript|>"


def test_default_prompt_is_transcribe_without_timestamps():
    assert build_prompt("en") == [
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
    ]


def test_language_tag_follows_the_argument():
    """Один и тот же вес, другой тег — другой язык. В этом весь мультитаск."""
    assert build_prompt("fr")[1] == "<|fr|>"
    assert build_prompt("DE")[1] == "<|de|>"


def test_timestamps_flag_drops_the_notimestamps_token():
    assert "<|notimestamps|>" not in build_prompt("en", timestamps=True)


def test_unknown_task_is_rejected_loudly():
    """Молча свалиться на transcribe хуже: модель сделает не то, о чём просили."""
    with pytest.raises(ValueError):
        build_prompt("en", task="summarize")


# -------------------------------------------------------------- parse_prompt
def test_parse_is_the_inverse_of_build():
    for lang, task, ts in [("en", "transcribe", False), ("fr", "translate", True)]:
        parsed = parse_prompt(build_prompt(lang, task, ts))
        assert parsed == {"language": lang, "task": task, "timestamps": ts}


def test_parse_reads_the_translate_task():
    tokens = ["<|startoftranscript|>", "<|de|>", "<|translate|>"]
    assert parse_prompt(tokens)["task"] == "translate"


def test_timestamps_are_on_until_explicitly_disabled():
    tokens = ["<|startoftranscript|>", "<|de|>", "<|transcribe|>"]
    assert parse_prompt(tokens)["timestamps"] is True


def test_prompt_without_start_token_is_rejected():
    with pytest.raises(ValueError):
        parse_prompt(["<|en|>", "<|transcribe|>"])


# ------------------------------------------------------------ chunk_schedule
def test_short_clip_needs_a_single_window():
    assert chunk_schedule(10.0) == [(0.0, 10.0)]


def test_windows_step_by_chunk_minus_stride():
    assert chunk_schedule(70.0) == [(0.0, 30.0), (25.0, 55.0), (50.0, 70.0)]


def test_consecutive_windows_overlap_by_the_stride():
    """Перекрытие нужно, чтобы слово на границе окна не пропало в обоих сразу."""
    sched = chunk_schedule(200.0, chunk_s=30.0, stride_s=5.0)
    for (_, end), (start, _) in zip(sched, sched[1:]):
        assert end - start == APPROX(5.0)


def test_schedule_covers_the_whole_recording():
    sched = chunk_schedule(137.0)
    assert sched[0][0] == APPROX(0.0)
    assert sched[-1][1] == APPROX(137.0)
    assert all(s <= e for (_, e), (s, _) in zip(sched, sched[1:]))


def test_stride_not_smaller_than_the_chunk_is_rejected():
    """Иначе шаг нулевой и расписание строится бесконечно."""
    with pytest.raises(ValueError):
        chunk_schedule(100.0, chunk_s=30.0, stride_s=30.0)


# --------------------------------------------------- merge_chunk_transcripts
def test_overlapping_tail_is_written_once():
    assert merge_chunk_transcripts(["turn on the", "on the light"]) == "turn on the light"


def test_chunks_without_overlap_are_simply_joined():
    assert merge_chunk_transcripts(["hello world", "goodbye"]) == "hello world goodbye"


def test_identical_chunks_collapse_completely():
    """Окно, целиком повторившее предыдущее, не должно удваивать текст."""
    assert merge_chunk_transcripts(["a b c", "a b c"]) == "a b c"


def test_merge_matches_whole_words_not_characters():
    """Ловушка: по символам «the» из «then» дало бы ложный стык."""
    assert merge_chunk_transcripts(["turn on the", "then light"]) == "turn on the then light"


def test_empty_input_gives_an_empty_string():
    assert merge_chunk_transcripts([]) == ""


# --------------------------------------------------------- lora_parameter_count
def test_lora_adds_two_thin_matrices_per_module():
    assert lora_parameter_count([(1280, 1280)], 16) == 16 * (1280 + 1280)


def test_parameter_count_is_linear_in_the_rank():
    shapes = [(1280, 1280), (768, 1280)]
    assert lora_parameter_count(shapes, 32) == 2 * lora_parameter_count(shapes, 16)


def test_rank_zero_trains_nothing():
    assert lora_parameter_count([(1280, 1280)], 0) == 0


def test_lora_is_a_tiny_fraction_of_the_full_matrices():
    """Ловушка: r * d_in * d_out вместо r * (d_in + d_out) даёт БОЛЬШЕ, чем
    полное дообучение, а не меньше."""
    shapes = [(1280, 1280)] * 64          # q_proj и v_proj 32 слоёв
    full = sum(d_out * d_in for d_out, d_in in shapes)
    assert lora_parameter_count(shapes, 16) < 0.05 * full
