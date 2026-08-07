"""Тесты к уроку «Генерация музыки». Правь exercise.py."""

import random

import pytest

from exercise import (
    chroma_vector,
    crossfade,
    fad,
    generate_tokens,
    is_prompt_blocked,
    midi_to_hz,
    repetition_rate,
    sample_token,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def cycle_model(context):
    """Заглушка вместо MusicGen: следующий токен всегда (последний + 1) % 4."""
    nxt = (context[-1] + 1) % 4
    return [10.0 if i == nxt else 0.0 for i in range(4)]


# ------------------------------------------------------------- midi_to_hz
def test_midi_note_69_is_440_hz():
    """A4 — опорная точка строя, от неё считается всё остальное."""
    assert midi_to_hz(69) == APPROX(440.0)


def test_midi_octave_up_doubles_the_frequency():
    assert midi_to_hz(81) == APPROX(880.0)
    assert midi_to_hz(57) == APPROX(220.0)


def test_midi_semitone_is_a_ratio_not_a_offset():
    """Полутон — умножение на 2**(1/12), поэтому шаг в герцах растёт с высотой."""
    low = midi_to_hz(70) - midi_to_hz(69)
    high = midi_to_hz(82) - midi_to_hz(81)
    assert high == APPROX(2 * low)


# ----------------------------------------------------------- chroma_vector
def test_chroma_vector_sums_to_one():
    assert sum(chroma_vector([60, 62, 67, 71])) == APPROX(1.0)


def test_chroma_vector_forgets_the_octave():
    """Одна и та же нота в разных октавах — один класс высоты."""
    assert chroma_vector([60]) == APPROX(chroma_vector([72, 48]))


def test_chroma_vector_splits_weight_between_pitch_classes():
    v = chroma_vector([60, 64])
    assert v[0] == APPROX(0.5)
    assert v[4] == APPROX(0.5)


def test_chroma_vector_rejects_an_empty_melody():
    with pytest.raises(ValueError):
        chroma_vector([])


# ------------------------------------------------------------ sample_token
def test_sample_token_at_zero_temperature_is_argmax():
    rng = random.Random(0)
    assert sample_token([0.0, 5.0, 1.0], rng, temperature=0) == 1


def test_sample_token_with_top_k_one_ignores_randomness():
    logits = [1.0, 0.5, 9.0, 0.2]
    assert {sample_token(logits, random.Random(s), top_k=1) for s in range(20)} == {2}


def test_sample_token_never_returns_a_token_cut_by_top_k():
    """Токен вне top_k имеет нулевую вероятность, сколько ни разыгрывай."""
    rng = random.Random(1)
    logits = [5.0, 4.9, 4.8]
    drawn = {sample_token(logits, rng, top_k=2) for _ in range(300)}
    assert drawn == {0, 1}


def test_sample_token_temperature_controls_diversity():
    """Холодная температура почти всегда даёт argmax, горячая — размазывает."""
    logits = [0.0, 1.0]
    cold_rng, hot_rng = random.Random(7), random.Random(7)
    cold = [sample_token(logits, cold_rng, temperature=0.05) for _ in range(400)]
    hot = [sample_token(logits, hot_rng, temperature=50.0) for _ in range(400)]
    assert sum(cold) > 395        # холодная: почти всегда argmax
    assert 150 < sum(hot) < 250   # горячая: почти честная монетка


def test_sample_token_survives_huge_logits():
    """Ловушка: math.exp(1000) переполняется, если не вычесть максимум."""
    assert sample_token([1000.0, 1001.0], random.Random(0)) in (0, 1)


def test_sample_token_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        sample_token([], random.Random(0))
    with pytest.raises(ValueError):
        sample_token([1.0, 2.0], random.Random(0), temperature=-1.0)
    with pytest.raises(ValueError):
        sample_token([1.0, 2.0], random.Random(0), top_k=5)


# --------------------------------------------------------- generate_tokens
def test_generate_tokens_returns_exactly_n_new_tokens():
    out = generate_tokens([0], cycle_model, 5, random.Random(0), temperature=0)
    assert len(out) == 5


def test_generate_tokens_feeds_its_own_output_back():
    """Авторегрессия: каждый шаг видит предыдущий, отсюда и цикл 1,2,3,0,..."""
    out = generate_tokens([0], cycle_model, 6, random.Random(0), temperature=0)
    assert out == [1, 2, 3, 0, 1, 2]


def test_generate_tokens_does_not_mutate_the_prompt():
    prompt = [0]
    generate_tokens(prompt, cycle_model, 4, random.Random(0), temperature=0)
    assert prompt == [0]


def test_generate_tokens_is_reproducible_for_the_same_seed():
    noisy = lambda ctx: [0.5, 0.5, 0.5, 0.5]
    a = generate_tokens([0], noisy, 20, random.Random(42))
    b = generate_tokens([0], noisy, 20, random.Random(42))
    assert a == b


# --------------------------------------------------------- repetition_rate
def test_repetition_rate_of_a_unique_sequence_is_zero():
    assert repetition_rate([1, 2, 3, 4, 5], window=2) == APPROX(0.0)


def test_repetition_rate_of_a_loop_is_high():
    """AR-модель зациклилась — метрика обязана это видеть."""
    assert repetition_rate([1, 2, 3, 0] * 5, window=4) > 0.7


def test_repetition_rate_counts_windows_not_tokens():
    assert repetition_rate([1, 2, 1, 2, 1, 2], window=2) == APPROX(0.6)


def test_repetition_rate_rejects_a_window_it_cannot_fit():
    with pytest.raises(ValueError):
        repetition_rate([1, 2, 3], window=4)
    with pytest.raises(ValueError):
        repetition_rate([1, 2, 3], window=0)


# ---------------------------------------------------------------- crossfade
def test_crossfade_shortens_the_result_by_the_overlap():
    out = crossfade([0.0] * 10, [0.0] * 6, 4)
    assert len(out) == 10 + 6 - 4


def test_crossfade_keeps_a_constant_signal_constant():
    """Веса в сумме дают 1 — иначе на стыке просядет громкость."""
    assert crossfade([1.0] * 5, [1.0] * 5, 3) == APPROX([1.0] * 7)


def test_crossfade_removes_the_click_at_the_seam():
    """Встык 1.0 → -1.0 даёт скачок 2.0; кроссфейд размазывает его плавно."""
    out = crossfade([1.0] * 8, [-1.0] * 8, 6)
    jumps = [abs(b - a) for a, b in zip(out, out[1:])]
    assert max(jumps) < 2.0
    overlap = out[2:9]
    assert all(b < a for a, b in zip(overlap, overlap[1:]))


def test_crossfade_with_zero_overlap_is_plain_concatenation():
    assert crossfade([1.0, 2.0], [3.0], 0) == APPROX([1.0, 2.0, 3.0])


def test_crossfade_rejects_an_overlap_longer_than_a_clip():
    with pytest.raises(ValueError):
        crossfade([1.0, 2.0], [3.0], 2)
    with pytest.raises(ValueError):
        crossfade([1.0, 2.0], [3.0, 4.0], -1)


# --------------------------------------------------------------------- fad
def test_fad_of_a_set_against_itself_is_zero():
    real = [[0.0, 1.0], [2.0, -1.0], [1.0, 3.0]]
    assert fad(real, real) == APPROX(0.0)


def test_fad_is_symmetric():
    a = [[0.0], [2.0], [4.0]]
    b = [[1.0], [1.0], [7.0]]
    assert fad(a, b) == APPROX(fad(b, a))


def test_fad_grows_with_the_distance_between_means():
    a = [[0.0], [1.0]]
    near = [[2.0], [3.0]]
    far = [[20.0], [21.0]]
    assert fad(a, far) > fad(a, near) > 0


def test_fad_sees_a_difference_in_spread_alone():
    """Средние совпадают, разброс разный — распределения всё равно разные."""
    tight = [[-1.0], [1.0]]
    wide = [[-10.0], [10.0]]
    assert fad(tight, wide) > 0


def test_fad_rejects_sets_it_cannot_describe():
    with pytest.raises(ValueError):
        fad([[1.0]], [[1.0], [2.0]])
    with pytest.raises(ValueError):
        fad([[1.0], [2.0]], [[1.0, 0.0], [2.0, 0.0]])


# ------------------------------------------------------- is_prompt_blocked
def test_blocked_artist_name_is_caught():
    assert is_prompt_blocked("song in the style of Taylor Swift", ["Taylor Swift"])


def test_block_list_is_case_insensitive():
    assert is_prompt_blocked("SONG LIKE taylor swift", ["Taylor Swift"])


def test_block_list_matches_whole_words_only():
    """Ловушка: наивный `in` находит Queen внутри queensland."""
    assert is_prompt_blocked("queensland ambient loop", ["Queen"]) is False


def test_clean_prompt_and_empty_block_list_pass():
    assert is_prompt_blocked("lo-fi hip-hop drums, 90 BPM", ["Taylor Swift"]) is False
    assert is_prompt_blocked("in the style of Taylor Swift", []) is False
