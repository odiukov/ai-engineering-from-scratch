"""Тесты к уроку «Распознавание речи: CTC, RNN-T, attention». Правь exercise.py."""

import pytest

from exercise import (
    collapse_ctc,
    count_ctc_alignments,
    ctc_beam_decode,
    ctc_greedy_decode,
    edit_counts,
    edit_distance,
    normalize_text,
    wer,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def _peak(index, size, high=0.9):
    """Кадр, где вся вероятность собрана на одном токене."""
    rest = (1.0 - high) / (size - 1)
    return [high if i == index else rest for i in range(size)]


# -------------------------------------------------------------- collapse_ctc
def test_collapse_merges_repeats_and_drops_blanks():
    assert collapse_ctc([1, 1, 0, 0, 1, 2, 2, 0, 3]) == [1, 1, 2, 3]


def test_adjacent_repeat_collapses_to_one_token():
    assert collapse_ctc([1, 1, 1]) == [1]


def test_blank_between_repeats_keeps_both():
    """Ради этого blank и существует: без него не написать «hello» и «сумма»."""
    assert collapse_ctc([1, 0, 1]) == [1, 1]


def test_all_blank_frames_decode_to_nothing():
    assert collapse_ctc([0, 0, 0]) == []


def test_blank_index_is_configurable():
    """Ловушка: blank не обязан быть нулём — на нуле может стоять буква."""
    assert collapse_ctc([0, 0, 5, 5, 0], blank=5) == [0, 0]


# --------------------------------------------------------- ctc_greedy_decode
def test_greedy_picks_the_argmax_of_every_frame():
    frames = [_peak(1, 3), _peak(1, 3), _peak(0, 3), _peak(2, 3)]
    assert ctc_greedy_decode(frames) == [1, 2]


def test_greedy_keeps_a_doubled_token_split_by_blank():
    frames = [_peak(1, 2), _peak(0, 2), _peak(1, 2)]
    assert ctc_greedy_decode(frames) == [1, 1]


def test_greedy_maps_ids_through_the_vocabulary():
    frames = [_peak(1, 3), _peak(0, 3), _peak(2, 3)]
    assert ctc_greedy_decode(frames, vocab="_ab") == "ab"


# ----------------------------------------------------------- ctc_beam_decode
def test_beam_agrees_with_greedy_on_confident_frames():
    frames = [_peak(1, 3), _peak(1, 3), _peak(0, 3), _peak(2, 3)]
    assert ctc_beam_decode(frames, beam=4) == ctc_greedy_decode(frames)


def test_beam_of_width_one_is_exactly_greedy():
    """Луч шириной 1 не имеет выбора — он обязан повторить жадный путь."""
    frames = [_peak(2, 4), _peak(0, 4), _peak(1, 4), _peak(1, 4), _peak(3, 4)]
    assert ctc_beam_decode(frames, beam=1) == ctc_greedy_decode(frames)


def test_beam_output_never_contains_the_blank():
    frames = [_peak(0, 3), _peak(1, 3), _peak(0, 3), _peak(2, 3), _peak(0, 3)]
    assert 0 not in ctc_beam_decode(frames, beam=6)


def test_beam_skeleton_cannot_emit_a_doubled_token():
    """Честное ограничение скелета из урока, а не ошибка в твоём коде.

    Настоящий prefix beam search хранит для гипотезы две вероятности —
    «закончилась blank» и «закончилась токеном» — и удвоение выдать умеет.
    Этот скелет складывает их в одну и потому не умеет.
    """
    frames = [_peak(1, 2), _peak(0, 2), _peak(1, 2)]
    assert ctc_greedy_decode(frames) == [1, 1]
    assert ctc_beam_decode(frames, beam=4) == [1]


# ------------------------------------------------------ count_ctc_alignments
def test_one_frame_per_token_leaves_a_single_alignment():
    assert count_ctc_alignments([1, 2], 2) == 1


def test_fewer_frames_than_tokens_is_impossible():
    assert count_ctc_alignments([1, 2], 1) == 0


def test_a_doubled_token_needs_a_blank_between():
    """[1,1] нельзя уложить в два кадра: [1,1] схлопнется обратно в [1]."""
    assert count_ctc_alignments([1, 1], 2) == 0
    assert count_ctc_alignments([1, 1], 3) == 1


def test_spare_frames_multiply_the_alignments():
    assert count_ctc_alignments([1], 2) == 3
    assert count_ctc_alignments([1, 2], 3) == 5


def test_alignment_count_grows_with_the_number_of_frames():
    """Отсюда и берётся сила CTC: разметка «кадр -> буква» не нужна."""
    counts = [count_ctc_alignments([1, 2, 3], t) for t in range(3, 10)]
    assert all(a < b for a, b in zip(counts, counts[1:]))
    assert counts[-1] > 100


# ------------------------------------------------------------- normalize_text
def test_normalize_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, WORLD!") == "hello world"


def test_normalize_collapses_runs_of_whitespace():
    assert normalize_text("  Don't   stop. ") == "dont stop"


def test_normalize_is_idempotent():
    """Применить дважды — то же, что один раз; иначе метрика зависит от того,
    сколько раз кто-то успел нормализовать строку."""
    text = "Turn ON the  kitchen lights, please!!"
    once = normalize_text(text)
    assert normalize_text(once) == once


# -------------------------------------------------------------- edit_distance
def test_identical_sequences_have_zero_distance():
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_distance_to_the_empty_sequence_is_its_length():
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance(["a", "b", "c"], []) == 3


def test_substitution_and_deletion_cost_one_each():
    assert edit_distance(["a", "b", "c"], ["a", "x"]) == 2


def test_edit_distance_is_symmetric():
    """Замена симметрична, а удаление и вставка взаимно обратны."""
    a = ["turn", "on", "the", "light"]
    b = ["turn", "off", "light", "now"]
    assert edit_distance(a, b) == edit_distance(b, a)


# ---------------------------------------------------------------- edit_counts
def test_counts_sum_to_the_edit_distance():
    ref = ["one", "two", "three", "four", "five"]
    hyp = ["one", "too", "three", "five", "extra"]
    c = edit_counts(ref, hyp)
    assert sum(c.values()) == edit_distance(ref, hyp)


def test_identical_sequences_produce_no_edits():
    assert edit_counts(["a", "b"], ["a", "b"]) == {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }


def test_a_missing_word_is_a_deletion():
    c = edit_counts(["a", "b"], ["a"])
    assert c["deletions"] == 1
    assert c["insertions"] == 0


def test_an_extra_word_is_an_insertion():
    """Галлюцинация на тишине выглядит именно так — сплошные вставки."""
    c = edit_counts(["a"], ["a", "thanks", "for", "watching"])
    assert c["insertions"] == 3
    assert c["deletions"] == 0


def test_a_wrong_word_is_a_substitution():
    c = edit_counts(["a", "b", "c"], ["a", "x", "c"])
    assert c["substitutions"] == 1
    assert c["deletions"] == 0 and c["insertions"] == 0


# ------------------------------------------------------------------------ wer
def test_perfect_transcript_has_zero_wer():
    assert wer("turn on the light", "turn on the light") == APPROX(0.0)


def test_wer_ignores_case_and_punctuation_after_normalization():
    """Главный источник фальшивых 100% WER: эталон капсом, гипотеза с точками."""
    assert wer("TURN ON THE LIGHT", "Turn on the light.") == APPROX(0.0)


def test_without_normalization_case_counts_as_errors():
    assert wer("TURN ON", "turn on", normalize=False) == APPROX(1.0)


def test_wer_divides_by_the_reference_length():
    assert wer("a b c d", "a x c") == APPROX(0.5)


def test_wer_can_exceed_one_when_the_model_hallucinates():
    """Ловушка: метрика не зажата в [0, 1]. Вставок бывает больше, чем слов."""
    assert wer("hello", "hello and thanks for watching this video") > 1.0


def test_empty_reference_does_not_divide_by_zero():
    assert wer("", "hello") == APPROX(1.0)
