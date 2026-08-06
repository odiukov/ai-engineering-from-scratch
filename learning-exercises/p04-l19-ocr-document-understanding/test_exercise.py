"""Тесты к уроку «OCR и понимание документов». Правь exercise.py."""

import pytest

from exercise import (
    cer,
    ctc_collapse,
    decode_text,
    field_f1,
    greedy_ctc_decode,
    levenshtein,
    reading_order,
    wer,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

VOCAB = ["_", "h", "e", "l", "o"]


def onehot(indices, size, hot=0.0, cold=-9.0):
    """Таблица log-вероятностей, где на каждом шаге победитель задан явно."""
    return [[hot if c == i else cold for c in range(size)] for i in indices]


# ----------------------------------------------------------- ctc_collapse
def test_collapse_merges_repeats_then_drops_blanks():
    assert ctc_collapse([1, 1, 1, 0, 0, 2, 2, 3, 3, 0, 3, 3, 4]) == [1, 2, 3, 3, 4]


def test_blank_between_repeats_preserves_the_double_letter():
    """Главный трюк CTC: blank разрывает серию, и "ll" в hello выживает."""
    assert ctc_collapse([3, 0, 3]) == [3, 3]


def test_repeats_without_a_blank_collapse_to_one():
    assert ctc_collapse([3, 3, 3]) == [3]


def test_all_blank_decodes_to_empty():
    assert ctc_collapse([0, 0, 0]) == []


def test_collapse_honours_a_custom_blank_index():
    """blank не обязан быть нулём — вокабуляры бывают разные."""
    assert ctc_collapse([0, 5, 5, 0], blank=5) == [0, 0]


# ------------------------------------------------------ greedy_ctc_decode
def test_greedy_decode_picks_argmax_then_collapses():
    assert greedy_ctc_decode([[0.0, -9.0], [-9.0, 0.0], [-9.0, 0.0]]) == [1]


def test_greedy_decode_ignores_absolute_scores():
    """Важен только порядок внутри шага: сдвиг всех логитов ничего не меняет."""
    a = [[0.0, -1.0, -2.0], [-5.0, -1.0, -9.0]]
    b = [[100.0, 99.0, 98.0], [95.0, 99.0, 91.0]]
    assert greedy_ctc_decode(a) == greedy_ctc_decode(b)


def test_greedy_decode_recovers_hello():
    frames = [1, 1, 1, 0, 0, 2, 2, 3, 3, 0, 3, 3, 4]
    assert greedy_ctc_decode(onehot(frames, 5)) == [1, 2, 3, 3, 4]


# ------------------------------------------------------------- decode_text
def test_decode_text_maps_indices_through_vocab():
    assert decode_text(onehot([0, 1, 1, 2], 3), ["_", "a", "b"]) == "ab"


def test_decode_text_spells_hello():
    frames = [1, 1, 1, 0, 0, 2, 2, 3, 3, 0, 3, 3, 4]
    assert decode_text(onehot(frames, 5), VOCAB) == "hello"


def test_decode_text_of_silence_is_empty_string():
    assert decode_text(onehot([0, 0, 0], 5), VOCAB) == ""


# ------------------------------------------------------------ levenshtein
def test_levenshtein_classic_kitten_sitting():
    assert levenshtein("kitten", "sitting") == 3


def test_levenshtein_of_equal_strings_is_zero():
    assert levenshtein("abc", "abc") == 0


def test_levenshtein_against_empty_is_the_length():
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3


def test_levenshtein_is_symmetric():
    assert levenshtein("flaw", "lawn") == levenshtein("lawn", "flaw")


def test_levenshtein_satisfies_the_triangle_inequality():
    a, b, c = "receipt", "recipe", "recipes"
    assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)


def test_levenshtein_works_on_lists_of_words():
    assert levenshtein(["the", "cat"], ["the", "dog"]) == 1


# --------------------------------------------------------------------- cer
def test_cer_counts_one_deletion_in_five():
    assert cer("hello", "helo") == APPROX(0.2)


def test_cer_of_a_perfect_read_is_zero():
    assert cer("hello", "hello") == APPROX(0.0)


def test_cer_divides_by_the_reference_not_the_hypothesis():
    """Длинная каша вместо короткого эталона обязана дать CER > 1."""
    assert cer("ab", "abcdefghij") > 1.0


def test_cer_of_empty_reference_is_defined():
    assert cer("", "") == APPROX(0.0)
    assert cer("", "junk") == APPROX(1.0)


# --------------------------------------------------------------------- wer
def test_wer_counts_one_wrong_word_in_three():
    assert wer("the cat sat", "the cat sit") == APPROX(1 / 3)


def test_wer_of_a_perfect_read_is_zero():
    assert wer("the cat sat", "the cat sat") == APPROX(0.0)


def test_wer_is_harsher_than_cer_on_a_single_typo():
    """Опечатка стоит одной буквы для CER и целого слова для WER."""
    ref, hyp = "the cat sat", "the cat sit"
    assert wer(ref, hyp) > cer(ref, hyp)


def test_wer_ignores_extra_whitespace():
    assert wer("the  cat", "the cat") == APPROX(0.0)


# ----------------------------------------------------------- reading_order
def test_reading_order_sorts_one_line_left_to_right():
    assert reading_order([(50, 0, 60, 10), (0, 2, 10, 12)]) == [1, 0]


def test_reading_order_sorts_lines_top_to_bottom():
    assert reading_order([(0, 100, 10, 110), (0, 0, 10, 10)]) == [1, 0]


def test_reading_order_groups_a_jittery_line_together():
    """Слова одной строки сдвинуты на пару пикселей — это всё ещё одна строка."""
    boxes = [(30, 3, 40, 13), (0, 0, 10, 10), (15, 1, 25, 11)]
    assert reading_order(boxes) == [1, 2, 0]


def test_reading_order_splits_when_the_gap_exceeds_the_tolerance():
    boxes = [(30, 0, 40, 10), (0, 40, 10, 50)]
    assert reading_order(boxes, line_tol=5.0) == [0, 1]


def test_huge_tolerance_flattens_everything_into_one_line():
    boxes = [(0, 100, 10, 110), (50, 0, 60, 10)]
    assert reading_order(boxes, line_tol=1000.0) == [0, 1]


def test_reading_order_of_nothing_is_nothing():
    assert reading_order([]) == []


# --------------------------------------------------------------- field_f1
def test_field_f1_on_an_exact_match():
    assert field_f1({"total": "42.50"}, {"total": "42.50"}) == APPROX((1.0, 1.0, 1.0))


def test_field_f1_penalises_an_invented_field():
    p, r, f = field_f1({"total": "42.50", "date": "2026-01-01"}, {"total": "42.50"})
    assert (p, r) == APPROX((0.5, 1.0))
    assert f == APPROX(2 / 3)


def test_field_f1_penalises_a_missing_field():
    p, r, f = field_f1({"total": "42.50"}, {"total": "42.50", "date": "2026-01-01"})
    assert (p, r) == APPROX((1.0, 0.5))


def test_right_key_with_wrong_value_scores_zero():
    """Ключ угадан, значение — нет. Для извлечения полей это провал."""
    assert field_f1({"total": "4.25"}, {"total": "42.50"}) == APPROX((0.0, 0.0, 0.0))


def test_field_f1_of_two_empty_dicts_does_not_divide_by_zero():
    assert field_f1({}, {}) == APPROX((0.0, 0.0, 0.0))
