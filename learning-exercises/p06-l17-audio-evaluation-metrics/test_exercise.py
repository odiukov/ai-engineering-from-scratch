"""Тесты к уроку «Метрики оценки аудио». Правь exercise.py."""

import pytest

from exercise import (
    cer,
    cosine_similarity,
    der,
    edit_ops,
    frechet_distance_1d,
    normalize_text,
    percentile,
    wer,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


# ------------------------------------------------------------- normalize_text
def test_normalization_drops_case_and_punctuation():
    assert normalize_text("Please turn on the lights.") == "please turn on the lights"


def test_normalization_collapses_runs_of_whitespace():
    assert normalize_text("  Привет,   МИР! ") == "привет мир"


def test_normalization_is_idempotent():
    """Второй проход не должен ничего менять — иначе метрика зависит от порядка."""
    once = normalize_text("Hello,  World!")
    assert normalize_text(once) == once


def test_pure_punctuation_normalizes_to_nothing():
    assert normalize_text("!!! ??? ...") == ""


# ------------------------------------------------------------------ edit_ops
def test_identical_sequences_need_no_edits():
    assert edit_ops(["a", "b", "c"], ["a", "b", "c"]) == (0, 0, 0)


def test_wrong_word_counts_as_a_substitution():
    assert edit_ops(["a", "b", "c"], ["a", "x", "c"]) == (1, 0, 0)


def test_missing_word_counts_as_a_deletion():
    """Удаление — лишнее слово в эталоне, которого нет в гипотезе."""
    assert edit_ops(["a", "b", "c"], ["a", "c"]) == (0, 1, 0)


def test_extra_word_counts_as_an_insertion():
    """Вставка — лишнее слово в гипотезе: ASR придумал то, чего не было."""
    assert edit_ops(["a", "c"], ["a", "b", "c"]) == (0, 0, 1)


def test_empty_hypothesis_deletes_the_whole_reference():
    assert edit_ops(["a", "b"], []) == (0, 2, 0)


def test_edit_ops_prefers_the_cheapest_alignment():
    """Замена всего слова дешевле, чем удаление плюс вставка."""
    subs, dels, ins = edit_ops(["one", "two"], ["one", "five"])
    assert subs + dels + ins == 1


# ------------------------------------------------------------------------ wer
def test_one_wrong_word_out_of_five():
    assert wer("Please turn on the lights.", "please turn on the light") == APPROX(0.2)


def test_wer_of_a_perfect_transcript_is_zero():
    assert wer("Привет, мир!", "привет мир") == APPROX(0.0)


def test_wer_can_exceed_one_on_hallucinated_output():
    """Знаменатель — слова эталона, поэтому потолка у WER нет."""
    assert wer("да", "да да да да да") > 1.0


def test_wer_normalizes_before_scoring():
    """Без нормализации регистр и точка дали бы ошибку на ровном месте."""
    assert wer("Hello World.", "hello world") == APPROX(0.0)


def test_wer_rejects_an_empty_reference():
    with pytest.raises(ValueError):
        wer("...", "что-то")


# ------------------------------------------------------------------------ cer
def test_cer_counts_characters_not_words():
    assert cer("кот", "кит") == APPROX(1 / 3)


def test_cer_ignores_word_boundaries():
    """Для тонального языка сегментация условна — пробел не должен считаться."""
    assert cer("один два", "одиндва") == APPROX(0.0)


def test_cer_is_gentler_than_wer_on_a_one_letter_slip():
    """Одна буква портит слово целиком для WER, но лишь долю строки для CER."""
    assert cer("привет мир", "привет мор") < wer("привет мир", "привет мор")


# ------------------------------------------------------------ cosine_similarity
def test_identical_embeddings_give_similarity_one():
    assert cosine_similarity([1, 0], [1, 0]) == APPROX(1.0)


def test_orthogonal_embeddings_give_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_secs_does_not_depend_on_recording_loudness():
    """Косинус смотрит на направление: громче — не значит другой диктор."""
    assert cosine_similarity([1, 2, 3], [10, 20, 30]) == APPROX(1.0)


def test_zero_embedding_has_no_direction():
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])


# ------------------------------------------------------------------ percentile
def test_median_of_an_even_sample_interpolates():
    assert percentile([1, 2, 3, 4], 50) == APPROX(2.5)


def test_extremes_are_the_min_and_the_max():
    assert percentile([4, 1, 3, 2], 0) == APPROX(1.0)
    assert percentile([4, 1, 3, 2], 100) == APPROX(4.0)


def test_tail_latency_hides_behind_the_median():
    """Один зависший запрос тянет среднее, но не медиану — виден только в хвосте."""
    latencies = [100.0] * 99 + [5000.0]
    mean = sum(latencies) / len(latencies)
    assert percentile(latencies, 50) == APPROX(100.0)
    assert mean > percentile(latencies, 50)
    assert percentile(latencies, 100) == APPROX(5000.0)


def test_percentile_does_not_reorder_the_caller_list():
    """Ловушка: .sort() вместо sorted() молча переставит чужие замеры."""
    latencies = [3.0, 1.0, 2.0]
    percentile(latencies, 50)
    assert latencies == [3.0, 1.0, 2.0]


# ------------------------------------------------------------------------ der
def test_der_sums_all_three_error_kinds():
    assert der(1.0, 2.0, 3.0, 100.0) == APPROX(0.06)


def test_perfect_diarization_scores_zero():
    assert der(0.0, 0.0, 0.0, 100.0) == APPROX(0.0)


def test_der_can_exceed_one_hundred_percent():
    """Знаменатель — время речи, а false alarm бывает и на тишине."""
    assert der(200.0, 0.0, 0.0, 100.0) > 1.0


def test_der_rejects_zero_speech_time():
    with pytest.raises(ValueError):
        der(1.0, 1.0, 1.0, 0.0)


# ----------------------------------------------------------- frechet_distance_1d
def test_distance_to_the_same_distribution_is_zero():
    assert frechet_distance_1d([1, 2, 3], [1, 2, 3]) == APPROX(0.0)


def test_shifting_the_generated_set_costs_the_squared_shift():
    assert frechet_distance_1d([0, 0, 2, 2], [3, 3, 5, 5]) == APPROX(9.0)


def test_fad_is_symmetric():
    a, b = [1.0, 2.0, 5.0], [0.0, 4.0, 4.0]
    assert frechet_distance_1d(a, b) == APPROX(frechet_distance_1d(b, a))


def test_matching_mean_but_narrower_spread_is_still_penalised():
    """Модель звучит «в среднем правильно», но однообразно — FAD это ловит."""
    reference = [-2.0, -1.0, 1.0, 2.0]
    monotonous = [-0.1, 0.0, 0.0, 0.1]
    assert frechet_distance_1d(reference, monotonous) > 0.0


def test_fad_needs_a_distribution_not_a_single_sample():
    with pytest.raises(ValueError):
        frechet_distance_1d([1.0], [1.0, 2.0])
