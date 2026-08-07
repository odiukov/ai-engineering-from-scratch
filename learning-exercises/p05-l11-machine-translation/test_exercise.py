"""Тесты к уроку «Машинный перевод». Правь exercise.py."""

import math

import pytest

from exercise import (
    brevity_penalty,
    chrf,
    clipped_ngram_counts,
    corpus_bleu,
    flag_length_explosion,
    glossary_violations,
    ngrams,
    sentence_bleu,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

REF = "the cat sat on the mat today ok".split()
HYP_PREFIX = "the cat sat on the".split()


# ------------------------------------------------------------------ ngrams
def test_ngrams_slides_a_window_of_the_requested_size():
    assert ngrams(["the", "cat", "sat"], 2) == [("the", "cat"), ("cat", "sat")]


def test_ngrams_returns_nothing_when_the_window_is_wider_than_the_sentence():
    assert ngrams(["the"], 2) == []


def test_ngrams_of_size_one_is_one_tuple_per_token():
    assert ngrams(["a", "b", "c"], 1) == [("a",), ("b",), ("c",)]


def test_ngrams_rejects_a_non_positive_size():
    with pytest.raises(ValueError):
        ngrams(["a", "b"], 0)


# ------------------------------------------------------ clipped_ngram_counts
def test_clipped_counts_of_an_identical_hypothesis_match_everything():
    assert clipped_ngram_counts(REF, [REF], 2) == (len(REF) - 1, len(REF) - 1)


def test_clipped_counts_cap_a_repeated_word_at_the_reference_count():
    """Без обрезки перевод "the the the the" получил бы точность 1.0."""
    assert clipped_ngram_counts(["the"] * 4, [["the", "cat"]], 1) == (1, 4)


def test_clipped_counts_take_the_best_reference():
    hypothesis = ["a", "b"]
    references = [["a", "x"], ["y", "b"]]
    assert clipped_ngram_counts(hypothesis, references, 1) == (2, 2)


def test_clipped_counts_are_zero_when_nothing_overlaps():
    assert clipped_ngram_counts(["p", "q", "r"], [["x", "y", "z"]], 1) == (0, 3)


# --------------------------------------------------------- brevity_penalty
def test_brevity_penalty_is_one_when_the_translation_is_long_enough():
    assert brevity_penalty(10, 10) == 1.0
    assert brevity_penalty(15, 10) == 1.0


def test_brevity_penalty_punishes_a_short_translation():
    assert brevity_penalty(5, 10) == APPROX(math.exp(-1.0))


def test_brevity_penalty_of_an_empty_translation_is_zero():
    assert brevity_penalty(0, 10) == 0.0


def test_brevity_penalty_gets_harsher_as_the_translation_shrinks():
    assert brevity_penalty(8, 10) > brevity_penalty(4, 10) > brevity_penalty(2, 10)


# ------------------------------------------------------------ sentence_bleu
def test_bleu_of_an_identical_translation_is_a_perfect_hundred():
    assert sentence_bleu(REF, [REF]) == pytest.approx(100.0)


def test_bleu_of_a_translation_with_no_shared_ngrams_is_zero():
    assert sentence_bleu(["p", "q", "r", "s", "t"], [REF]) == 0.0


def test_bleu_punishes_a_truncated_translation_through_the_brevity_penalty():
    """Каждая n-грамма гипотезы есть в референсе, но перевод оборван."""
    score = sentence_bleu(HYP_PREFIX, [REF])
    expected = 100.0 * math.exp(1 - len(REF) / len(HYP_PREFIX))
    assert score == pytest.approx(expected)
    assert 0.0 < score < 100.0


def test_bleu_of_a_sentence_shorter_than_max_n_is_zero():
    """У BLEU нет сглаживания: нет 4-граммы — нет и счёта."""
    short = ["the", "cat"]
    assert sentence_bleu(short, [short]) == 0.0
    assert sentence_bleu(short, [short], max_n=2) == pytest.approx(100.0)


def test_bleu_accepts_several_references():
    hypothesis = "a b c d e".split()
    assert sentence_bleu(hypothesis, ["z z z z z".split(), hypothesis]) == pytest.approx(
        100.0
    )


# -------------------------------------------------------------- corpus_bleu
def test_corpus_bleu_of_a_perfect_corpus_is_a_hundred():
    corpus = ["a b c d e".split(), "f g h i j".split()]
    assert corpus_bleu(corpus, [[s] for s in corpus]) == pytest.approx(100.0)


def test_corpus_bleu_is_not_the_average_of_sentence_bleu():
    """Главная ловушка отчётности: усреднять по предложениям нельзя."""
    perfect = "a b c d e".split()
    partial = "a b c q r".split()
    hypotheses = [perfect, partial]
    references = [[perfect], [perfect]]
    per_sentence = [sentence_bleu(h, r) for h, r in zip(hypotheses, references)]
    assert per_sentence[0] == pytest.approx(100.0)
    assert per_sentence[1] == 0.0
    corpus = corpus_bleu(hypotheses, references)
    assert corpus != pytest.approx(sum(per_sentence) / 2)
    assert corpus > 60.0


def test_corpus_bleu_of_a_disjoint_corpus_is_zero():
    hypotheses = ["p q r s t".split()]
    references = [["a b c d e".split()]]
    assert corpus_bleu(hypotheses, references) == 0.0


def test_corpus_bleu_rejects_a_mismatched_number_of_references():
    with pytest.raises(ValueError):
        corpus_bleu([["a"], ["b"]], [[["a"]]])


# --------------------------------------------------------------------- chrf
def test_chrf_of_an_identical_translation_is_a_perfect_hundred():
    assert chrf("les chats courent", "les chats courent") == pytest.approx(100.0)


def test_chrf_of_a_translation_with_no_shared_characters_is_zero():
    assert chrf("xyz", "abq") == 0.0


def test_chrf_sees_a_shared_root_where_bleu_sees_nothing():
    """Морфология: BLEU считает «courent» и «court» разными словами, chrF — нет."""
    hypothesis, reference = "les chats courent", "le chat court"
    assert sentence_bleu(hypothesis.split(), [reference.split()]) == 0.0
    assert chrf(hypothesis, reference) > 30.0


def test_chrf_beta_two_weights_recall_over_precision():
    """Гипотеза — начало референса: precision высокий, recall низкий."""
    hypothesis, reference = "abcdef", "abcdefghijkl"
    assert chrf(hypothesis, reference, beta=2.0) < chrf(hypothesis, reference, beta=0.5)


def test_chrf_ignores_whitespace():
    assert chrf("les chats", "leschats") == pytest.approx(100.0)


# ------------------------------------------------------ flag_length_explosion
def test_length_explosion_stays_quiet_on_a_normal_translation():
    assert flag_length_explosion(["a"] * 10, ["b"] * 12) is False


def test_length_explosion_fires_when_short_input_becomes_long_output():
    assert flag_length_explosion(["a"] * 2, ["b"] * 20) is True


def test_length_explosion_does_not_fire_exactly_at_the_ratio():
    assert flag_length_explosion(["a"] * 4, ["b"] * 10, max_ratio=2.5) is False
    assert flag_length_explosion(["a"] * 4, ["b"] * 11, max_ratio=2.5) is True


def test_length_explosion_treats_output_from_an_empty_source_as_hallucination():
    assert flag_length_explosion([], ["b"]) is True
    assert flag_length_explosion([], []) is False


# ------------------------------------------------------- glossary_violations
def test_glossary_reports_a_term_translated_the_wrong_way():
    assert glossary_violations(
        "Sign up now", "Créez un compte", {"sign up": "s'inscrire"}
    ) == ["sign up"]


def test_glossary_stays_quiet_when_the_required_translation_is_present():
    assert (
        glossary_violations(
            "Sign up now", "Inscrivez-vous: s'inscrire", {"sign up": "s'inscrire"}
        )
        == []
    )


def test_glossary_ignores_terms_that_are_not_in_the_source():
    glossary = {"log out": "se déconnecter", "sign up": "s'inscrire"}
    assert glossary_violations("Sign up now", "Créez un compte", glossary) == ["sign up"]


def test_glossary_matching_is_case_insensitive_and_sorted():
    glossary = {"Sign Up": "S'inscrire", "Cart": "Panier"}
    assert glossary_violations("SIGN UP or open the CART", "rien", glossary) == [
        "Cart",
        "Sign Up",
    ]
