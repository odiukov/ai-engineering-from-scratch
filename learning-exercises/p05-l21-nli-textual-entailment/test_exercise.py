"""Тесты к уроку «NLI и textual entailment». Правь exercise.py."""

import pytest

from exercise import (
    has_negation,
    hypothesis_only_label,
    is_faithful,
    lexical_overlap,
    nli_scores,
    softmax,
    tokenize,
    zero_shot_classify,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CTX = "The Eiffel Tower is in Paris. It was completed in 1889."


def top_label(scores):
    """Метка с наибольшей вероятностью."""
    return max(scores, key=scores.get)


# ---------------------------------------------------------------- tokenize
def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("The cat is on the mat.") == ["the", "cat", "is", "on", "the", "mat"]


def test_tokenize_splits_apostrophes_and_shouting():
    assert tokenize("Nobody's HOME!") == ["nobody", "s", "home"]


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("") == []


# ------------------------------------------------------------- has_negation
def test_has_negation_finds_a_negation_word():
    assert has_negation(["there", "is", "no", "cat"]) is True


def test_plain_sentence_has_no_negation():
    assert has_negation(["there", "is", "a", "cat"]) is False


def test_negation_is_matched_as_a_whole_word():
    """Ловушка: "not" сидит внутри "note" и "cannot"."""
    assert has_negation(tokenize("The note is on the table.")) is False


def test_every_listed_negation_is_recognized():
    words = ["not", "no", "never", "nobody", "nothing", "none", "neither", "nor", "without"]
    assert all(has_negation([w]) for w in words)


# ---------------------------------------------------------- lexical_overlap
def test_overlap_counts_hypothesis_words_found_in_premise():
    assert lexical_overlap("A cat is sleeping on the couch.", "There is a cat.") == APPROX(0.75)


def test_overlap_of_unrelated_texts_is_zero():
    assert lexical_overlap("A cat is sleeping.", "The dog chased the ball.") == APPROX(0.0)


def test_overlap_is_asymmetric():
    """Длинный premise, покрывающий короткую гипотезу, — это 1.0, а не наоборот."""
    long_text = "The dog chased the ball in the park."
    short_text = "The dog chased the ball."
    assert lexical_overlap(long_text, short_text) == APPROX(1.0)
    assert lexical_overlap(short_text, long_text) < 1.0


def test_overlap_of_empty_hypothesis_is_zero_not_a_crash():
    assert lexical_overlap("A cat is sleeping.", "") == APPROX(0.0)


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    assert sum(softmax({"a": 2.0, "b": -1.0, "c": 0.5}).values()) == pytest.approx(1.0, abs=1e-12)


def test_softmax_of_equal_scores_is_uniform():
    assert softmax({"a": 0.0, "b": 0.0}) == {"a": APPROX(0.5), "b": APPROX(0.5)}


def test_softmax_preserves_the_ranking():
    probs = softmax({"low": -2.0, "high": 3.0, "mid": 0.0})
    assert probs["high"] > probs["mid"] > probs["low"]


def test_softmax_survives_huge_scores():
    """Без вычитания максимума math.exp(1000) даёт OverflowError."""
    probs = softmax({"a": 1000.0, "b": 999.0})
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-12)


def test_softmax_rejects_an_empty_dict():
    with pytest.raises(ValueError):
        softmax({})


# --------------------------------------------------------------- nli_scores
def test_nli_scores_are_a_distribution_over_three_labels():
    scores = nli_scores("A cat is sleeping on the couch.", "There is a cat.")
    assert set(scores) == {"entailment", "contradiction", "neutral"}
    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-12)


def test_high_overlap_without_negation_reads_as_entailment():
    assert top_label(nli_scores("A cat is sleeping on the couch.", "There is a cat.")) == "entailment"


def test_one_sided_negation_mirrors_entailment_into_contradiction():
    """Одна и та же поддержка: отрицание только решает, кому она достанется."""
    premise = "A cat is sleeping on the couch."
    plain = nli_scores(premise, "There is one cat on the couch.")
    negated = nli_scores(premise, "There is no cat on the couch.")
    assert top_label(negated) == "contradiction"
    assert negated["contradiction"] == APPROX(plain["entailment"])
    assert negated["entailment"] == APPROX(plain["contradiction"])


def test_negation_on_both_sides_cancels_out():
    """Ловушка: mismatch — исключающее ИЛИ, а не «где-то есть отрицание»."""
    scores = nli_scores("There is no cat in the room.", "There is no cat.")
    assert top_label(scores) == "entailment"


def test_unrelated_texts_read_as_neutral():
    assert top_label(nli_scores("A cat is sleeping.", "The dog chased the ball.")) == "neutral"


def test_half_overlap_is_the_point_of_indifference():
    """При перекрытии 0.5 поддержка нулевая, и entailment ровно равен neutral."""
    scores = nli_scores("A cat sleeps.", "A cat chased dogs.")
    assert scores["entailment"] == APPROX(scores["neutral"])


# ------------------------------------------------------- hypothesis_only_label
def test_negated_hypothesis_alone_reads_as_contradiction():
    assert hypothesis_only_label("Nobody is in the room.") == "contradiction"


def test_plain_hypothesis_alone_reads_as_neutral():
    assert hypothesis_only_label("A man plays guitar.") == "neutral"


def test_hypothesis_only_baseline_is_wrong_when_the_premise_agrees():
    """Вот и утечка метки: та же гипотеза с подтверждающим premise — entailment."""
    hypothesis = "There is no cat."
    assert hypothesis_only_label(hypothesis) == "contradiction"
    assert top_label(nli_scores("There is no cat in the room.", hypothesis)) == "entailment"


# --------------------------------------------------------- zero_shot_classify
def test_zero_shot_scores_sum_to_one():
    result = zero_shot_classify("The finance ministry cut interest rates today.",
                                ["finance", "sports", "politics"])
    assert sum(score for _, score in result) == pytest.approx(1.0, abs=1e-12)


def test_zero_shot_is_sorted_by_score_descending():
    result = zero_shot_classify("The finance ministry cut interest rates today.",
                                ["sports", "finance", "politics"])
    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True)


def test_zero_shot_picks_the_label_the_text_is_about():
    result = zero_shot_classify("The finance ministry cut interest rates today.",
                                ["finance", "sports"])
    assert result[0][0] == "finance"


def test_zero_shot_template_moves_the_scores():
    """Смена шаблона меняет разрыв между метками — то самое template sensitivity."""
    text = "Interest rates and the finance ministry dominated the news."
    default = dict(zero_shot_classify(text, ["finance", "sports"]))
    bare = dict(zero_shot_classify(text, ["finance", "sports"], "{label}"))
    assert bare["finance"] > default["finance"]


def test_zero_shot_keeps_input_order_on_ties():
    text = "The match ended late."
    assert zero_shot_classify(text, ["sports", "finance"], "{label}")[0][0] == "sports"
    assert zero_shot_classify(text, ["finance", "sports"], "{label}")[0][0] == "finance"


# --------------------------------------------------------------- is_faithful
def test_answer_copied_from_the_context_is_faithful():
    assert is_faithful("The Eiffel Tower is in Paris.", CTX, 1.0) is True


def test_one_invented_sentence_breaks_a_strict_threshold():
    answer = "The Eiffel Tower is in Paris. It is made of chocolate and floats."
    assert is_faithful(answer, CTX, 1.0) is False


def test_threshold_comparison_is_inclusive():
    """Половина подтверждённых claim-ов проходит порог 0.5, а не проваливает его."""
    answer = "The Eiffel Tower is in Paris. It is made of chocolate and floats."
    assert is_faithful(answer, CTX, 0.5) is True


def test_faithfulness_asks_whether_the_context_supports_the_answer():
    """Порядок аргументов несимметричен: context — premise, ответ — гипотеза."""
    assert is_faithful("Paris.", "The Eiffel Tower is in Paris.", 1.0) is True
    assert is_faithful("The Eiffel Tower is in Paris.", "Paris.", 1.0) is False


def test_empty_answer_is_rejected():
    with pytest.raises(ValueError):
        is_faithful("", CTX)
