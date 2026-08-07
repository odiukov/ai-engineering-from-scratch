"""Тесты к уроку «Системы вопрос-ответ». Правь exercise.py."""

import pytest

from exercise import (
    answer_span,
    answer_with_refusal,
    best_span,
    exact_match,
    normalize_answer,
    recall_at_k,
    retrieve_top_k,
    token_f1,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CORPUS = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]


# --------------------------------------------------------- normalize_answer
def test_normalize_strips_case_and_punctuation():
    assert normalize_answer("The  Beatles!") == "beatles"


def test_normalize_keeps_digits_and_collapses_spaces():
    assert normalize_answer("June 29,   2007") == "june 29 2007"


def test_normalize_removes_articles_only_as_whole_words():
    """Ловушка: вырезание подстроки превратило бы theater в ater."""
    assert normalize_answer("the theater") == "theater"


def test_normalize_drops_punctuation_without_inserting_spaces():
    """SQuAD удаляет пунктуацию, а не заменяет её пробелом: don't -> dont."""
    assert normalize_answer("don't") == "dont"


def test_normalize_of_pure_punctuation_is_empty():
    assert normalize_answer("!!! ???") == ""


# ------------------------------------------------------------- exact_match
def test_exact_match_ignores_case_and_articles():
    assert exact_match("The Beatles", "beatles") == APPROX(1.0)


def test_exact_match_is_broken_by_an_ordinal_suffix():
    """Про это прямо сказано в уроке: 29th против 29 даёт EM = 0."""
    assert exact_match("June 29th, 2007", "June 29, 2007") == APPROX(0.0)


def test_exact_match_returns_zero_for_a_different_answer():
    assert exact_match("2001", "2007") == APPROX(0.0)


# ----------------------------------------------------------------- token_f1
def test_token_f1_gives_partial_credit_where_exact_match_gives_none():
    """Та же пара, что провалила EM, у F1 набирает две трети."""
    assert exact_match("June 29th 2007", "June 29 2007") == APPROX(0.0)
    assert token_f1("June 29th 2007", "June 29 2007") == APPROX(2 / 3)


def test_token_f1_is_one_for_an_exact_answer():
    assert token_f1("the cat", "a cat") == APPROX(1.0)


def test_token_f1_is_zero_without_shared_tokens():
    assert token_f1("dogs", "cats") == APPROX(0.0)


def test_token_f1_punishes_padding_the_answer():
    """Длинный ответ с тем же ядром теряет precision, значит и F1."""
    short = token_f1("June 29 2007", "June 29 2007")
    long = token_f1("June 29 2007 in California by Apple", "June 29 2007")
    assert long < short


def test_token_f1_counts_repeats_as_multiset():
    """Три раза "cat" против одного дают зачёт один, а не три."""
    assert token_f1("cat cat cat", "cat") == APPROX(0.5)


def test_token_f1_of_two_empty_answers_is_one():
    assert token_f1("!!!", "???") == APPROX(1.0)


def test_token_f1_of_one_empty_answer_is_zero():
    assert token_f1("!!!", "cat") == APPROX(0.0)


# ----------------------------------------------------------------- best_span
def test_best_span_picks_the_highest_scoring_pair():
    assert best_span([0.1, 5.0, 0.2], [0.3, 0.4, 9.0]) == (1, 2)


def test_best_span_never_returns_end_before_start():
    """Раздельный argmax дал бы (2, 0) — спан наизнанку."""
    start, end = best_span([0.0, 0.0, 9.0], [9.0, 0.0, 0.0])
    assert start <= end


def test_best_span_respects_the_length_limit():
    span = best_span([9.0, 0.0, 0.0], [0.0, 0.0, 9.0], max_answer_len=2)
    assert span[1] - span[0] + 1 <= 2


def test_best_span_of_length_one_collapses_to_a_single_token():
    assert best_span([9.0, 0.1], [0.1, 0.2], max_answer_len=1) == (0, 0)


def test_best_span_breaks_ties_towards_the_earliest_span():
    assert best_span([1.0, 1.0], [1.0, 1.0]) == (0, 0)


def test_best_span_rejects_mismatched_score_lengths():
    with pytest.raises(ValueError):
        best_span([1.0, 2.0], [1.0])


def test_best_span_rejects_empty_scores():
    with pytest.raises(ValueError):
        best_span([], [])


# ---------------------------------------------------------------- answer_span
def test_answer_span_joins_the_tokens_of_the_winning_span():
    tokens = ["it", "was", "June", "29"]
    assert answer_span(tokens, [0, 0, 5, 0], [0, 0, 0, 5]) == "June 29"


def test_answer_span_includes_the_last_token_of_the_span():
    """Срез должен идти до end + 1, иначе теряется хвост ответа."""
    tokens = ["Steve", "Jobs", "spoke"]
    assert answer_span(tokens, [5, 0, 0], [0, 5, 0]) == "Steve Jobs"


def test_answer_span_can_return_a_single_token():
    tokens = ["released", "in", "2007"]
    assert answer_span(tokens, [0, 0, 5], [0, 0, 5]) == "2007"


def test_answer_span_rejects_scores_that_do_not_match_the_tokens():
    with pytest.raises(ValueError):
        answer_span(["a", "b"], [1.0], [1.0])


# -------------------------------------------------------------- retrieve_top_k
def test_retrieve_puts_the_relevant_passage_first():
    ranking = retrieve_top_k("When was the first iPhone released?", CORPUS, top_k=2)
    assert ranking[0][1] == 0


def test_retrieve_returns_exactly_top_k_items():
    assert len(retrieve_top_k("iPhone", CORPUS, top_k=3)) == 3


def test_retrieve_score_is_a_fraction_of_the_question_words():
    """Три из четырёх слов вопроса нашлись в пассаже."""
    ranking = retrieve_top_k("when was the iPhone released", ["The iPhone was released in 2007."], 1)
    assert ranking[0][0] == APPROX(0.75)


def test_retrieve_gives_zero_to_a_passage_with_no_shared_words():
    ranking = retrieve_top_k("capital of Peru", ["Dogs bark loudly."], 1)
    assert ranking[0][0] == APPROX(0.0)


def test_retrieve_breaks_ties_by_index():
    ranking = retrieve_top_k("cat", ["a cat", "the cat"], top_k=2)
    assert [idx for _, idx in ranking] == [0, 1]


# ----------------------------------------------------------------- recall_at_k
def test_recall_counts_a_hit_inside_top_k():
    assert recall_at_k([[(0.9, 3), (0.2, 1)]], [3], top_k=1) == APPROX(1.0)


def test_recall_counts_a_miss_outside_top_k():
    assert recall_at_k([[(0.9, 3), (0.2, 1)]], [1], top_k=1) == APPROX(0.0)


def test_recall_never_decreases_when_k_grows():
    """Расширение окна может только добавить попаданий, но не отнять."""
    rankings = [[(0.9, 3), (0.2, 1)], [(0.8, 0), (0.1, 5)]]
    gold = [1, 5]
    values = [recall_at_k(rankings, gold, top_k=k) for k in (1, 2, 3)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_recall_averages_over_questions():
    rankings = [[(0.9, 0)], [(0.9, 1)]]
    assert recall_at_k(rankings, [0, 7], top_k=1) == APPROX(0.5)


def test_recall_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        recall_at_k([[(0.9, 0)]], [0, 1], top_k=1)


# ---------------------------------------------------------- answer_with_refusal
def test_answers_when_the_passage_is_close_enough():
    assert answer_with_refusal("who released the iPhone", ["Apple released the iPhone."], 0.5) == (
        "Apple released the iPhone."
    )


def test_refuses_when_nothing_relevant_was_retrieved():
    assert answer_with_refusal("what is the capital of Peru", CORPUS, 0.5) == "I don't know."


def test_raising_the_threshold_only_ever_adds_refusals():
    """Монотонность отказа: строже порог — ответов не становится больше."""
    question = "when was the first iPhone released"
    loose = answer_with_refusal(question, CORPUS, 0.1)
    strict = answer_with_refusal(question, CORPUS, 0.99)
    assert loose != "I don't know."
    assert strict == "I don't know."


def test_empty_corpus_is_a_refusal_not_a_crash():
    assert answer_with_refusal("anything at all", [], 0.5) == "I don't know."
