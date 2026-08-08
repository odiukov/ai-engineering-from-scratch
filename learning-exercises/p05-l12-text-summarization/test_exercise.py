"""Тесты к уроку «Суммаризация текста». Правь exercise.py."""

import math

import pytest

from exercise import (
    hallucinated_entities,
    lcs_length,
    rouge_l,
    rouge_n,
    sentence_split,
    similarity,
    textrank_scores,
    textrank_summary,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ARTICLE = (
    "The city council approved the new transit plan. "
    "The transit plan adds three bus lines to the city. "
    "Local businesses welcomed the transit plan. "
    "Rain is expected on Thursday. "
    "The council said the plan starts in June. "
    "A cat was seen near the harbour."
)


# ---------------------------------------------------------- sentence_split
def test_sentence_split_breaks_on_terminal_punctuation():
    assert sentence_split("Hi there. How are you? Fine!") == [
        "Hi there.",
        "How are you?",
        "Fine!",
    ]


def test_sentence_split_keeps_the_punctuation_verbatim():
    """Экстрактивное резюме возвращает предложения дословно, вместе с точкой."""
    assert sentence_split("One sentence.")[0].endswith(".")


def test_sentence_split_of_blank_text_is_empty():
    assert sentence_split("   ") == []


def test_sentence_split_keeps_text_without_terminal_punctuation_whole():
    assert sentence_split("no full stop here") == ["no full stop here"]


# -------------------------------------------------------------- similarity
def test_similarity_of_disjoint_sentences_is_zero():
    assert similarity("alpha beta", "gamma delta") == 0.0


def test_similarity_is_symmetric():
    a, b = "the cat sat on the mat", "the cat ran away"
    assert similarity(a, b) == APPROX(similarity(b, a))


def test_similarity_of_a_sentence_with_itself_is_positive():
    assert similarity("the cat sat", "the cat sat") > 0.0


def test_similarity_normalisation_penalises_long_sentences():
    """Одно и то же одно совпадение весит меньше в длинных предложениях."""
    short = similarity("a b", "a c")
    long = similarity("a b p q r", "a c x y z")
    assert short > long > 0.0


def test_similarity_denominator_counts_repeated_tokens_in_sentence_length():
    """Повторы увеличивают token count, даже если distinct vocabulary тот же."""
    score = similarity("echo echo x", "echo echo y")
    assert score == APPROX(2 / (math.log(4) + math.log(4)))


def test_similarity_of_empty_sentences_is_zero():
    assert similarity("", "") == 0.0


# --------------------------------------------------------- textrank_scores
def test_textrank_returns_one_score_per_sentence():
    sentences = sentence_split(ARTICLE)
    assert len(textrank_scores(sentences)) == len(sentences)


def test_textrank_of_no_sentences_is_empty():
    assert textrank_scores([]) == []


def test_textrank_ranks_a_connected_sentence_above_an_isolated_one():
    """«transit plan» связано со всем текстом, кот у гавани — ни с чем."""
    sentences = sentence_split(ARTICLE)
    scores = textrank_scores(sentences)
    assert scores[1] > scores[-1]


def test_textrank_gives_identical_sentences_identical_scores():
    sentences = ["the cat sat.", "a dog ran.", "the cat sat."]
    scores = textrank_scores(sentences)
    assert scores[0] == APPROX(scores[2])


# -------------------------------------------------------- textrank_summary
def test_textrank_summary_returns_exactly_top_k_sentences():
    assert len(textrank_summary(ARTICLE, top_k=3)) == 3


def test_textrank_summary_never_invents_a_sentence():
    """Экстрактивная суммаризация не галлюцинирует по построению."""
    sentences = sentence_split(ARTICLE)
    assert all(s in sentences for s in textrank_summary(ARTICLE, top_k=3))


def test_textrank_summary_keeps_the_original_order():
    sentences = sentence_split(ARTICLE)
    chosen = textrank_summary(ARTICLE, top_k=3)
    positions = [sentences.index(s) for s in chosen]
    assert positions == sorted(positions)


def test_textrank_summary_returns_everything_when_the_text_is_short():
    text = "One. Two."
    assert textrank_summary(text, top_k=3) == sentence_split(text)


# -------------------------------------------------------------- lcs_length
def test_lcs_of_identical_sequences_is_their_length():
    tokens = "a b c d".split()
    assert lcs_length(tokens, tokens) == 4


def test_lcs_skips_over_an_inserted_word():
    """Подпоследовательность, а не подстрока: вставка не обнуляет счёт."""
    assert lcs_length(["a", "x", "b"], ["a", "y", "b"]) == 2


def test_lcs_of_disjoint_sequences_is_zero():
    assert lcs_length(["a", "b"], ["c", "d"]) == 0


def test_lcs_is_symmetric_and_handles_the_empty_case():
    a, b = "the cat sat on the mat".split(), "the mat was warm".split()
    assert lcs_length(a, b) == lcs_length(b, a)
    assert lcs_length([], a) == 0


# ----------------------------------------------------------------- rouge_n
def test_rouge_n_of_an_identical_summary_is_perfect():
    tokens = "the cat sat on the mat".split()
    assert rouge_n(tokens, tokens) == APPROX((1.0, 1.0, 1.0))


def test_rouge_n_of_a_disjoint_summary_is_zero():
    assert rouge_n(["p", "q"], ["x", "y"]) == (0.0, 0.0, 0.0)


def test_rouge_n_recall_measures_how_much_of_the_reference_is_covered():
    reference = "a b c d".split()
    precision, recall, _ = rouge_n("a b".split(), reference)
    assert precision == APPROX(1.0)
    assert recall == APPROX(0.5)


def test_rouge_n_bigrams_punish_reordering_that_unigrams_ignore():
    """Порядок слов не виден ROUGE-1 и решает всё для ROUGE-2."""
    reference = "the cat sat on the mat".split()
    shuffled = "mat the on sat cat the".split()
    assert rouge_n(shuffled, reference, n=1)[2] == APPROX(1.0)
    assert rouge_n(shuffled, reference, n=2)[2] < 0.3


def test_rouge_n_clips_a_repeated_word():
    """«the the the» не может покрыть референс, где «the» встречается один раз."""
    precision, _, _ = rouge_n(["the"] * 3, ["the", "cat"])
    assert precision == APPROX(1 / 3)


# ----------------------------------------------------------------- rouge_l
def test_rouge_l_of_an_identical_summary_is_perfect():
    tokens = "the cat sat on the mat".split()
    assert rouge_l(tokens, tokens) == APPROX((1.0, 1.0, 1.0))


def test_rouge_l_forgives_an_insertion_that_rouge_2_punishes():
    reference = "the cat sat on the mat".split()
    candidate = "the cat quickly sat on the mat".split()
    assert rouge_l(candidate, reference)[2] > rouge_n(candidate, reference, n=2)[2]


def test_rouge_l_still_cares_about_order():
    reference = "a b c d".split()
    assert rouge_l(reference, reference)[2] > rouge_l(list(reversed(reference)), reference)[2]


def test_rouge_l_of_an_empty_summary_is_zero():
    assert rouge_l([], "a b c".split()) == (0.0, 0.0, 0.0)


# ---------------------------------------------------- hallucinated_entities
SOURCE = "The board met. Yesterday John Smith rejected the 25,000 dollar offer."


def test_extractive_summary_never_hallucinates_an_entity():
    """Резюме, собранное из исходных предложений, не может добавить сущность."""
    assert hallucinated_entities(SOURCE, SOURCE) == []


def test_entity_swap_is_caught():
    summary = "The report says John Brown rejected the offer."
    assert hallucinated_entities(SOURCE, summary) == ["Brown"]


def test_number_drift_is_caught():
    summary = "The report says John Smith rejected 25 million dollars."
    assert hallucinated_entities(SOURCE, summary) == ["25"]


def test_a_summary_that_drops_entities_is_not_flagged():
    """Проверка ловит выдуманное, а не пропущенное: это precision, не recall."""
    assert hallucinated_entities(SOURCE, "The board rejected the offer.") == []
