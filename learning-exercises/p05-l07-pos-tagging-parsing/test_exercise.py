"""Тесты к уроку «POS-теггинг и синтаксический разбор». Правь exercise.py."""

import math

import pytest

from exercise import (
    count_hmm,
    extract_svo,
    laplace_logprob,
    predict_mft,
    ptb_to_ud,
    tag_accuracy,
    train_mft,
    viterbi,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def ambiguous_corpus():
    """Мини-корпус, в котором слово saw бывает и глаголом, и существительным.

    Существительным оно встречается 3 раза, глаголом 2 — значит baseline
    «самый частый тег» на нём всегда будет ошибаться в предложениях вида
    "i saw the film".
    """
    return [
        (["i", "saw", "the", "movie"], ["PRON", "VERB", "DET", "NOUN"]),
        (["you", "saw", "the", "film"], ["PRON", "VERB", "DET", "NOUN"]),
        (["the", "saw", "is", "sharp"], ["DET", "NOUN", "VERB", "ADJ"]),
        (["a", "saw", "is", "old"], ["DET", "NOUN", "VERB", "ADJ"]),
        (["the", "saw", "is", "new"], ["DET", "NOUN", "VERB", "ADJ"]),
    ]


# ---------------------------------------------------------------- ptb_to_ud
def test_ptb_plural_noun_maps_to_noun():
    assert ptb_to_ud("NNS") == "NOUN"


def test_ptb_proper_noun_is_propn_not_noun():
    """Ловушка: префикс NN не должен съедать имена собственные."""
    assert ptb_to_ud("NNP") == "PROPN"
    assert ptb_to_ud("NNPS") == "PROPN"


def test_ptb_verb_tenses_collapse_into_one_ud_tag():
    """UD грубее PTB: время и лицо теряются, все VB* становятся VERB."""
    assert {ptb_to_ud(t) for t in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")} == {"VERB"}


def test_ptb_unknown_tag_falls_back_to_x():
    assert ptb_to_ud("ZZZ") == "X"


def test_ptb_punctuation_maps_to_punct():
    assert ptb_to_ud(".") == "PUNCT"


# ---------------------------------------------------------------- train_mft
def test_train_mft_picks_the_most_frequent_tag_per_word():
    word_best, _ = train_mft(ambiguous_corpus())
    assert word_best["saw"] == "NOUN"


def test_train_mft_lowercases_the_vocabulary():
    word_best, _ = train_mft([(["The", "Cat"], ["DET", "NOUN"])])
    assert word_best == {"the": "DET", "cat": "NOUN"}


def test_train_mft_breaks_ties_alphabetically():
    """DET и NOUN встречаются по 5 раз — ответ обязан быть детерминированным."""
    _, default_tag = train_mft(ambiguous_corpus())
    assert default_tag == "DET"


def test_train_mft_on_empty_corpus_has_no_default_tag():
    assert train_mft([]) == ({}, None)


# -------------------------------------------------------------- predict_mft
def test_predict_mft_uses_the_table():
    word_best, default_tag = train_mft(ambiguous_corpus())
    assert predict_mft(["the", "film"], word_best, default_tag) == ["DET", "NOUN"]


def test_predict_mft_falls_back_on_unseen_words():
    assert predict_mft(["zzz"], {"the": "DET"}, "NOUN") == ["NOUN"]


def test_predict_mft_is_case_insensitive():
    """Слово в начале предложения с большой буквы — не повод терять тег."""
    word_best, default_tag = train_mft(ambiguous_corpus())
    assert predict_mft(["The"], word_best, default_tag) == ["DET"]


# ------------------------------------------------------------- tag_accuracy
def test_tag_accuracy_of_perfect_prediction_is_one():
    assert tag_accuracy(["DET", "NOUN"], ["DET", "NOUN"]) == APPROX(1.0)


def test_tag_accuracy_counts_partial_matches():
    assert tag_accuracy(["DET", "VERB"], ["DET", "NOUN"]) == APPROX(0.5)


def test_tag_accuracy_of_empty_sequences_is_zero():
    assert tag_accuracy([], []) == APPROX(0.0)


def test_tag_accuracy_rejects_length_mismatch():
    """Молча обрезать по zip нельзя — отчёт получится враньём."""
    with pytest.raises(ValueError):
        tag_accuracy(["DET"], ["DET", "NOUN"])


# ----------------------------------------------------------------- count_hmm
def test_count_hmm_records_bos_and_eos_transitions():
    transitions, _, _, _ = count_hmm([(["The", "cat"], ["DET", "NOUN"])])
    assert transitions["<BOS>"] == {"DET": 1}
    assert transitions["DET"] == {"NOUN": 1}
    assert transitions["NOUN"] == {"<EOS>": 1}


def test_count_hmm_emissions_are_lowercased():
    _, emissions, _, _ = count_hmm([(["The", "Cat"], ["DET", "NOUN"])])
    assert emissions == {"DET": {"the": 1}, "NOUN": {"cat": 1}}


def test_count_hmm_returns_sorted_tags_and_vocab():
    """Порядок тегов задаёт порядок столбцов решётки — он обязан быть стабильным."""
    _, _, tags, vocab = count_hmm(ambiguous_corpus())
    assert tags == sorted(tags) and vocab == sorted(vocab)
    assert "<BOS>" not in tags and "<EOS>" not in tags


def test_count_hmm_transition_counts_match_the_corpus():
    transitions, _, _, _ = count_hmm(ambiguous_corpus())
    # PRON встречается дважды, и оба раза за ним идёт VERB
    assert transitions["PRON"] == {"VERB": 2}


# ----------------------------------------------------------- laplace_logprob
def test_laplace_logprob_matches_the_formula():
    assert laplace_logprob({"a": 3, "b": 1}, "a", 4, alpha=1.0) == APPROX(math.log(0.5))


def test_laplace_logprob_gives_unseen_outcomes_a_finite_score():
    """Без сглаживания было бы log(0) = -inf, и весь путь Витерби умирал бы."""
    value = laplace_logprob({"a": 3, "b": 1}, "c", 4, alpha=1.0)
    assert math.isfinite(value)
    assert value == APPROX(math.log(1 / 8))


def test_laplace_probabilities_sum_to_one_over_all_outcomes():
    counts = {"a": 3, "b": 1}
    total = sum(
        math.exp(laplace_logprob(counts, k, 4, alpha=0.5)) for k in ("a", "b", "c", "d")
    )
    assert total == APPROX(1.0)


def test_laplace_logprob_ranks_seen_above_unseen():
    counts = {"a": 3, "b": 1}
    assert laplace_logprob(counts, "a", 4) > laplace_logprob(counts, "b", 4)
    assert laplace_logprob(counts, "b", 4) > laplace_logprob(counts, "z", 4)


# ------------------------------------------------------------------ viterbi
def test_viterbi_on_empty_input_returns_empty():
    transitions, emissions, tags, vocab = count_hmm(ambiguous_corpus())
    assert viterbi([], transitions, emissions, tags, vocab) == []


def test_viterbi_returns_one_tag_per_token_from_the_tagset():
    transitions, emissions, tags, vocab = count_hmm(ambiguous_corpus())
    tokens = ["the", "unseen", "word", "here"]
    result = viterbi(tokens, transitions, emissions, tags, vocab)
    assert len(result) == len(tokens)
    assert set(result) <= set(tags)


def test_viterbi_recovers_gold_tags_of_a_training_sentence():
    transitions, emissions, tags, vocab = count_hmm(ambiguous_corpus())
    result = viterbi(["the", "saw", "is", "sharp"], transitions, emissions, tags, vocab)
    assert result == ["DET", "NOUN", "VERB", "ADJ"]


def test_viterbi_beats_the_baseline_on_noun_verb_ambiguity():
    """Главный сюжет урока: переходы решают то, чего частотность не может.

    Baseline видит только слово saw и всегда говорит NOUN. HMM видит, что
    перед ним PRON, а после PRON в корпусе всегда шёл VERB.
    """
    corpus = ambiguous_corpus()
    gold = ["PRON", "VERB", "DET", "NOUN"]
    tokens = ["i", "saw", "the", "film"]

    word_best, default_tag = train_mft(corpus)
    transitions, emissions, tags, vocab = count_hmm(corpus)

    baseline = tag_accuracy(predict_mft(tokens, word_best, default_tag), gold)
    hmm = tag_accuracy(viterbi(tokens, transitions, emissions, tags, vocab), gold)
    assert hmm > baseline
    assert hmm == APPROX(1.0)


def test_viterbi_is_case_insensitive():
    transitions, emissions, tags, vocab = count_hmm(ambiguous_corpus())
    lower = viterbi(["the", "film"], transitions, emissions, tags, vocab)
    upper = viterbi(["The", "Film"], transitions, emissions, tags, vocab)
    assert lower == upper


# --------------------------------------------------------------- extract_svo
def test_extract_svo_finds_the_triple():
    tokens = ["cats", "eat", "fish"]
    arcs = [(-1, 1, "ROOT"), (1, 0, "nsubj"), (1, 2, "dobj")]
    assert extract_svo(tokens, arcs) == [("cats", "eat", "fish")]


def test_extract_svo_skips_verbs_without_an_object():
    tokens = ["cats", "sleep"]
    arcs = [(-1, 1, "ROOT"), (1, 0, "nsubj")]
    assert extract_svo(tokens, arcs) == []


def test_extract_svo_ignores_unrelated_relations():
    """det, aux и prep не участвуют в тройке — их наличие ничего не меняет."""
    tokens = ["the", "cats", "were", "eating", "fish"]
    arcs = [
        (-1, 3, "ROOT"),
        (1, 0, "det"),
        (3, 1, "nsubj"),
        (3, 2, "aux"),
        (3, 4, "dobj"),
    ]
    assert extract_svo(tokens, arcs) == [("cats", "eating", "fish")]


def test_extract_svo_returns_triples_ordered_by_verb():
    tokens = ["dogs", "chase", "cats", "and", "cats", "catch", "mice"]
    arcs = [
        (5, 1, "conj"),
        (1, 0, "nsubj"),
        (1, 2, "dobj"),
        (5, 4, "nsubj"),
        (5, 6, "dobj"),
        (-1, 5, "ROOT"),
    ]
    assert extract_svo(tokens, arcs) == [
        ("dogs", "chase", "cats"),
        ("cats", "catch", "mice"),
    ]
