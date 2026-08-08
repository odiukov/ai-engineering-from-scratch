"""Тесты к уроку «Генерация текста до трансформеров». Правь exercise.py."""

import random

import pytest

from exercise import (
    bits_per_token,
    continuation_probability,
    generate,
    kneser_ney_bigram,
    laplace_probability,
    perplexity,
    raw_probability,
    train_ngram,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

SMALL = [["a", "b"], ["a", "c"]]

# "francisco" встречается 5 раз, но всегда после "san";
# "cat" встречается те же 5 раз, но каждый раз в новом контексте
FRANCISCO = [["san", "francisco"]] * 5 + [
    ["the", "cat"],
    ["a", "cat"],
    ["big", "cat"],
    ["my", "cat"],
    ["one", "cat"],
]


# --------------------------------------------------------------- train_ngram
def test_train_ngram_pads_with_one_start_and_one_end_for_bigrams():
    ngrams, contexts = train_ngram([["a", "b"]], n=2)
    assert dict(ngrams) == {("<s>", "a"): 1, ("a", "b"): 1, ("b", "</s>"): 1}
    assert dict(contexts) == {("<s>",): 1, ("a",): 1, ("b",): 1}


def test_train_ngram_pads_with_two_starts_for_trigrams():
    ngrams, _ = train_ngram([["a"]], n=3)
    assert dict(ngrams) == {("<s>", "<s>", "a"): 1, ("<s>", "a", "</s>"): 1}


def test_train_ngram_accumulates_across_sentences():
    ngrams, contexts = train_ngram([["a"], ["a"]], n=2)
    assert ngrams[("<s>", "a")] == 2
    assert contexts[("<s>",)] == 2


def test_train_ngram_never_makes_the_end_token_a_context():
    """Ловушка: contexts считает «за ним что-то следовало», а за </s> ничего."""
    _, contexts = train_ngram([["a", "b"], ["c"]], n=2)
    assert ("</s>",) not in contexts


# ---------------------------------------------------------- raw_probability
def test_raw_probability_splits_mass_between_seen_continuations():
    ngrams, contexts = train_ngram(SMALL, n=2)
    assert raw_probability(ngrams, contexts, ["a"], "b") == APPROX(0.5)


def test_raw_probability_of_an_unseen_word_is_zero():
    ngrams, contexts = train_ngram(SMALL, n=2)
    assert raw_probability(ngrams, contexts, ["a"], "zzz") == APPROX(0.0)


def test_raw_probability_of_an_unknown_context_is_zero_not_a_crash():
    """Ловушка: count(context) == 0, делить нельзя."""
    ngrams, contexts = train_ngram(SMALL, n=2)
    assert raw_probability(ngrams, contexts, ["zzz"], "b") == APPROX(0.0)


def test_raw_probability_sums_to_one_over_seen_continuations():
    ngrams, contexts = train_ngram(SMALL, n=2)
    total = sum(raw_probability(ngrams, contexts, ["a"], w) for w in ["b", "c"])
    assert total == APPROX(1.0)


# ------------------------------------------------------ laplace_probability
def test_laplace_probability_adds_one_to_the_count():
    ngrams, contexts = train_ngram(SMALL, n=2)
    assert laplace_probability(ngrams, contexts, 4, ["a"], "b") == APPROX(2 / 6)


def test_laplace_probability_of_an_unseen_word_is_positive():
    """Смысл сглаживания: нулей больше нет, log-likelihood конечен."""
    ngrams, contexts = train_ngram(SMALL, n=2)
    assert laplace_probability(ngrams, contexts, 4, ["a"], "zzz") > 0.0


def test_laplace_probability_sums_to_one_over_the_whole_vocabulary():
    """Знаменатель прибавляет vocab_size именно ради этого."""
    vocab = ["a", "b", "c", "</s>"]
    ngrams, contexts = train_ngram(SMALL, n=2)
    total = sum(
        laplace_probability(ngrams, contexts, len(vocab), ["a"], w) for w in vocab
    )
    assert total == APPROX(1.0)


def test_laplace_probability_takes_mass_away_from_the_seen_bigram():
    ngrams, contexts = train_ngram(SMALL, n=2)
    smoothed = laplace_probability(ngrams, contexts, 4, ["a"], "b")
    assert smoothed < raw_probability(ngrams, contexts, ["a"], "b")


# ------------------------------------------------- continuation_probability
def test_continuation_probability_sums_to_one():
    assert sum(continuation_probability(FRANCISCO).values()) == APPROX(1.0)


def test_continuation_probability_counts_distinct_predecessors():
    """У "b" два разных предшественника, у "a" и "c" по одному."""
    p_cont = continuation_probability([["a", "b"], ["c", "b"]])
    assert p_cont["b"] > p_cont["a"]
    assert p_cont["a"] == APPROX(p_cont["c"])


def test_continuation_probability_punishes_the_single_context_word():
    """Идея Кнесера-Нея: "francisco" и "cat" равночастотны, но не равноценны."""
    p_cont = continuation_probability(FRANCISCO)
    assert p_cont["francisco"] < p_cont["cat"]


def test_continuation_probability_has_no_entry_for_the_start_token():
    """"<s>" никогда ни за кем не следует, продолжением быть не может."""
    assert "<s>" not in continuation_probability(SMALL)


# ------------------------------------------------------- kneser_ney_bigram
def test_kneser_ney_is_a_distribution_for_a_known_context():
    """Сколько скидка забрала, столько lambda и вернула — сумма ровно 1."""
    model = kneser_ney_bigram(SMALL)
    words = list(continuation_probability(SMALL))
    assert sum(model(("a",), w) for w in words) == APPROX(1.0)


def test_kneser_ney_gives_an_unseen_bigram_a_positive_probability():
    ngrams, contexts = train_ngram(SMALL, n=2)
    model = kneser_ney_bigram(SMALL)
    assert raw_probability(ngrams, contexts, ["a"], "c") == APPROX(0.5)
    assert model(("b",), "c") > 0.0


def test_kneser_ney_discounts_the_seen_bigram_below_the_mle():
    ngrams, contexts = train_ngram(SMALL, n=2)
    model = kneser_ney_bigram(SMALL)
    assert model(("a",), "b") < raw_probability(ngrams, contexts, ["a"], "b")


def test_kneser_ney_backs_off_to_continuation_for_an_unknown_context():
    model = kneser_ney_bigram(SMALL)
    p_cont = continuation_probability(SMALL)
    assert model(("zzz",), "b") == APPROX(p_cont["b"])


def test_kneser_ney_prefers_the_many_context_word_in_a_novel_context():
    """Ради этого всё и затевалось: "cat" продолжает новый контекст, не "francisco"."""
    model = kneser_ney_bigram(FRANCISCO)
    assert model(("zzz",), "cat") > model(("zzz",), "francisco")


# ---------------------------------------------------------- bits_per_token
def test_bits_per_token_of_a_uniform_model_over_four_words_is_two():
    assert bits_per_token(lambda prev, w: 0.25, [["a", "b"]]) == APPROX(2.0)


def test_bits_per_token_of_a_perfect_model_is_zero():
    assert bits_per_token(lambda prev, w: 1.0, [["a", "b"]]) == APPROX(0.0)


def test_bits_per_token_survives_a_zero_probability():
    """Ловушка: log2(0) — это ValueError, нужен пол вроде 1e-12."""
    assert bits_per_token(lambda prev, w: 0.0, [["a"]]) > 30.0


def test_bits_per_token_is_lower_for_the_better_model():
    held_out = [["a", "b"]]
    smart = kneser_ney_bigram(SMALL)
    assert bits_per_token(smart, held_out) < bits_per_token(lambda p, w: 0.02, held_out)


def test_bits_per_token_passes_two_token_contexts_to_a_trigram_model():
    ngrams, contexts = train_ngram([["a", "b"]], n=3)
    model = lambda context, word: raw_probability(ngrams, contexts, context, word)
    assert bits_per_token(model, [["a", "b"]], n=3) == APPROX(0.0)


# --------------------------------------------------------------- perplexity
def test_perplexity_of_a_uniform_model_over_four_words_is_four():
    assert perplexity(lambda prev, w: 0.25, [["a", "b"]]) == APPROX(4.0)


def test_perplexity_of_a_perfect_model_is_one():
    assert perplexity(lambda prev, w: 1.0, [["a", "b"]]) == APPROX(1.0)


def test_perplexity_equals_two_to_the_bits_per_token():
    """Одна величина в двух единицах: нат против бита."""
    model = kneser_ney_bigram(SMALL)
    held_out = [["a", "c", "b"]]
    assert perplexity(model, held_out) == pytest.approx(
        2.0 ** bits_per_token(model, held_out), rel=1e-9
    )


def test_perplexity_explodes_without_smoothing_and_stays_finite_with_it():
    """Тот самый zero-count problem: несглаженная модель на held-out бесполезна."""
    ngrams, contexts = train_ngram(SMALL, n=2)
    mle = lambda context, w: raw_probability(ngrams, contexts, context, w)
    smart = kneser_ney_bigram(SMALL)
    held_out = [["b", "c"]]
    assert perplexity(mle, held_out) > 1e3
    assert perplexity(smart, held_out) < 100.0


# ----------------------------------------------------------------- generate
def test_generate_keeps_the_prefix_and_follows_the_only_likely_word():
    only_b = lambda prev, w: 1.0 if w == "b" else 0.0
    assert generate(only_b, ["a", "b", "</s>"], ["<s>"], random.Random(0), 3) == [
        "<s>",
        "b",
        "b",
        "b",
    ]


def test_generate_stops_at_the_end_token():
    only_eos = lambda prev, w: 1.0 if w == "</s>" else 0.0
    assert generate(only_eos, ["a", "</s>"], ["<s>"], random.Random(0), 99) == [
        "<s>",
        "</s>",
    ]


def test_generate_never_exceeds_max_len_steps():
    uniform = lambda prev, w: 1.0
    out = generate(uniform, ["a", "b", "c"], ["<s>"], random.Random(0), 5)
    assert len(out) == 6


def test_generate_is_reproducible_for_the_same_seed():
    """rng параметром: один seed — один и тот же текст, всегда."""
    model = kneser_ney_bigram(FRANCISCO)
    vocab = ["san", "francisco", "cat", "the", "</s>"]
    a = generate(model, vocab, ["<s>"], random.Random(3), 20)
    b = generate(model, vocab, ["<s>"], random.Random(3), 20)
    assert a == b


def test_generate_differs_for_a_different_seed():
    uniform = lambda prev, w: 1.0
    vocab = ["a", "b", "c"]
    a = generate(uniform, vocab, ["<s>"], random.Random(1), 20)
    b = generate(uniform, vocab, ["<s>"], random.Random(2), 20)
    assert a != b


def test_generate_uses_the_trained_trigram_context_width():
    ngrams, contexts = train_ngram([["a", "b"]], n=3)
    model = lambda context, word: raw_probability(ngrams, contexts, context, word)
    assert generate(
        model,
        ["a", "b", "</s>"],
        ["<s>"],
        random.Random(0),
        max_len=3,
        n=3,
    ) == ["<s>", "a", "b", "</s>"]
