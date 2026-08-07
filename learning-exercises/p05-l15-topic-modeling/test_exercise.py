"""Тесты к уроку «Тематическое моделирование: LDA и BERTopic». Правь exercise.py."""

import random

import pytest

from exercise import (
    build_corpus,
    class_based_tfidf,
    count_tables,
    fit_lda,
    top_words,
    topic_coherence_npmi,
    topic_conditional,
    topic_diversity,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in matrix for x in row]


# два тематически разъехавшихся блока: на них проверяется весь LDA
PETS = "cat dog pet cat dog pet"
MONEY = "stock bond market stock bond market"
TWO_TOPIC_CORPUS = [PETS] * 6 + [MONEY] * 6


# ------------------------------------------------------------- build_corpus
def test_build_corpus_lowercases_and_sorts_the_vocabulary():
    vocab, doc_ids = build_corpus(["Cat dog", "cat sky"])
    assert vocab == ["cat", "dog", "sky"]
    assert doc_ids == [[0, 1], [0, 2]]


def test_build_corpus_min_df_counts_documents_not_occurrences():
    """Слово, повторённое в одном документе, имеет document frequency 1."""
    vocab, _ = build_corpus(["cat cat cat cat", "dog dog"], min_df=2)
    assert vocab == []


def test_build_corpus_max_df_drops_a_ubiquitous_word():
    docs = ["the cat", "the dog", "the sky", "the sea"]
    vocab, _ = build_corpus(docs, max_df=0.9)
    assert "the" not in vocab
    assert vocab == ["cat", "dog", "sea", "sky"]


def test_build_corpus_removes_stopwords_before_indexing():
    vocab, doc_ids = build_corpus(["cat dog", "cat sky"], stopwords=["cat"])
    assert vocab == ["dog", "sky"]
    assert doc_ids == [[0], [1]]


def test_build_corpus_keeps_repeated_tokens_in_the_id_sequence():
    """Гиббсу нужен каждый токен отдельно, поэтому повторы не схлопываются."""
    _, doc_ids = build_corpus(["cat cat cat"])
    assert doc_ids == [[0, 0, 0]]


# -------------------------------------------------------------- count_tables
def test_count_tables_counts_tokens_per_document_and_per_topic():
    doc_topic, topic_word = count_tables([[0, 1]], [[0, 0]], 2, 2)
    assert doc_topic == [[2, 0]]
    assert topic_word == [[1, 1], [0, 0]]


def test_count_tables_splits_the_same_word_between_topics():
    doc_topic, topic_word = count_tables([[0], [0]], [[0], [1]], 2, 1)
    assert doc_topic == [[1, 0], [0, 1]]
    assert topic_word == [[1], [1]]


def test_count_tables_both_tables_sum_to_the_token_count():
    doc_ids = [[0, 1, 2], [2, 2]]
    assignments = [[0, 1, 0], [1, 1]]
    doc_topic, topic_word = count_tables(doc_ids, assignments, 2, 3)
    n_tokens = sum(len(d) for d in doc_ids)
    assert sum(flat(doc_topic)) == n_tokens
    assert sum(flat(topic_word)) == n_tokens


def test_count_tables_keeps_shape_for_empty_documents():
    doc_topic, topic_word = count_tables([[], []], [[], []], 3, 4)
    assert doc_topic == [[0, 0, 0], [0, 0, 0]]
    assert topic_word == [[0, 0, 0, 0]] * 3


# --------------------------------------------------------- topic_conditional
def test_topic_conditional_sums_to_one():
    probs = topic_conditional([3, 1, 0], [2, 0, 5], [7, 3, 9], 0.1, 0.01, 20)
    assert sum(probs) == APPROX(1.0)


def test_topic_conditional_is_uniform_on_empty_counts():
    assert topic_conditional([0, 0], [0, 0], [0, 0], 1.0, 1.0, 2) == APPROX([0.5, 0.5])


def test_topic_conditional_prefers_the_topic_that_dominates_the_document():
    """Левый множитель: тема, уже занявшая документ, тянет токен к себе."""
    probs = topic_conditional([9, 0], [0, 0], [9, 0], 1.0, 1.0, 2)
    assert probs[0] > probs[1]


def test_topic_conditional_prefers_the_topic_that_already_owns_the_word():
    """Правый множитель: тема, уже любящая это слово, тянет токен к себе."""
    probs = topic_conditional([1, 1], [9, 0], [9, 0], 1.0, 1.0, 2)
    assert probs[0] > probs[1]


def test_topic_conditional_never_gives_zero_thanks_to_alpha():
    """Ловушка: при alpha=0 покинутая тема получила бы 0 и не вернулась бы."""
    probs = topic_conditional([50, 0], [50, 0], [50, 0], 0.1, 0.01, 10)
    assert probs[1] > 0.0


# ---------------------------------------------------------------- fit_lda
def test_fit_lda_rows_of_both_matrices_are_distributions():
    vocab, doc_ids = build_corpus(TWO_TOPIC_CORPUS)
    doc_topic, topic_word = fit_lda(doc_ids, 2, len(vocab), random.Random(0), n_iter=20)
    for row in doc_topic:
        assert sum(row) == APPROX(1.0)
    for row in topic_word:
        assert sum(row) == APPROX(1.0)
    assert all(0.0 <= x <= 1.0 for x in flat(doc_topic) + flat(topic_word))


def test_fit_lda_is_reproducible_for_the_same_seed():
    """rng параметром, глобального random нет: два прогона совпадают побитово."""
    vocab, doc_ids = build_corpus(TWO_TOPIC_CORPUS)
    a = fit_lda(doc_ids, 2, len(vocab), random.Random(7), n_iter=20)
    b = fit_lda(doc_ids, 2, len(vocab), random.Random(7), n_iter=20)
    assert flat(a[0]) == APPROX(flat(b[0]))
    assert flat(a[1]) == APPROX(flat(b[1]))


def test_fit_lda_separates_two_disjoint_vocabularies():
    """Настоящая проверка смысла: темы обязаны разъехаться по группам слов."""
    vocab, doc_ids = build_corpus(TWO_TOPIC_CORPUS)
    _, topic_word = fit_lda(doc_ids, 2, len(vocab), random.Random(0), n_iter=80)
    found = {frozenset(words) for words in top_words(topic_word, vocab, 3)}
    assert found == {
        frozenset({"cat", "dog", "pet"}),
        frozenset({"bond", "market", "stock"}),
    }


def test_fit_lda_gives_each_document_one_dominant_topic():
    vocab, doc_ids = build_corpus(TWO_TOPIC_CORPUS)
    doc_topic, _ = fit_lda(doc_ids, 2, len(vocab), random.Random(0), n_iter=80)
    assert all(max(row) > 0.9 for row in doc_topic)
    # первые шесть документов про животных, вторые шесть про деньги
    labels = [row.index(max(row)) for row in doc_topic]
    assert len(set(labels[:6])) == 1
    assert len(set(labels[6:])) == 1
    assert labels[0] != labels[6]


def test_fit_lda_topic_numbers_carry_no_meaning():
    """Ловушка урока: разные seed дают ту же разбивку под другими номерами."""
    vocab, doc_ids = build_corpus(TWO_TOPIC_CORPUS)
    _, tw_a = fit_lda(doc_ids, 2, len(vocab), random.Random(0), n_iter=80)
    _, tw_b = fit_lda(doc_ids, 2, len(vocab), random.Random(1), n_iter=80)
    as_sets = lambda tw: {frozenset(w) for w in top_words(tw, vocab, 3)}
    assert as_sets(tw_a) == as_sets(tw_b)
    assert top_words(tw_a, vocab, 3) != top_words(tw_b, vocab, 3)


# -------------------------------------------------------------- top_words
def test_top_words_orders_by_descending_weight():
    assert top_words([[0.7, 0.2, 0.1]], ["cat", "dog", "sky"], 2) == [["cat", "dog"]]


def test_top_words_breaks_ties_by_vocabulary_order():
    assert top_words([[0.5, 0.5]], ["cat", "dog"], 1) == [["cat"]]


def test_top_words_handles_every_topic_independently():
    matrix = [[0.9, 0.1], [0.1, 0.9]]
    assert top_words(matrix, ["a", "b"], 1) == [["a"], ["b"]]


# --------------------------------------------------------- topic_diversity
def test_topic_diversity_of_disjoint_topics_is_one():
    assert topic_diversity([["cat", "dog"], ["stock", "bond"]]) == APPROX(1.0)


def test_topic_diversity_of_identical_topics_is_one_half():
    assert topic_diversity([["cat", "dog"], ["cat", "dog"]]) == APPROX(0.5)


def test_topic_diversity_drops_when_a_stopword_heads_every_topic():
    """Именно так метрика ловит невычищенные стоп-слова."""
    clean = [["cat", "dog"], ["stock", "bond"]]
    dirty = [["the", "cat"], ["the", "stock"]]
    assert topic_diversity(dirty) < topic_diversity(clean)


# ---------------------------------------------------- topic_coherence_npmi
COH_DOCS = [["cat", "dog"], ["cat", "dog"], ["stock"], ["bond"]]


def test_coherence_of_always_co_occurring_words_is_one():
    assert topic_coherence_npmi([["cat", "dog"]], COH_DOCS) == APPROX(1.0)


def test_coherence_of_never_co_occurring_words_is_minus_one():
    assert topic_coherence_npmi([["cat", "stock"]], COH_DOCS) == APPROX(-1.0)


def test_coherence_of_independent_words_is_zero():
    """p(a, b) = p(a) * p(b) — независимость, NPMI ровно 0."""
    docs = [["a", "b"], ["a"], ["b"], []]
    assert topic_coherence_npmi([["a", "b"]], docs) == APPROX(0.0)


def test_coherence_of_a_word_absent_from_the_corpus_is_zero():
    assert topic_coherence_npmi([["cat", "unicorn"]], COH_DOCS) == APPROX(0.0)


def test_coherence_of_a_pair_present_everywhere_is_one():
    """Ловушка: -log p(a, b) здесь ноль, делить нельзя — соглашение даёт 1."""
    docs = [["a", "b"], ["a", "b"]]
    assert topic_coherence_npmi([["a", "b"]], docs) == APPROX(1.0)


# --------------------------------------------------------- class_based_tfidf


def test_class_based_tfidf_is_never_negative():
    """log(1 + A / f) всегда положителен — отрицательных весов не бывает."""
    matrix = class_based_tfidf([[0, 0, 1], [1, 2, 2], [0, 2]], 3)
    assert all(x >= 0.0 for x in flat(matrix))


def test_class_based_tfidf_ranks_the_cluster_specific_word_first():
    """Слово 0 только у кластера 0, слово 1 есть у всех — вот и вся c-TF-IDF."""
    matrix = class_based_tfidf([[0, 1], [1, 2], [1, 3]], 4)
    assert matrix[0][0] > matrix[0][1]


def test_class_based_tfidf_gives_zero_to_a_word_the_cluster_never_uses():
    matrix = class_based_tfidf([[0, 0], [1, 1]], 2)
    assert matrix[0][1] == APPROX(0.0)
    assert matrix[1][0] == APPROX(0.0)


def test_class_based_tfidf_of_an_empty_cluster_is_all_zeros():
    matrix = class_based_tfidf([[], [0]], 1)
    assert matrix[0] == APPROX([0.0])
    assert matrix[1][0] > 0.0


def test_class_based_tfidf_feeds_top_words_like_bertopic_does():
    """Связка из урока: кластеры -> c-TF-IDF -> топ-слова кластера."""
    vocab = ["bond", "cat", "dog", "stock"]
    clusters = [[1, 1, 2, 3], [0, 0, 3, 3, 2]]
    assert top_words(class_based_tfidf(clusters, len(vocab)), vocab, 1) == [
        ["cat"],
        ["bond"],
    ]


