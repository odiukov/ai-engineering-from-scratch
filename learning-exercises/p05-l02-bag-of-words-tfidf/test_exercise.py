"""Тесты к уроку «Мешок слов, TF-IDF и представление текста». Правь exercise.py."""

import math

import pytest

from exercise import (
    bag_of_words,
    build_vocab,
    cosine_similarity,
    document_frequency,
    inverse_document_frequency,
    l2_normalize,
    term_frequency,
    tfidf,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(matrix):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in matrix for x in row]


DOCS = [["the", "cat", "sat"], ["the", "dog", "sat"], ["the", "cat", "ran"]]


# ------------------------------------------------------------ build_vocab
def test_vocab_numbers_words_in_order_of_first_appearance():
    assert build_vocab([["cat", "sat"], ["cat", "ran"]]) == {"cat": 0, "sat": 1, "ran": 2}


def test_vocab_has_one_entry_per_unique_word():
    vocab = build_vocab(DOCS)
    assert len(vocab) == len({t for doc in DOCS for t in doc})


def test_vocab_indices_are_a_dense_range():
    """Индексы обязаны покрыть 0..n-1 без дыр — иначе матрица развалится."""
    vocab = build_vocab(DOCS)
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_vocab_of_no_documents_is_empty():
    assert build_vocab([]) == {}


# ----------------------------------------------------------- bag_of_words
def test_bag_of_words_counts_repeats():
    docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
    vocab = build_vocab(docs)
    assert flat(bag_of_words(docs, vocab)) == flat([[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]])


def test_bag_of_words_forgets_word_order():
    """"dog bites man" и "man bites dog" неотличимы — цена и суть модели."""
    docs = [["dog", "bites", "man"], ["man", "bites", "dog"]]
    vocab = build_vocab(docs)
    rows = bag_of_words(docs, vocab)
    assert rows[0] == rows[1]


def test_bag_of_words_skips_out_of_vocabulary_tokens():
    """Слово, которого не было при обучении, просто теряется."""
    vocab = {"cat": 0}
    assert bag_of_words([["cat", "zoomerapproved"]], vocab) == [[1]]


def test_bag_of_words_rows_are_independent_objects():
    """Ловушка [[0] * n] * m: строки оказались бы одним и тем же списком."""
    rows = bag_of_words([["a"], ["b"]], {"a": 0, "b": 1})
    assert rows[0] is not rows[1]
    assert flat(rows) == [1, 0, 0, 1]


def test_row_sum_equals_number_of_known_tokens():
    vocab = build_vocab(DOCS)
    rows = bag_of_words(DOCS, vocab)
    assert [sum(r) for r in rows] == [len(d) for d in DOCS]


# --------------------------------------------------------- term_frequency
def test_term_frequency_divides_by_document_length():
    assert term_frequency([1, 1, 2], 4) == APPROX([0.25, 0.25, 0.5])


def test_term_frequency_of_an_empty_document_does_not_divide_by_zero():
    assert term_frequency([0, 0], 0) == APPROX([0.0, 0.0])


def test_term_frequencies_sum_to_one():
    row = [3, 1, 2]
    assert sum(term_frequency(row, sum(row))) == APPROX(1.0)


def test_term_frequency_is_scale_free():
    """Документ, повторённый дважды, даёт те же частоты."""
    assert term_frequency([2, 4], 6) == APPROX(term_frequency([1, 2], 3))


# ----------------------------------------------------- document_frequency
def test_document_frequency_counts_documents_not_occurrences():
    """Пять вхождений в одном документе — это df = 1, а не 5."""
    assert document_frequency([[5, 0], [1, 1]]) == [2, 1]


def test_document_frequency_of_a_word_nobody_used_is_zero():
    assert document_frequency([[1, 0], [1, 0]]) == [2, 0]


def test_document_frequency_never_exceeds_the_number_of_documents():
    df = document_frequency(bag_of_words(DOCS, build_vocab(DOCS)))
    assert max(df) <= len(DOCS)


def test_document_frequency_of_an_empty_corpus_is_empty():
    assert document_frequency([]) == []


# -------------------------------------------- inverse_document_frequency
def test_idf_of_a_ubiquitous_word_is_one_not_zero():
    """Сглаженная формула не даёт слову исчезнуть: log(N+1 / N+1) + 1 = 1."""
    assert inverse_document_frequency([3], 3) == APPROX([1.0])


def test_idf_falls_as_the_word_gets_more_common():
    idf = inverse_document_frequency([1, 2, 3], 3)
    assert idf[0] > idf[1] > idf[2]


def test_idf_of_an_unseen_word_is_finite():
    """Без сглаживания здесь было бы деление на ноль."""
    value = inverse_document_frequency([0], 3)[0]
    assert math.isfinite(value) and value > 1.0


def test_idf_matches_the_smoothed_formula():
    assert inverse_document_frequency([1], 2) == APPROX([math.log(3 / 2) + 1])


# ------------------------------------------------------------------ tfidf
def test_tfidf_worked_example():
    assert flat(tfidf([[1, 1], [1, 0]])) == pytest.approx(
        flat([[0.5, 0.5 * (math.log(1.5) + 1)], [1.0, 0.0]])
    )


def test_tfidf_downweights_the_word_that_is_everywhere():
    """"the" есть во всех трёх документах, "cat" — в двух. Вес "the" ниже."""
    vocab = build_vocab(DOCS)
    weights = tfidf(bag_of_words(DOCS, vocab))
    assert weights[0][vocab["the"]] < weights[0][vocab["cat"]]


def test_tfidf_keeps_zeros_zero():
    """Слова нет в документе — вес ровно ноль, каким бы ни был idf."""
    vocab = build_vocab(DOCS)
    weights = tfidf(bag_of_words(DOCS, vocab))
    assert weights[0][vocab["dog"]] == APPROX(0.0)


def test_tfidf_shape_matches_the_input():
    matrix = bag_of_words(DOCS, build_vocab(DOCS))
    weights = tfidf(matrix)
    assert [len(r) for r in weights] == [len(r) for r in matrix]


def test_tfidf_of_an_empty_document_is_all_zeros():
    assert flat(tfidf([[1, 1], [0, 0]]))[2:] == APPROX([0.0, 0.0])


# ----------------------------------------------------------- l2_normalize
def test_l2_normalize_makes_rows_unit_length():
    assert flat(l2_normalize([[3.0, 4.0]])) == APPROX([0.6, 0.8])


def test_every_normalized_row_has_norm_one():
    rows = l2_normalize(tfidf(bag_of_words(DOCS, build_vocab(DOCS))))
    for row in rows:
        assert math.sqrt(sum(x * x for x in row)) == pytest.approx(1.0, abs=1e-9)


def test_l2_normalize_leaves_a_zero_row_alone():
    """Нулевую строку нормировать не на что — но и падать нельзя."""
    assert flat(l2_normalize([[0.0, 0.0]])) == APPROX([0.0, 0.0])


def test_l2_normalize_does_not_change_direction():
    """Нормировка меняет длину, но не пропорции между координатами."""
    row = l2_normalize([[1.0, 2.0]])[0]
    assert row[1] / row[0] == APPROX(2.0)


# ------------------------------------------------------- cosine_similarity
def test_identical_documents_score_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == APPROX(1.0)


def test_disjoint_vocabularies_score_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_ignores_vector_length():
    """Главное свойство: удвоенный документ остаётся тем же документом."""
    assert cosine_similarity([1, 1], [2, 2]) == APPROX(1.0)
    assert cosine_similarity([1, 3], [7, 21]) == APPROX(1.0)


def test_cosine_of_a_zero_vector_is_zero_not_a_crash():
    assert cosine_similarity([0, 0], [1, 1]) == APPROX(0.0)


def test_cosine_is_symmetric():
    a, b = [1.0, 2.0, 0.5], [0.3, 1.0, 4.0]
    assert cosine_similarity(a, b) == APPROX(cosine_similarity(b, a))


def test_on_normalized_rows_cosine_equals_the_dot_product():
    """Ради этого и делают L2-нормировку перед поиском по эмбеддингам."""
    rows = l2_normalize(tfidf(bag_of_words(DOCS, build_vocab(DOCS))))
    dot = sum(x * y for x, y in zip(rows[0], rows[1]))
    assert cosine_similarity(rows[0], rows[1]) == APPROX(dot)
