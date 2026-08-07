"""Тесты к уроку «Информационный поиск». Правь exercise.py."""

import math

import pytest

from exercise import (
    bm25_idf,
    bm25_rank,
    bm25_score,
    build_bm25_index,
    dense_rank,
    evaluate_rankings,
    reciprocal_rank_fusion,
    tokenize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CORPUS = [
    "Section 420 IPC covers cheating and dishonest inducement.",
    "The penalty for fraud is imprisonment up to seven years.",
    "Deceiving a person to obtain property is a criminal offence.",
    "The cat sat on the mat in the sunny garden.",
    "Bus timetables for the city centre change every summer.",
]

# «эмбеддинги» руками: первая координата — юридическая тема, вторая —
# бытовая, третья — транспорт
DOC_VECTORS = [
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.8, 0.2, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]


# ---------------------------------------------------------------- tokenize
def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Section 420 IPC!") == ["section", "420", "ipc"]


def test_tokenize_keeps_digits_as_tokens():
    """Номера статей и коды ошибок — ровно то, ради чего держат BM25."""
    assert "420" in tokenize("see section 420 today")


def test_tokenize_of_pure_punctuation_is_empty():
    assert tokenize("--- ??? ...") == []


# -------------------------------------------------------- build_bm25_index
def test_index_counts_the_documents():
    assert build_bm25_index(CORPUS)["n_docs"] == len(CORPUS)


def test_index_average_document_length_is_in_tokens():
    index = build_bm25_index(["a b", "c d e f"])
    assert index["avg_dl"] == APPROX(3.0)


def test_index_document_frequency_counts_documents_not_occurrences():
    """Слово, десять раз повторённое в одном документе, добавляет к df единицу."""
    index = build_bm25_index(["fraud fraud fraud fraud", "nothing here"])
    assert index["df"]["fraud"] == 1


def test_index_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        build_bm25_index([])


# ---------------------------------------------------------------- bm25_idf
def test_idf_is_higher_for_a_rarer_term():
    index = build_bm25_index(["fraud rare", "fraud common", "fraud words"])
    assert bm25_idf(index, "rare") > bm25_idf(index, "fraud")


def test_idf_of_a_term_in_every_document_stays_positive():
    """В этом варианте формулы idf никогда не уходит в минус — единица спасает."""
    index = build_bm25_index(["fraud a", "fraud b", "fraud c"])
    assert 0.0 < bm25_idf(index, "fraud") < 0.5


def test_idf_of_an_unknown_term_is_the_largest():
    index = build_bm25_index(CORPUS)
    unknown = bm25_idf(index, "zzzqqq")
    assert unknown > max(bm25_idf(index, t) for t in ("fraud", "the", "cat"))


def test_idf_decreases_as_document_frequency_grows():
    index = build_bm25_index(["x a", "x b", "y c", "z d"])
    assert bm25_idf(index, "x") < bm25_idf(index, "y")


# -------------------------------------------------------------- bm25_score
def test_score_is_zero_when_no_query_term_appears():
    index = build_bm25_index(CORPUS)
    assert bm25_score(index, "zzzqqq wwwvvv", 0) == 0.0


def test_score_grows_with_term_frequency():
    index = build_bm25_index(["fraud x x x x x x x x x", "fraud " * 10])
    assert bm25_score(index, "fraud", 1) > bm25_score(index, "fraud", 0)


def test_score_saturates_instead_of_growing_linearly():
    """Главное отличие от tf-idf: десять вхождений не дают десятикратный вклад."""
    index = build_bm25_index(["fraud x x x x x x x x x", "fraud " * 10])
    once = bm25_score(index, "fraud", 0)
    ten_times = bm25_score(index, "fraud", 1)
    assert ten_times < 3 * once
    # верхняя граница вклада термина — idf * (k1 + 1), у tf-idf её нет
    assert ten_times < bm25_idf(index, "fraud") * (index["k1"] + 1)


def test_k1_zero_turns_bm25_into_a_binary_match():
    index = build_bm25_index(["fraud x x x x x x x x x", "fraud " * 10], k1=0.0)
    assert bm25_score(index, "fraud", 0) == APPROX(bm25_score(index, "fraud", 1))


def test_b_controls_length_normalisation():
    docs = ["fraud aaa", "fraud aaa bbb ccc ddd eee"]
    normalised = build_bm25_index(docs, b=0.75)
    assert bm25_score(normalised, "fraud", 0) > bm25_score(normalised, "fraud", 1)
    plain = build_bm25_index(docs, b=0.0)
    assert bm25_score(plain, "fraud", 0) == APPROX(bm25_score(plain, "fraud", 1))


# --------------------------------------------------------------- bm25_rank
def test_rank_puts_the_lexically_matching_document_first():
    index = build_bm25_index(CORPUS)
    assert bm25_rank(index, "section 420 IPC")[0][1] == 0


def test_rank_returns_nothing_when_the_query_shares_no_vocabulary():
    """Хрупкость BM25: перефразированный запрос даёт пустую выдачу."""
    index = build_bm25_index(CORPUS)
    assert bm25_rank(index, "zzzqqq wwwvvv") == []


def test_rank_respects_top_k():
    index = build_bm25_index(CORPUS)
    assert len(bm25_rank(index, "the", top_k=2)) == 2


def test_rank_breaks_ties_towards_the_smaller_index():
    index = build_bm25_index(["fraud here", "fraud here"])
    assert [doc_idx for _, doc_idx in bm25_rank(index, "fraud")] == [0, 1]


# -------------------------------------------------------------- dense_rank
def test_dense_rank_orders_by_cosine_similarity():
    ranking = dense_rank([0.85, 0.15, 0.0], DOC_VECTORS)
    assert [doc_idx for _, doc_idx in ranking[:3]] == [1, 2, 0]
    assert ranking[-1][1] == 4


def test_dense_rank_ignores_vector_length():
    """Косинус, а не скалярное произведение: длинный вектор не выигрывает даром."""
    ranking = dense_rank([1.0, 0.0], [[1.0, 0.0], [50.0, 0.0]])
    assert ranking[0][0] == APPROX(1.0)
    assert ranking[1][0] == APPROX(1.0)


def test_dense_rank_gives_orthogonal_documents_zero():
    ranking = dense_rank([1.0, 0.0], [[0.0, 1.0]])
    assert ranking[0][0] == APPROX(0.0)


def test_dense_rank_survives_a_zero_vector():
    """Ловушка: нулевая норма — это деление на ноль, а не близость."""
    assert dense_rank([1.0, 0.0], [[0.0, 0.0]])[0][0] == 0.0


# ---------------------------------------------------- reciprocal_rank_fusion
def test_rrf_score_is_the_sum_of_reciprocal_ranks():
    fused = reciprocal_rank_fusion([[(9.0, 7)], [(0.2, 7)]], k=60)
    assert fused == [(APPROX(2.0 / 61), 7)]


def test_rrf_ignores_the_raw_scores_entirely():
    """BM25 выдаёт единицы, косинус — доли: складывать можно только ранги."""
    big = reciprocal_rank_fusion([[(1000.0, 5), (999.0, 3)]])
    small = reciprocal_rank_fusion([[(0.001, 5), (0.0009, 3)]])
    assert [d for _, d in big] == [d for _, d in small]
    assert [s for s, _ in big] == pytest.approx([s for s, _ in small])


def test_rrf_prefers_a_document_found_by_both_retrievers():
    sparse = [(5.0, 0), (4.0, 2)]
    dense = [(0.9, 2), (0.8, 1)]
    fused = reciprocal_rank_fusion([sparse, dense])
    assert fused[0][1] == 2


def test_rrf_of_a_single_ranking_keeps_its_order():
    ranking = [(5.0, 4), (4.0, 1), (3.0, 0)]
    fused = reciprocal_rank_fusion([ranking])
    assert [d for _, d in fused] == [4, 1, 0]


# --------------------------------------------------------- evaluate_rankings
def test_metrics_of_a_perfect_retriever_are_one():
    rankings = [[(1.0, 3), (0.5, 1)], [(1.0, 7), (0.5, 2)]]
    assert evaluate_rankings(rankings, [3, 7]) == APPROX((1.0, 1.0))


def test_mrr_halves_when_the_answer_slips_to_second_place():
    assert evaluate_rankings([[(1.0, 9), (0.5, 3)]], [3])[1] == APPROX(0.5)


def test_a_query_whose_answer_is_missing_contributes_nothing():
    rankings = [[(1.0, 3)], [(1.0, 9)]]
    recall, mrr = evaluate_rankings(rankings, [3, 42])
    assert recall == APPROX(0.5)
    assert mrr == APPROX(0.5)


def test_recall_depends_on_the_cutoff_but_mrr_does_not():
    rankings = [[(3.0, 0), (2.0, 1), (1.0, 5)]]
    assert evaluate_rankings(rankings, [5], k=1)[0] == 0.0
    assert evaluate_rankings(rankings, [5], k=3)[0] == 1.0
    assert evaluate_rankings(rankings, [5], k=1)[1] == APPROX(1 / 3)


def test_metrics_reject_a_mismatched_number_of_answers():
    with pytest.raises(ValueError):
        evaluate_rankings([[(1.0, 0)]], [0, 1])


# ------------------------------------------- гибрид: сумма больше слагаемых
def test_hybrid_beats_either_retriever_alone_on_a_paraphrased_query():
    """Запрос без общих слов: BM25 нем, плотный поиск отвечает, RRF собирает."""
    index = build_bm25_index(CORPUS)
    query = "swindling someone out of belongings"
    sparse = bm25_rank(index, query)
    dense = dense_rank([0.8, 0.2, 0.0], DOC_VECTORS, top_k=3)
    assert sparse == []
    fused = reciprocal_rank_fusion([sparse, dense])
    assert fused[0][1] == 2
    assert evaluate_rankings([fused], [2], k=1)[0] == 1.0
    assert evaluate_rankings([sparse], [2], k=1)[0] == 0.0


def test_hybrid_keeps_the_exact_match_that_dense_search_buries():
    """Обратный случай: код статьи находит только BM25, и слияние его удержит."""
    index = build_bm25_index(CORPUS)
    sparse = bm25_rank(index, "420 IPC", top_k=3)
    dense = dense_rank([0.0, 0.0, 1.0], DOC_VECTORS, top_k=3)
    assert sparse[0][1] == 0
    assert dense[0][1] == 4
    fused = reciprocal_rank_fusion([sparse, dense])
    assert 0 in [doc_idx for _, doc_idx in fused[:2]]
