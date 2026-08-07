"""Тесты к уроку «Advanced RAG». Правь exercise.py."""

import pytest

from exercise import (
    STOP_WORDS,
    bm25_score,
    bm25_search,
    build_bm25_index,
    hybrid_search,
    parent_child_chunks,
    reciprocal_rank_fusion,
    rerank,
    tokenize,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

DOCS = [
    "Enterprise customers get a 60 day pro rated refund",
    "Standard plan refund window is 30 days",
    "Error e-4021 means the payment gateway timed out",
    "Steps to terminate your plan and stop billing",
    "Q3 earnings were 47.2m according to the finance report",
]


def ids(ranked):
    """Только doc_id из списка пар (doc_id, score) — скоры тут не важны."""
    return [doc_id for doc_id, _ in ranked]


# --------------------------------------------------------------- tokenize
def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Error code E-4021!") == ["error", "code", "e-4021"]


def test_tokenize_keeps_error_codes_in_one_piece():
    """Ради таких кодов keyword-поиск и держат рядом с векторным."""
    assert "e-4021" in tokenize("The log says E-4021, retry later.")


def test_tokenize_keeps_decimal_numbers_in_one_piece():
    assert tokenize("Q3 earnings: $47.2M.") == ["q3", "earnings", "47.2m"]


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("   ...   ") == []


# -------------------------------------------------------- build_bm25_index
def test_index_counts_documents_not_occurrences():
    """df — в скольких документах встретился термин, а не сколько раз всего."""
    index = build_bm25_index(["cat cat cat", "cat sat"])
    assert index["doc_freqs"]["cat"] == 2
    assert index["doc_freqs"]["sat"] == 1


def test_index_average_length_is_the_mean_of_doc_lengths():
    index = build_bm25_index(["one two three four", "five two"])
    assert index["doc_lens"] == [4, 2]
    assert index["avg_dl"] == APPROX(3.0)


def test_index_of_empty_corpus_does_not_divide_by_zero():
    index = build_bm25_index([])
    assert index["n_docs"] == 0
    assert index["avg_dl"] == APPROX(0.0)


# ------------------------------------------------------------- bm25_score
def test_bm25_score_is_zero_when_no_query_term_is_present():
    index = build_bm25_index(DOCS)
    assert bm25_score("e-4021", 0, index) == APPROX(0.0)


def test_bm25_score_of_a_matching_document_is_positive():
    index = build_bm25_index(DOCS)
    assert bm25_score("e-4021", 2, index) > 0


def test_bm25_score_prefers_the_rare_term():
    """"refund" есть в двух документах, "e-4021" — в одном: idf выше у редкого."""
    index = build_bm25_index(DOCS)
    rare = bm25_score("e-4021", 2, index)
    common = bm25_score("refund", 1, index)
    assert rare > common


def test_bm25_score_saturates_on_term_repetition():
    """Слово 50 раз не в 50 раз релевантнее, чем один раз — за это отвечает k1."""
    index = build_bm25_index(["refund " * 1 + "policy", "refund " * 50 + "policy"])
    once = bm25_score("refund", 0, index)
    fifty = bm25_score("refund", 1, index)
    assert fifty < 50 * once


def test_bm25_score_penalises_the_longer_document():
    """Одинаковый tf, но второй документ длиннее — за это отвечает b."""
    filler = " ".join(f"w{i}" for i in range(40))
    index = build_bm25_index(["refund policy", "refund policy " + filler])
    assert bm25_score("refund", 0, index) > bm25_score("refund", 1, index)


def test_bm25_score_of_a_common_term_stays_non_negative():
    """Без "+1" внутри логарифма idf уходит в минус и штрафует документ."""
    index = build_bm25_index(["the cat", "the dog", "the bird"])
    assert bm25_score("the", 0, index) > 0


def test_bm25_score_counts_a_repeated_query_term_twice():
    index = build_bm25_index(DOCS)
    once = bm25_score("refund", 0, index)
    twice = bm25_score("refund refund", 0, index)
    assert twice == APPROX(2 * once)


# ------------------------------------------------------------ bm25_search
def test_bm25_search_finds_the_exact_error_code():
    """То, ради чего BM25 и нужен: точное совпадение по коду ошибки."""
    index = build_bm25_index(DOCS)
    assert ids(bm25_search("e-4021", index)) == [2]


def test_bm25_search_misses_a_paraphrase():
    """Обратная сторона: "cancel my subscription" не найдёт "terminate your plan"."""
    index = build_bm25_index(DOCS)
    assert ids(bm25_search("cancel my subscription", index)) == []


def test_bm25_search_drops_documents_with_zero_score():
    index = build_bm25_index(DOCS)
    assert len(bm25_search("refund", index)) == 2


def test_bm25_search_respects_top_k():
    index = build_bm25_index(DOCS)
    assert len(bm25_search("refund plan", index, top_k=1)) == 1


def test_bm25_search_breaks_ties_by_doc_id():
    index = build_bm25_index(["alpha beta", "alpha beta"])
    assert ids(bm25_search("alpha", index)) == [0, 1]


# ------------------------------------------------- reciprocal_rank_fusion
def test_rrf_uses_one_based_ranks():
    assert reciprocal_rank_fusion([[(7, 0.0)]]) == [(7, APPROX(1 / 61))]


def test_rrf_lifts_the_document_found_by_both_retrievers():
    """Второе место в обоих списках бьёт первое место в одном."""
    vector = [(1, 0.99), (2, 0.10)]
    keyword = [(3, 50.0), (2, 1.0)]
    fused = reciprocal_rank_fusion([vector, keyword])
    assert fused[0][0] == 2


def test_rrf_reproduces_the_worked_example_from_the_lesson():
    """#1 и #5 дают 0.0318, #3 и #2 дают 0.0320 — второй документ впереди."""
    vector = [(10, 0.0), (91, 0.0), (30, 0.0), (92, 0.0), (93, 0.0)]
    keyword = [(94, 0.0), (30, 0.0), (95, 0.0), (96, 0.0), (10, 0.0)]
    scores = dict(reciprocal_rank_fusion([vector, keyword]))
    assert scores[10] == APPROX(1 / 61 + 1 / 65)
    assert scores[30] == APPROX(1 / 63 + 1 / 62)
    assert scores[30] > scores[10]


def test_rrf_ignores_the_raw_score_scale():
    """Складываются ранги, поэтому косинус 0.9 и BM25 900 весят одинаково."""
    small = reciprocal_rank_fusion([[(1, 0.001), (2, 0.0005)]])
    huge = reciprocal_rank_fusion([[(1, 900.0), (2, 400.0)]])
    assert small == huge


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([]) == []


# ----------------------------------------------------------- hybrid_search
def test_hybrid_search_keeps_the_keyword_only_hit():
    """Векторный поиск не видит e-4021, гибридный всё равно возвращает документ."""
    index = build_bm25_index(DOCS)
    vector = [(0, 0.71), (1, 0.65), (4, 0.60)]
    assert 2 in ids(hybrid_search("e-4021", index, vector))


def test_hybrid_search_keeps_the_vector_only_hit():
    """И наоборот: перефразировку BM25 не находит, а гибрид её сохраняет."""
    index = build_bm25_index(DOCS)
    vector = [(3, 0.88)]
    assert 3 in ids(hybrid_search("cancel my subscription", index, vector))


def test_hybrid_search_puts_the_doubly_found_document_first():
    index = build_bm25_index(DOCS)
    vector = [(0, 0.90), (1, 0.80)]
    assert ids(hybrid_search("30 days", index, vector))[0] == 1


def test_hybrid_search_respects_top_k():
    index = build_bm25_index(DOCS)
    vector = [(0, 0.9), (1, 0.8), (3, 0.7), (4, 0.6)]
    assert len(hybrid_search("refund plan", index, vector, top_k=2)) == 2


# ------------------------------------------------------------------ rerank
def test_rerank_promotes_the_chunk_with_the_actual_number():
    """Bi-encoder ставит первым "revenue strategy", реранкер — реальные цифры."""
    docs = ["Revenue strategy for the next year", "Q3 earnings were 47.2m"]
    candidates = [(0, 0.02), (1, 0.01)]
    assert ids(rerank("q3 earnings", candidates, docs))[0] == 1


def test_rerank_rewards_word_order_through_bigrams():
    docs = ["refund policy for enterprise", "policy refund enterprise for"]
    candidates = [(0, 0.0), (1, 0.0)]
    ranked = dict(rerank("refund policy", candidates, docs))
    assert ranked[0] > ranked[1]


def test_rerank_ignores_stop_words():
    """"what is the" не должно приносить очков ни одному документу."""
    docs = ["what is the refund", "refund"]
    plain = dict(rerank("refund", [(0, 0.0), (1, 0.0)], docs))
    noisy = dict(rerank("what is the refund", [(0, 0.0), (1, 0.0)], docs))
    assert plain == noisy
    assert "the" in STOP_WORDS


def test_rerank_rewards_an_early_mention():
    tail = " ".join(f"w{i}" for i in range(20))
    docs = ["refund " + tail, tail + " refund"]
    ranked = dict(rerank("refund", [(0, 0.0), (1, 0.0)], docs))
    assert ranked[0] > ranked[1]


def test_rerank_keeps_the_retrieval_signal():
    """Исходный скор входит с весом 5.0: при равном тексте выигрывает он."""
    docs = ["refund policy", "refund policy"]
    assert ids(rerank("refund", [(0, 0.1), (1, 0.9)], docs))[0] == 1


def test_rerank_of_no_candidates_is_empty():
    assert rerank("anything", [], DOCS) == []


# ---------------------------------------------------- parent_child_chunks
def test_parent_child_splits_the_worked_example():
    parents, children, mapping = parent_child_chunks("a b c d e f", 4, 2)
    assert parents == ["a b c d", "e f"]
    assert children == ["a b", "c d", "e f"]
    assert mapping == {0: 0, 1: 0, 2: 1}


def test_every_child_belongs_to_exactly_one_parent():
    """Ловушка: ребёнок, вылезший за границу родителя, принадлежал бы двум."""
    text = " ".join(f"w{i}" for i in range(37))
    parents, children, mapping = parent_child_chunks(text, 10, 4)
    assert set(mapping) == set(range(len(children)))
    for child_idx, parent_idx in mapping.items():
        assert children[child_idx] in parents[parent_idx]


def test_children_of_a_parent_reconstruct_it():
    text = " ".join(f"w{i}" for i in range(37))
    parents, children, mapping = parent_child_chunks(text, 10, 4)
    for parent_idx, parent in enumerate(parents):
        own = [children[c] for c in sorted(mapping) if mapping[c] == parent_idx]
        assert " ".join(own) == parent


def test_parent_chunk_is_never_shorter_than_its_child():
    """Смысл приёма: искать мелким, отдавать в промпт крупное."""
    text = " ".join(f"w{i}" for i in range(100))
    parents, children, mapping = parent_child_chunks(text, 20, 5)
    for child_idx, parent_idx in mapping.items():
        assert len(children[child_idx].split()) <= len(parents[parent_idx].split())


def test_parent_child_of_empty_text_is_empty():
    assert parent_child_chunks("", 10, 4) == ([], [], {})
