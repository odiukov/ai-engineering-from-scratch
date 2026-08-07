"""Тесты к уроку «RAG: чанки, TF-IDF, поиск, промпт». Правь exercise.py."""

import pytest

from exercise import (
    RAG_INSTRUCTION,
    build_rag_prompt,
    build_vocabulary,
    chunk_text,
    compute_idf,
    cosine_similarity,
    recall_at_k,
    search,
    tfidf_embed,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

DOCS = [
    "enterprise customers get a sixty day refund window",
    "starter plans have no refund after fourteen days",
    "the office coffee machine is on the second floor",
]


def _index(documents):
    """Мини-индекс: словарь, idf и векторы корпуса. Нужен нескольким тестам."""
    vocab = build_vocabulary(documents)
    idf = compute_idf(documents, vocab)
    return vocab, idf, [tfidf_embed(d, vocab, idf) for d in documents]


# ------------------------------------------------------------------ chunk_text
def test_chunk_text_splits_with_overlap():
    assert chunk_text("a b c d e", chunk_size=3, overlap=1) == ["a b c", "c d e"]


def test_chunk_text_of_empty_input_is_empty():
    assert chunk_text("", chunk_size=3, overlap=1) == []


def test_chunk_text_keeps_neighbouring_words_together_somewhere():
    """Смысл перекрытия: фраза на границе целиком лежит хотя бы в одном чанке."""
    words = [f"w{i}" for i in range(10)]
    chunks = chunk_text(" ".join(words), chunk_size=4, overlap=2)
    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i + 1]}"
        assert any(pair in chunk for chunk in chunks)


def test_chunk_text_without_overlap_cuts_phrases_on_the_boundary():
    """Контраст к предыдущему тесту: без перекрытия граница рвёт фразу."""
    words = [f"w{i}" for i in range(10)]
    chunks = chunk_text(" ".join(words), chunk_size=4, overlap=0)
    assert not any("w3 w4" in chunk for chunk in chunks)


def test_chunk_text_rejects_overlap_that_would_loop_forever():
    """Ловушка: overlap >= chunk_size даёт нулевой шаг и вечный цикл."""
    with pytest.raises(ValueError):
        chunk_text("a b c d", chunk_size=3, overlap=3)


# ------------------------------------------------------------ build_vocabulary
def test_build_vocabulary_is_sorted_and_lowercase():
    assert build_vocabulary(["Cat sat", "cat ran"]) == ["cat", "ran", "sat"]


def test_build_vocabulary_deduplicates_across_documents():
    assert build_vocabulary(["cat cat", "cat"]) == ["cat"]


def test_build_vocabulary_order_is_stable_across_calls():
    """Позиция слова — номер координаты вектора: разъедется порядок, разъедется поиск."""
    assert build_vocabulary(DOCS) == build_vocabulary(list(reversed(DOCS)))


# ------------------------------------------------------------------ compute_idf
def test_compute_idf_has_one_weight_per_vocabulary_entry():
    vocab = build_vocabulary(DOCS)
    assert len(compute_idf(DOCS, vocab)) == len(vocab)


def test_compute_idf_of_a_word_in_every_document_is_one():
    assert compute_idf(["cat dog", "cat bird"], ["cat"]) == APPROX([1.0])


def test_compute_idf_penalises_the_more_frequent_word():
    """Слово из всех документов не различает их и весит меньше редкого."""
    common, rare = compute_idf(["cat dog", "cat bird"], ["cat", "dog"])
    assert common < rare


def test_compute_idf_never_zeroes_a_word_out():
    """Сглаживание единицами: без него слово из всех документов выпало бы из вектора."""
    assert all(w > 0 for w in compute_idf(DOCS, build_vocabulary(DOCS)))


# ------------------------------------------------------------------ tfidf_embed
def test_tfidf_embed_weights_by_term_frequency():
    assert tfidf_embed("cat cat", ["cat", "dog"], [1.0, 1.0]) == APPROX([1.0, 0.0])


def test_tfidf_embed_of_empty_text_is_a_zero_vector():
    assert tfidf_embed("", ["cat", "dog"], [1.0, 1.0]) == APPROX([0.0, 0.0])


def test_tfidf_embed_ignores_words_outside_the_vocabulary():
    assert tfidf_embed("cat zebra", ["cat"], [1.0]) == APPROX([0.5])


def test_tfidf_embed_dimension_always_matches_the_vocabulary():
    vocab, idf, _ = _index(DOCS)
    assert len(tfidf_embed("what is the refund policy", vocab, idf)) == len(vocab)


# ------------------------------------------------------------ cosine_similarity
def test_cosine_of_identical_directions_is_one():
    assert cosine_similarity([1, 0], [1, 0]) == APPROX(1.0)


def test_cosine_of_disjoint_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_does_not_depend_on_vector_length():
    """Короткий вопрос сравним с длинным чанком — ради этого косинус и берут."""
    assert cosine_similarity([1, 0], [5, 0]) == APPROX(1.0)


def test_cosine_with_a_zero_vector_is_zero_not_an_error():
    """Запрос из слов, которых нет в словаре, даёт ровно нулевой вектор."""
    assert cosine_similarity([0, 0], [1, 1]) == APPROX(0.0)


# ----------------------------------------------------------------------- search
def test_search_returns_at_most_top_k():
    assert len(search([1, 0], [[1, 0], [0, 1], [1, 1]], top_k=2)) == 2


def test_search_ranks_the_exact_match_first():
    assert search([1, 0], [[0, 1], [1, 0]], top_k=1) == [(1, APPROX(1.0))]


def test_search_breaks_ties_by_index():
    hits = search([1, 0], [[1, 0], [1, 0], [0, 1]], top_k=2)
    assert [i for i, _ in hits] == [0, 1]


def test_search_finds_the_chunk_that_answers_the_question():
    """Сквозная проверка конвейера: словарь, idf, векторы, выдача."""
    vocab, idf, embeddings = _index(DOCS)
    query = tfidf_embed("enterprise refund window", vocab, idf)
    assert search(query, embeddings, top_k=1)[0][0] == 0


# --------------------------------------------------------------- build_rag_prompt
def test_build_rag_prompt_numbers_the_sources():
    prompt = build_rag_prompt("q", ["first chunk", "second chunk"])
    assert "[Source 1]" in prompt and "[Source 2]" in prompt


def test_build_rag_prompt_contains_the_question_and_the_instruction():
    prompt = build_rag_prompt("What is the refund policy?", ["text"])
    assert RAG_INSTRUCTION in prompt
    assert "Question: What is the refund policy?" in prompt


def test_build_rag_prompt_ends_by_inviting_the_answer():
    assert build_rag_prompt("q", ["text"]).endswith("Answer:")


def test_build_rag_prompt_survives_an_empty_retrieval():
    """Ретривер ничего не нашёл — промпт всё равно валиден, отказ разрешён инструкцией."""
    prompt = build_rag_prompt("q", [])
    assert "[Source 1]" not in prompt
    assert "Question: q" in prompt


# ------------------------------------------------------------------ recall_at_k
def test_recall_at_k_counts_the_share_of_relevant_documents_found():
    assert recall_at_k([3, 1, 7], [1, 9], 3) == APPROX(0.5)


def test_recall_at_k_ignores_documents_below_the_cutoff():
    assert recall_at_k([3, 1, 7], [1, 9], 1) == APPROX(0.0)


def test_recall_at_k_never_decreases_as_k_grows():
    """Если recall@10 меньше recall@5 — ошибка в реализации, а не в данных."""
    retrieved = [5, 2, 9, 1, 4, 8, 3]
    relevant = [1, 3, 5]
    values = [recall_at_k(retrieved, relevant, k) for k in range(1, 8)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_recall_at_k_reaches_one_when_k_covers_everything():
    assert recall_at_k([5, 2, 9, 1], [1, 5], 4) == APPROX(1.0)


def test_recall_at_k_without_relevant_documents_is_one():
    assert recall_at_k([1], [], 1) == APPROX(1.0)
