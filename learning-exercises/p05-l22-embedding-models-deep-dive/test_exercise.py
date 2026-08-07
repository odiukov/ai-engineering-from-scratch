"""Тесты к уроку «Embedding-модели: dense, sparse, multi-vector». Правь exercise.py."""

import math

import pytest

from exercise import (
    cosine,
    embed,
    matryoshka_truncate,
    maxsim,
    rank,
    reciprocal_rank_fusion,
    sparse_embed,
    sparse_score,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CORPUS = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]


def norm(vec):
    """Длина вектора — она понадобится в нескольких тестах."""
    return math.sqrt(sum(x * x for x in vec))


# ------------------------------------------------------------------- embed
def test_embed_returns_a_vector_of_the_requested_dim():
    assert len(embed("the cat sat", 8)) == 8
    assert len(embed("the cat sat", 256)) == 256


def test_embed_is_deterministic_across_calls():
    """Встроенный hash() солится на процесс, и такой индекс не переживёт рестарт."""
    assert embed("the cat sat on the mat", 32) == embed("the cat sat on the mat", 32)


def test_embed_averages_instead_of_summing():
    """Повтор текста не должен раздувать норму: длина документа — не смысл."""
    assert embed("cat", 8) == APPROX(embed("cat cat", 8))


def test_embed_rejects_a_text_without_tokens():
    with pytest.raises(ValueError):
        embed("!!! ???", 8)


# ------------------------------------------------------------------ cosine
def test_cosine_of_perpendicular_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == APPROX(-1.0)


def test_cosine_ignores_vector_length():
    """Ключевое свойство: масштаб эмбеддинга на ранжирование не влияет."""
    a, b = [1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]
    scaled = [100.0 * x for x in a]
    assert cosine(scaled, b) == APPROX(cosine(a, b))
    assert cosine([1.0, 0.0], [2.0, 0.0]) == APPROX(1.0)


def test_cosine_rejects_a_zero_vector():
    """Пустой документ — это ValueError, а не «близость 0.0»."""
    with pytest.raises(ValueError):
        cosine([0.0, 0.0], [1.0, 0.0])


def test_cosine_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])


# ----------------------------------------------------- matryoshka_truncate
def test_truncate_keeps_the_first_coordinates():
    assert matryoshka_truncate([3.0, 4.0, 10.0], 2) == APPROX([0.6, 0.8])


def test_truncated_vector_is_re_normalized():
    """Без перенормировки скалярное произведение перестаёт быть косинусом."""
    truncated = matryoshka_truncate(embed("apple released the ipod in 2001", 64), 16)
    assert norm(truncated) == pytest.approx(1.0, abs=1e-12)


def test_truncation_keeps_close_texts_closer_than_far_ones():
    """Смысл matryoshka: обрезанный эмбеддинг всё ещё ранжирует правильно."""
    query = matryoshka_truncate(embed("apple released the ipod", 64), 32)
    same = matryoshka_truncate(embed("apple released the ipod today", 64), 32)
    other = matryoshka_truncate(embed("android from google", 64), 32)
    assert cosine(query, same) > cosine(query, other)


def test_truncate_rejects_a_dim_larger_than_the_vector():
    with pytest.raises(ValueError):
        matryoshka_truncate([1.0, 2.0], 5)


def test_truncate_rejects_an_all_zero_head():
    """Обрезали слишком агрессивно — от эмбеддинга ничего не осталось."""
    with pytest.raises(ValueError):
        matryoshka_truncate([0.0, 0.0, 5.0], 2)


# -------------------------------------------------------------------- rank
def test_rank_orders_documents_by_similarity():
    assert rank([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]) == [
        (0, APPROX(1.0)),
        (1, APPROX(0.0)),
        (2, APPROX(-1.0)),
    ]


def test_rank_top_k_cuts_the_tail():
    assert rank([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], top_k=1) == [(0, APPROX(1.0))]


def test_rank_keeps_input_order_on_ties():
    result = rank([1.0, 0.0], [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    assert [i for i, _ in result] == [0, 1, 2]


def test_rank_is_not_fooled_by_document_length():
    """Длинный документ не выигрывает просто потому, что его вектор длиннее."""
    short = embed("android from google", 64)
    long_same = [5.0 * x for x in short]
    query = embed("android operating system from google", 64)
    result = rank(query, [long_same, short])
    assert result[0][1] == APPROX(result[1][1])


def test_rank_finds_the_matching_passage_in_a_corpus():
    docs = [embed(text, 128) for text in CORPUS]
    query = embed("When was the first iPhone launched?", 128)
    assert rank(query, docs)[0][0] == 0


# ------------------------------------------------------------ sparse_embed
def test_sparse_embed_keeps_only_the_words_of_the_text():
    assert sorted(sparse_embed("cat dog cat")) == ["cat", "dog"]


def test_sparse_embed_is_unit_length():
    weights = sparse_embed("the cat sat on the mat with the dog")
    assert math.sqrt(sum(w * w for w in weights.values())) == pytest.approx(1.0, abs=1e-12)


def test_repeated_word_weighs_more_but_sublinearly():
    """tf в лоб сделал бы десять повторов в десять раз важнее — логарифм гасит."""
    weights = sparse_embed("cat " * 10 + "dog")
    assert weights["cat"] > weights["dog"]
    assert weights["cat"] < 10 * weights["dog"]


# ------------------------------------------------------------ sparse_score
def test_identical_texts_score_one():
    a = sparse_embed("apple released the ipod in 2001")
    assert sparse_score(a, a) == pytest.approx(1.0, abs=1e-12)


def test_texts_without_shared_words_score_zero():
    assert sparse_score(sparse_embed("cat"), sparse_embed("dog")) == APPROX(0.0)


def test_sparse_score_is_symmetric():
    a = sparse_embed("iphone launched in 2007")
    b = sparse_embed("the ipod launched in 2001")
    assert sparse_score(a, b) == APPROX(sparse_score(b, a))


def test_sparse_score_catches_a_rare_term_dense_would_blur():
    """Лексическое совпадение по редкому коду — сильная сторона sparse."""
    query = sparse_embed("error e4711")
    hit = sparse_embed("the reader reports error e4711 on boot")
    miss = sparse_embed("the reader reports a boot problem")
    assert sparse_score(query, hit) > sparse_score(query, miss)


# ------------------------------------------------------------------ maxsim
def test_maxsim_takes_the_best_match_per_query_token():
    assert maxsim([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]) == APPROX(1.0)


def test_unmatched_query_token_contributes_nothing():
    assert maxsim([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]]) == APPROX(1.0)


def test_extra_document_tokens_never_lower_the_score():
    """max по документу монотонен: лишний токен может только помочь."""
    query = [[1.0, 0.0], [0.0, 1.0]]
    short_doc = [[1.0, 0.0]]
    long_doc = [[1.0, 0.0], [0.0, 1.0]]
    assert maxsim(query, long_doc) >= maxsim(query, short_doc)


def test_maxsim_is_asymmetric():
    """Сумма идёт по токенам запроса, поэтому стороны менять нельзя."""
    q = [[1.0, 0.0], [1.0, 0.0]]
    d = [[1.0, 0.0]]
    assert maxsim(q, d) == APPROX(2.0)
    assert maxsim(d, q) == APPROX(1.0)


def test_maxsim_rejects_an_empty_side():
    with pytest.raises(ValueError):
        maxsim([], [[1.0, 0.0]])


# ------------------------------------------------- reciprocal_rank_fusion
def test_rrf_sums_contributions_from_every_ranking():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "c"]], k=1)
    assert fused[0] == ("a", APPROX(1.0))
    assert dict(fused)["b"] == APPROX(1 / 3)


def test_agreement_beats_a_single_first_place():
    """Документ, второй в обеих выдачах, обгоняет первого только в одной."""
    fused = reciprocal_rank_fusion([["x", "b"], ["y", "b"]], k=1)
    assert fused[0][0] == "b"


def test_rrf_positions_start_at_one():
    """С нуля и k=0 первый документ дал бы деление на ноль."""
    assert reciprocal_rank_fusion([["b"]], k=0)[0][1] == APPROX(1.0)


def test_rrf_breaks_ties_by_first_appearance():
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=1)
    assert [doc for doc, _ in fused] == ["a", "b"]
    assert fused[0][1] == APPROX(fused[1][1])
