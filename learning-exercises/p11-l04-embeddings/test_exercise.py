"""Тесты к уроку «Эмбеддинги: похожесть, индекс, сжатие». Правь exercise.py."""

import pytest

from exercise import (
    METRICS,
    binary_quantize,
    cosine_similarity,
    dot,
    euclidean_distance,
    norm,
    normalize,
    search,
    truncate_embedding,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Эмбеддинги заданы явными числами, а не сгенерированы: тест обязан
# воспроизводиться при любом состоянии генератора случайных чисел.
PAYMENT_FAILED = [0.9, 0.1, 0.0, 0.2]
CHARGE_DECLINED = [0.8, 0.2, 0.1, 0.1]
PAYMENT_ON_TIME = [0.1, 0.9, 0.2, 0.0]
CORPUS = [PAYMENT_FAILED, PAYMENT_ON_TIME, CHARGE_DECLINED]


# ------------------------------------------------------------------------ dot
def test_dot_multiplies_and_sums():
    assert dot([1, 2, 3], [4, 5, 6]) == APPROX(32.0)


def test_dot_of_perpendicular_vectors_is_zero():
    assert dot([1, 0], [0, 1]) == APPROX(0.0)


def test_dot_rejects_a_dimension_mismatch():
    """Молчаливый zip обрезал бы длинный вектор и дал правдоподобную чушь."""
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])


# ----------------------------------------------------------------------- norm
def test_norm_of_a_three_four_vector_is_five():
    assert norm([3, 4]) == APPROX(5.0)


def test_norm_of_a_zero_vector_is_zero():
    assert norm([0, 0, 0]) == APPROX(0.0)


# ---------------------------------------------------------- cosine_similarity
def test_cosine_of_identical_directions_is_one():
    assert cosine_similarity([1, 0], [1, 0]) == APPROX(1.0)


def test_cosine_of_perpendicular_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == APPROX(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1, 0], [-1, 0]) == APPROX(-1.0)


def test_cosine_does_not_depend_on_vector_length():
    """Главное свойство метрики: запрос из трёх слов сравним с документом из пятисот."""
    long_doc = [10 * x for x in CHARGE_DECLINED]
    assert cosine_similarity(PAYMENT_FAILED, long_doc) == APPROX(
        cosine_similarity(PAYMENT_FAILED, CHARGE_DECLINED)
    )


def test_cosine_with_a_zero_vector_is_zero_not_an_error():
    assert cosine_similarity([0, 0], [1, 1]) == APPROX(0.0)


def test_cosine_ranks_a_paraphrase_above_an_unrelated_text():
    """«charge was declined» ближе к «payment didn't go through», чем «payment on time»."""
    assert cosine_similarity(PAYMENT_FAILED, CHARGE_DECLINED) > cosine_similarity(
        PAYMENT_FAILED, PAYMENT_ON_TIME
    )


# --------------------------------------------------------- euclidean_distance
def test_euclidean_distance_of_a_three_four_offset_is_five():
    assert euclidean_distance([0, 0], [3, 4]) == APPROX(5.0)


def test_euclidean_distance_to_itself_is_zero():
    assert euclidean_distance([1, 2], [1, 2]) == APPROX(0.0)


def test_euclidean_distance_does_depend_on_vector_length():
    """Обратная сторона косинуса: удвоенный документ уезжает далеко от оригинала."""
    doubled = [2 * x for x in PAYMENT_FAILED]
    assert euclidean_distance(PAYMENT_FAILED, doubled) > 0


# ------------------------------------------------------------------ normalize
def test_normalize_produces_unit_length():
    assert norm(normalize([3, 4])) == APPROX(1.0)


def test_normalize_keeps_the_direction():
    assert normalize([3, 4]) == APPROX([0.6, 0.8])


def test_normalize_of_a_zero_vector_stays_zero():
    assert normalize([0.0, 0.0]) == APPROX([0.0, 0.0])


def test_dot_of_normalized_vectors_equals_cosine():
    """Почему провайдеры отдают нормированные эмбеддинги: тогда хватит быстрого dot."""
    a, b = PAYMENT_FAILED, CHARGE_DECLINED
    assert dot(normalize(a), normalize(b)) == APPROX(cosine_similarity(a, b))


# ---------------------------------------------------------- truncate_embedding
def test_truncate_embedding_keeps_the_leading_dimensions():
    assert truncate_embedding([0.6, 0.8, 0.0], 2) == APPROX([0.6, 0.8])


def test_truncate_embedding_renormalizes():
    """Без нормировки длина зависит от того, сколько ты отрезал, и косинус плывёт."""
    assert norm(truncate_embedding(PAYMENT_FAILED, 2)) == APPROX(1.0)


def test_truncate_embedding_to_more_dimensions_than_there_are_is_a_no_op():
    assert truncate_embedding([3.0, 4.0], 5) == APPROX([0.6, 0.8])


def test_truncate_embedding_rejects_zero_dimensions():
    with pytest.raises(ValueError):
        truncate_embedding([1.0, 2.0], 0)


def test_truncate_embedding_mostly_preserves_the_ranking():
    """Ради этого Matryoshka и нужна: обрезали втрое, порядок выдачи тот же."""
    full = cosine_similarity(PAYMENT_FAILED, CHARGE_DECLINED) > cosine_similarity(
        PAYMENT_FAILED, PAYMENT_ON_TIME
    )
    q = truncate_embedding(PAYMENT_FAILED, 2)
    cut = cosine_similarity(q, truncate_embedding(CHARGE_DECLINED, 2)) > cosine_similarity(
        q, truncate_embedding(PAYMENT_ON_TIME, 2)
    )
    assert cut == full


# ------------------------------------------------------------- binary_quantize
def test_binary_quantize_keeps_only_the_sign():
    assert binary_quantize([0.4, -0.1, 2.0, -2.0]) == [1, 0, 1, 0]


def test_binary_quantize_maps_zero_to_zero():
    assert binary_quantize([0.0]) == [0]


def test_binary_quantize_output_length_matches_the_input():
    assert len(binary_quantize(PAYMENT_FAILED)) == len(PAYMENT_FAILED)


# --------------------------------------------------------------------- search
def test_search_ranks_the_exact_match_first():
    hits = search(PAYMENT_FAILED, CORPUS, top_k=2)
    assert [i for i, _ in hits] == [0, 2]
    assert hits[0][1] == APPROX(1.0)


def test_search_returns_at_most_top_k():
    assert len(search(PAYMENT_FAILED, CORPUS, top_k=1)) == 1


def test_search_breaks_ties_by_index():
    """Иначе порядок выдачи для одинаковых документов пляшет от запуска к запуску."""
    hits = search([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], top_k=2)
    assert [i for i, _ in hits] == [0, 1]


def test_search_by_euclidean_scores_closer_as_higher():
    hits = search([0.0, 0.0], [[3.0, 4.0], [0.0, 1.0]], top_k=2, metric="euclidean")
    assert [i for i, _ in hits] == [1, 0]
    assert hits[0][1] == APPROX(-1.0)


def test_search_by_dot_rewards_long_vectors_unlike_cosine():
    """Ровно та ловушка, из-за которой dot берут только на нормированных векторах."""
    vectors = [[1.0, 0.0], [5.0, 0.0]]
    assert [i for i, _ in search([1.0, 0.0], vectors, top_k=1, metric="dot")] == [1]
    assert [i for i, _ in search([1.0, 0.0], vectors, top_k=1, metric="cosine")] == [0]


def test_search_by_hamming_agrees_with_cosine_on_the_top_hit():
    """32-кратное сжатие памяти не должно менять первого кандидата на простом корпусе."""
    query = [0.9, -0.1, 0.5, -0.4]
    corpus = [[0.8, -0.2, 0.6, -0.3], [-0.7, 0.5, -0.6, 0.9]]
    assert search(query, corpus, top_k=1, metric="hamming")[0][0] == 0
    assert search(query, corpus, top_k=1, metric="cosine")[0][0] == 0


def test_search_rejects_an_unknown_metric():
    assert "manhattan" not in METRICS
    with pytest.raises(ValueError):
        search(PAYMENT_FAILED, CORPUS, metric="manhattan")
