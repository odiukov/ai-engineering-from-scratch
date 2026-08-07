"""Тесты к уроку «Поиск по картинкам и metric learning». Правь exercise.py."""

import math

import pytest

from exercise import (
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
    precision_at_k,
    rank_gallery,
    recall_at_k,
    semi_hard_negative,
    triplet_loss,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)


def flat(rows):
    """pytest.approx не умеет вложенные списки — разворачиваем в плоский."""
    return [x for row in rows for x in row]


GALLERY = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
LABELS = [0, 0, 1, 2]


# ------------------------------------------------------------ l2_normalize
def test_normalize_gives_unit_length():
    assert l2_normalize([3.0, 4.0]) == APPROX([0.6, 0.8])


def test_normalized_vector_has_norm_one():
    v = l2_normalize([2.0, -7.0, 0.5])
    assert math.sqrt(sum(x * x for x in v)) == APPROX(1.0)


def test_normalize_keeps_the_direction():
    """Нормировка меняет длину, но не направление: косинус с оригиналом = 1."""
    v = [2.0, -7.0, 0.5]
    assert cosine_similarity(v, l2_normalize(v)) == pytest.approx(1.0, abs=1e-12)


def test_normalize_of_zero_vector_does_not_divide_by_zero():
    assert l2_normalize([0.0, 0.0]) == APPROX([0.0, 0.0])


# ------------------------------------------------------- cosine_similarity
def test_cosine_of_parallel_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == APPROX(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == APPROX(-1.0)


def test_cosine_does_not_depend_on_length():
    """Яркая и тусклая копии одной картинки должны быть одинаково похожи."""
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    scaled = [100 * x for x in b]
    assert cosine_similarity(a, scaled) == pytest.approx(cosine_similarity(a, b), abs=1e-12)


def test_cosine_with_a_zero_vector_is_zero_not_an_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == APPROX(0.0)


# ------------------------------------------------------ euclidean_distance
def test_euclidean_of_the_classic_triangle():
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == APPROX(5.0)


def test_euclidean_of_a_point_with_itself_is_zero():
    assert euclidean_distance([1.0, 2.0], [1.0, 2.0]) == APPROX(0.0)


def test_squared_distance_equals_two_minus_two_cosine_on_the_unit_sphere():
    """Тождество из урока: на нормированных векторах метрики эквивалентны."""
    a = l2_normalize([1.0, 2.0, -3.0])
    b = l2_normalize([-2.0, 0.5, 1.0])
    assert euclidean_distance(a, b) ** 2 == pytest.approx(
        2 - 2 * cosine_similarity(a, b), abs=1e-9
    )


# ------------------------------------------------------------ triplet_loss
def test_triplet_loss_is_zero_when_the_negative_is_already_far():
    assert triplet_loss([0.0, 0.0], [1.0, 0.0], [0.0, 3.0]) == APPROX(0.0)


def test_triplet_loss_is_positive_when_the_negative_is_closer():
    assert triplet_loss([0.0, 0.0], [0.0, 2.0], [1.0, 0.0]) == APPROX(1.2)


def test_triplet_loss_never_goes_below_zero():
    """max(0, ...) обязателен: без него уже разведённые триплеты тянут лосс вниз."""
    assert triplet_loss([0.0], [0.1], [50.0]) >= 0.0


def test_margin_is_what_keeps_a_barely_separated_triplet_alive():
    """d_ap = d_an: без margin лосс был бы нулевым, с margin — ровно margin."""
    assert triplet_loss([0.0], [1.0], [-1.0], margin=0.3) == APPROX(0.3)


# ------------------------------------------------------ semi_hard_negative
def test_semi_hard_picks_the_negative_inside_the_band():
    assert semi_hard_negative([0.0], [1.0], [[5.0], [1.1], [0.5]], margin=0.5) == 1


def test_semi_hard_skips_the_hard_negative_closer_than_the_positive():
    """[0.5] ближе positive — это hard negative, на нём эмбеддинг коллапсирует."""
    chosen = semi_hard_negative([0.0], [1.0], [[0.5], [1.2]], margin=0.5)
    assert chosen == 1


def test_semi_hard_falls_back_to_the_hardest_when_the_band_is_empty():
    assert semi_hard_negative([0.0], [1.0], [[9.0], [8.0]], margin=0.5) == 1


def test_semi_hard_takes_the_closest_candidate_inside_the_band():
    """Из полосы берём самый трудный, то есть ближайший к anchor."""
    assert semi_hard_negative([0.0], [1.0], [[1.45], [1.1], [1.3]], margin=0.5) == 1


def test_semi_hard_stays_outside_the_positive_radius_when_it_can():
    negs = [[0.2], [0.4], [1.3], [4.0]]
    chosen = semi_hard_negative([0.0], [1.0], negs, margin=0.5)
    assert negs[chosen][0] > 1.0


# ------------------------------------------------------------ rank_gallery
def test_rank_puts_the_most_similar_first():
    assert rank_gallery([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]) == [1, 0, 2]


def test_rank_returns_every_index_exactly_once():
    order = rank_gallery([0.3, -0.7], GALLERY)
    assert sorted(order) == list(range(len(GALLERY)))


def test_rank_breaks_ties_by_index():
    """Два одинаковых кандидата обязаны выходить в стабильном порядке."""
    assert rank_gallery([1.0, 0.0], [[2.0, 0.0], [5.0, 0.0]]) == [0, 1]


def test_rank_ignores_the_length_of_gallery_vectors():
    small = [[0.001, 0.0], [0.0, 0.001]]
    big = [[1000.0, 0.0], [0.0, 1000.0]]
    assert rank_gallery([1.0, 0.0], small) == rank_gallery([1.0, 0.0], big)


# ------------------------------------------------------------- recall_at_k
def test_recall_at_one_finds_the_exact_duplicate():
    assert recall_at_k([[1.0, 0.0]], [0], GALLERY, LABELS, k=1) == APPROX(1.0)


def test_recall_is_zero_when_the_only_match_is_ranked_last():
    assert recall_at_k([[1.0, 0.0]], [2], GALLERY, LABELS, k=1) == APPROX(0.0)


def test_recall_never_decreases_as_k_grows():
    """Главное свойство метрики: топ-K с ростом K только расширяется."""
    queries = [[1.0, 0.0], [0.1, 1.0], [-0.9, 0.2]]
    q_labels = [2, 1, 0]
    values = [recall_at_k(queries, q_labels, GALLERY, LABELS, k) for k in range(1, 5)]
    assert values == sorted(values)


def test_recall_reaches_one_when_k_covers_the_whole_gallery():
    queries = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    q_labels = [0, 1, 2]
    assert recall_at_k(queries, q_labels, GALLERY, LABELS, k=4) == APPROX(1.0)


def test_recall_averages_over_queries():
    """Один запрос попал, второй нет — ровно половина."""
    queries = [[1.0, 0.0], [1.0, 0.0]]
    assert recall_at_k(queries, [0, 2], GALLERY, LABELS, k=1) == APPROX(0.5)


# ---------------------------------------------------------- precision_at_k
def test_precision_counts_how_many_of_the_top_k_are_right():
    assert precision_at_k([[1.0, 0.0]], [0], GALLERY, LABELS, k=3) == APPROX(2 / 3)


def test_precision_at_one_matches_recall_at_one():
    """При K = 1 "хотя бы один" и "какая доля" — одно и то же."""
    queries = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    q_labels = [0, 1, 2]
    assert precision_at_k(queries, q_labels, GALLERY, LABELS, 1) == APPROX(
        recall_at_k(queries, q_labels, GALLERY, LABELS, 1)
    )


def test_precision_never_exceeds_recall():
    queries = [[1.0, 0.0], [0.1, 1.0], [-0.9, 0.2]]
    q_labels = [0, 1, 2]
    for k in range(1, 5):
        p = precision_at_k(queries, q_labels, GALLERY, LABELS, k)
        r = recall_at_k(queries, q_labels, GALLERY, LABELS, k)
        assert p <= r + 1e-12


def test_precision_falls_as_k_grows_past_the_matches():
    """Класс 1 в галерее один: расширяя K, мы добираем только мусор."""
    q = [[0.0, 1.0]]
    assert precision_at_k(q, [1], GALLERY, LABELS, 1) > precision_at_k(
        q, [1], GALLERY, LABELS, 4
    )
