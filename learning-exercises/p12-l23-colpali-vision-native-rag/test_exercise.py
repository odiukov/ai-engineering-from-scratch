"""Тесты к уроку «ColPali и vision-native RAG по документам». Правь exercise.py."""

import math

import pytest

from exercise import (
    COLPALI_DIM,
    PATCHES_PER_PAGE,
    bi_encoder_score,
    cosine,
    l2_normalize,
    maxsim,
    mean_sim,
    pool_page,
    retrieve,
    storage_bytes,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

# Страница, где ответ занимает ОДИН патч из десяти, остальное — пустое поле.
NEEDLE_PAGE = [[1.0, 0.0, 0.0]] + [[0.0, 1.0, 0.0]] * 9
# Страница, где к запросу слабо относится каждый патч.
BLAND_PAGE = [[0.5, math.sqrt(3) / 2, 0.0]] * 10
QUERY = [[1.0, 0.0, 0.0]]


# -------------------------------------------------------------------- cosine
def test_cosine_of_identical_vectors_is_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == APPROX(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == APPROX(0.0)


def test_cosine_ignores_vector_length():
    """Именно поэтому косинус, а не скалярное произведение: яркий патч не должен побеждать длиной."""
    assert cosine([1.0, 2.0], [5.0, 10.0]) == APPROX(1.0)


def test_cosine_of_a_zero_vector_is_zero_not_a_crash():
    """Пустое белое поле страницы даёт ровно такой патч."""
    assert cosine([0.0, 0.0], [1.0, 1.0]) == APPROX(0.0)


# -------------------------------------------------------------- l2_normalize
def test_l2_normalize_worked_example():
    assert l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8], abs=1e-12)


def test_l2_normalize_gives_unit_length():
    v = l2_normalize([1.0, -2.0, 3.0, 0.5])
    assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-12)


def test_l2_normalize_keeps_the_direction():
    v = [1.0, -2.0, 3.0]
    assert cosine(v, l2_normalize(v)) == pytest.approx(1.0, abs=1e-12)


def test_l2_normalize_leaves_a_zero_vector_alone():
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


# -------------------------------------------------------------------- maxsim
def test_maxsim_takes_the_best_patch_per_query_token():
    assert maxsim([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]) == APPROX(1.0)


def test_maxsim_sums_across_query_tokens():
    assert maxsim([[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]]) == pytest.approx(
        math.sqrt(2.0), abs=1e-9
    )


def test_maxsim_does_not_care_about_patch_order():
    """Патчи страницы — множество, а не последовательность."""
    page = [[0.3, 0.9], [1.0, 0.0], [-0.2, 0.4]]
    q = [[0.8, 0.1], [0.0, 1.0]]
    assert maxsim(q, page) == APPROX(maxsim(q, list(reversed(page))))


def test_extra_irrelevant_patches_never_lower_the_maxsim_score():
    """Максимум по патчам монотонен: лишнее содержимое страницы не мешает найти ответ."""
    small = maxsim(QUERY, [[1.0, 0.0, 0.0]])
    big = maxsim(QUERY, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert big == APPROX(small)


def test_maxsim_is_a_sum_not_a_mean():
    """Удвоенный запрос даёт удвоенный балл — страницы сравнивают при ОДНОМ запросе."""
    page = [[1.0, 0.0], [0.0, 1.0]]
    one = maxsim([[0.6, 0.8]], page)
    two = maxsim([[0.6, 0.8], [0.6, 0.8]], page)
    assert two == APPROX(2 * one)


# ------------------------------------------------------------------ mean_sim
def test_mean_sim_worked_example():
    assert mean_sim([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]) == APPROX(0.5)


def test_mean_sim_punishes_a_page_for_having_other_content():
    """Вот что теряет усреднение и сохраняет MaxSim."""
    small = mean_sim(QUERY, [[1.0, 0.0, 0.0]])
    big = mean_sim(QUERY, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    assert big < small


def test_mean_sim_does_not_care_about_patch_order():
    page = [[0.3, 0.9], [1.0, 0.0], [-0.2, 0.4]]
    q = [[0.8, 0.1]]
    assert mean_sim(q, page) == APPROX(mean_sim(q, list(reversed(page))))


# ----------------------------------------------------------------- pool_page
def test_pool_page_of_identical_patches_is_that_patch():
    assert pool_page([[1.0, 0.0], [1.0, 0.0]]) == pytest.approx([1.0, 0.0], abs=1e-12)


def test_pool_page_returns_a_unit_vector():
    v = pool_page([[1.0, 2.0], [3.0, -1.0], [0.0, 0.5]])
    assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0, abs=1e-12)


def test_pool_page_does_not_care_about_patch_order():
    page = [[1.0, 2.0], [3.0, -1.0], [0.0, 0.5]]
    assert pool_page(page) == pytest.approx(pool_page(list(reversed(page))), abs=1e-12)


# --------------------------------------------------------- bi_encoder_score
def test_bi_encoder_score_on_a_uniform_page():
    assert bi_encoder_score([[1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]) == APPROX(1.0)


def test_bi_encoder_dilutes_a_single_matching_patch():
    """Один точный патч из десяти растворяется в усреднении — это и есть потеря сигнала."""
    assert bi_encoder_score(QUERY, NEEDLE_PAGE) < 0.2
    assert maxsim(QUERY, NEEDLE_PAGE) == APPROX(1.0)


def test_maxsim_and_bi_encoder_rank_these_two_pages_in_opposite_order():
    """Разрыв ColPali и VisRAG на ViDoRe в одном тесте."""
    assert maxsim(QUERY, NEEDLE_PAGE) > maxsim(QUERY, BLAND_PAGE)
    assert bi_encoder_score(QUERY, NEEDLE_PAGE) < bi_encoder_score(QUERY, BLAND_PAGE)


# ------------------------------------------------------------------ retrieve
def test_retrieve_puts_the_matching_page_first():
    pages = {"bland": BLAND_PAGE, "needle": NEEDLE_PAGE}
    assert retrieve(QUERY, pages, k=1)[0][0] == "needle"


def test_retrieve_returns_scores_in_descending_order():
    pages = {"a": NEEDLE_PAGE, "b": BLAND_PAGE, "c": [[0.0, 0.0, 1.0]] * 4}
    scores = [s for _, s in retrieve(QUERY, pages, k=3)]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_does_not_choke_when_k_exceeds_the_corpus():
    pages = {"a": NEEDLE_PAGE, "b": BLAND_PAGE}
    assert len(retrieve(QUERY, pages, k=99)) == 2


def test_ties_break_by_page_id_so_the_ranking_is_reproducible():
    """Недетерминированный top-k невозможно ни отладить, ни замерить на ViDoRe."""
    pages = {"b": NEEDLE_PAGE, "a": list(NEEDLE_PAGE), "c": list(NEEDLE_PAGE)}
    assert [pid for pid, _ in retrieve(QUERY, pages, k=3)] == ["a", "b", "c"]


# ------------------------------------------------------------- storage_bytes
def test_storage_bytes_for_a_fifty_page_report():
    assert storage_bytes(50) == 50 * PATCHES_PER_PAGE * COLPALI_DIM * 4


def test_pq_compression_shrinks_the_index():
    assert storage_bytes(50, compression=8) == storage_bytes(50) // 8


def test_colpali_costs_far_more_than_text_rag_per_document():
    """50 чанков по 768 измерений — это ~150 КБ против ~18 МБ у ColPali."""
    text_rag = 50 * 768 * 4
    assert storage_bytes(50) > 100 * text_rag


def test_storage_grows_linearly_with_the_corpus():
    assert storage_bytes(200) == 4 * storage_bytes(50)
