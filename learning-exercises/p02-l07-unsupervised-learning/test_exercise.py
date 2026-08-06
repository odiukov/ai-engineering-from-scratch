"""Тесты к уроку «Обучение без учителя». Правь exercise.py."""

import math

import pytest

from exercise import (
    assign_clusters,
    best_k_by_silhouette,
    dbscan,
    euclidean_distance,
    inertia,
    kmeans,
    silhouette_score,
    update_centroids,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

BLOB_A = [[0, 0], [0, 1], [1, 0], [1, 1]]
BLOB_B = [[10, 0], [10, 1], [11, 0], [11, 1]]
BLOB_C = [[5, 10], [5, 11], [6, 10], [6, 11]]
THREE_BLOBS = BLOB_A + BLOB_B + BLOB_C


def groups(assignments):
    """Разбиение как множество множеств: сравнение, не зависящее от номеров меток."""
    buckets = {}
    for i, label in enumerate(assignments):
        buckets.setdefault(label, set()).add(i)
    return {frozenset(v) for v in buckets.values()}


# ------------------------------------------------------ euclidean_distance
def test_euclidean_distance_is_the_hypotenuse():
    assert euclidean_distance([0, 0], [3, 4]) == APPROX(5.0)


def test_euclidean_distance_to_itself_is_zero():
    assert euclidean_distance([1.5, -2.0, 7.0], [1.5, -2.0, 7.0]) == APPROX(0.0)


def test_euclidean_distance_is_symmetric():
    """Метрика симметрична: порядок аргументов не меняет ответ."""
    a, b = [1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]
    assert euclidean_distance(a, b) == APPROX(euclidean_distance(b, a))


# --------------------------------------------------------- assign_clusters
def test_assign_clusters_picks_the_nearest_centroid():
    assert assign_clusters([[0, 0], [9, 9]], [[0, 0], [10, 10]]) == [0, 1]


def test_assign_clusters_uses_distance_not_index_order():
    """Центры перечислены в обратном порядке — метки обязаны перевернуться."""
    assert assign_clusters([[4, 0], [6, 0]], [[10, 0], [0, 0]]) == [1, 0]


def test_assign_clusters_returns_one_label_per_point():
    assignments = assign_clusters(THREE_BLOBS, [[0, 0], [10, 0], [5, 10]])
    assert len(assignments) == len(THREE_BLOBS)


# -------------------------------------------------------- update_centroids
def test_update_centroids_is_the_mean_of_its_members():
    assert update_centroids([[0, 0], [2, 2]], [0, 0], [[9, 9]]) == [[1.0, 1.0]]


def test_update_centroids_keeps_an_empty_cluster_in_place():
    """Ловушка: у пустого кластера нет среднего — делить на ноль нельзя."""
    result = update_centroids([[0, 0]], [0], [[0, 0], [5, 5]])
    assert result[1] == [5, 5]


def test_update_centroids_does_not_mutate_the_data():
    data = [[0.0, 0.0], [2.0, 2.0]]
    update_centroids(data, [0, 0], [[9.0, 9.0]])
    assert data == [[0.0, 0.0], [2.0, 2.0]]


# ------------------------------------------------------------------ kmeans
def test_kmeans_separates_two_far_blobs():
    assignments, _ = kmeans([[0, 0], [0, 1], [9, 9], [9, 8]], 2, seed=0)
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]


def test_kmeans_recovers_three_blobs():
    assignments, centroids = kmeans(THREE_BLOBS, 3, seed=0)
    assert len(centroids) == 3
    assert groups(assignments) == {
        frozenset(range(0, 4)),
        frozenset(range(4, 8)),
        frozenset(range(8, 12)),
    }


def test_kmeans_is_reproducible_for_the_same_seed():
    """Без этого свойства ни один тест на k-means не имел бы смысла."""
    assert kmeans(THREE_BLOBS, 3, seed=1) == kmeans(THREE_BLOBS, 3, seed=1)


def test_kmeans_can_land_in_a_local_minimum():
    """Неудачный старт даёт заметно худшее разбиение — это и есть свойство метода."""
    good_a, good_c = kmeans(THREE_BLOBS, 3, seed=0)
    bad_a, bad_c = kmeans(THREE_BLOBS, 3, seed=5)
    assert inertia(THREE_BLOBS, bad_a, bad_c) > inertia(THREE_BLOBS, good_a, good_c)


def test_kmeans_does_not_mutate_the_data():
    """Ловушка: центры, взятые ссылкой на точки данных, портят сами данные."""
    data = [[0.0, 0.0], [0.0, 1.0], [9.0, 9.0], [9.0, 8.0]]
    kmeans(data, 2, seed=3)
    assert data == [[0.0, 0.0], [0.0, 1.0], [9.0, 9.0], [9.0, 8.0]]


# ----------------------------------------------------------------- inertia
def test_inertia_counts_squared_distance():
    assert inertia([[0, 0], [2, 0]], [0, 0], [[1, 0]]) == APPROX(2.0)


def test_inertia_of_perfect_centroids_is_zero():
    assert inertia([[5, 5]], [0], [[5, 5]]) == APPROX(0.0)


def test_inertia_never_grows_when_k_grows():
    """Инерция монотонно падает с ростом k — поэтому по её минимуму k не выбирают."""
    values = []
    for k in (1, 2, 3, 4):
        assignments, centroids = kmeans(THREE_BLOBS, k, seed=0)
        values.append(inertia(THREE_BLOBS, assignments, centroids))
    assert all(later <= earlier + 1e-9 for earlier, later in zip(values, values[1:]))


# -------------------------------------------------------- silhouette_score
def test_silhouette_of_well_separated_clusters_is_near_one():
    score = silhouette_score([[0, 0], [0, 1], [9, 9], [9, 8]], [0, 0, 1, 1])
    assert score == pytest.approx(0.917, abs=1e-3)


def test_silhouette_of_a_single_cluster_is_zero():
    """Ловушка: не с чем сравнивать — метрика не определена, договорились про 0.0."""
    assert silhouette_score([[0, 0], [1, 1], [2, 2]], [0, 0, 0]) == APPROX(0.0)


def test_silhouette_is_negative_when_labels_are_swapped():
    """Точки приписаны чужим кластерам — силуэт уходит в минус."""
    data = [[0, 0], [0, 1], [9, 9], [9, 8]]
    assert silhouette_score(data, [0, 1, 0, 1]) < 0


def test_silhouette_scores_a_lonely_point_as_zero():
    """У одиночки нет «своих», среднее a не определено — его вклад ровно 0.0."""
    data = [[0, 0], [0, 1], [9, 9]]
    lonely = silhouette_score(data, [0, 0, 1])
    pair_only = silhouette_score(data[:2] + [[9, 9], [9, 8]], [0, 0, 1, 1])
    assert lonely < pair_only


# --------------------------------------------------- best_k_by_silhouette
def test_best_k_finds_three_blobs():
    assert best_k_by_silhouette(THREE_BLOBS, [2, 3, 4], seed=0) == 3


def test_best_k_with_one_candidate_returns_it():
    assert best_k_by_silhouette(THREE_BLOBS, [2], seed=0) == 2


# ------------------------------------------------------------------ dbscan
def test_dbscan_finds_two_dense_blobs():
    labels = dbscan(BLOB_A + BLOB_B, 2.0, 3)
    assert len(set(labels)) == 2
    assert labels[0] == labels[3] != labels[4] == labels[7]


def test_dbscan_marks_a_far_point_as_noise():
    labels = dbscan(BLOB_A + BLOB_B + [[100, 100]], 2.0, 3)
    assert labels[-1] == -1
    assert labels.count(-1) == 1


def test_dbscan_calls_everything_noise_when_min_samples_is_too_high():
    """Ловушка: min_samples больше размера облака — ядровых точек нет вообще."""
    assert dbscan(BLOB_A, 2.0, 10) == [-1, -1, -1, -1]


def test_dbscan_merges_a_chain_into_one_cluster():
    """Цепочка не шарообразна — k-means её разрежет, DBSCAN сохранит целой."""
    chain = [[float(i), 0.0] for i in range(6)]
    assert dbscan(chain, 1.1, 2) == [0, 0, 0, 0, 0, 0]


def test_dbscan_is_deterministic():
    """Случайности внутри нет: два прогона обязаны совпасть до метки."""
    data = BLOB_A + BLOB_B + [[100, 100]]
    assert dbscan(data, 2.0, 3) == dbscan(data, 2.0, 3)


def test_dbscan_needs_eps_big_enough_to_reach_neighbours():
    """Радиус меньше шага сетки — соседей нет ни у кого, всё становится шумом."""
    assert set(dbscan(BLOB_A, 0.5, 2)) == {-1}
    assert math.isclose(euclidean_distance(BLOB_A[0], BLOB_A[1]), 1.0)
