"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_centers = [(2.0, 2.0), (8.0, 3.0), (5.0, 8.0)]
_data = [
    [cx + random.gauss(0, 0.6), cy + random.gauss(0, 0.6)]
    for cx, cy in _centers
    for _ in range(40)
]
_assignments = [i // 40 for i in range(len(_data))]
_centroids = [list(c) for c in _centers]

BENCH = {
    "euclidean_distance": (_data[0], _data[-1]),
    "assign_clusters": (_data, _centroids),
    "update_centroids": (_data, _assignments, _centroids),
    "kmeans": (_data, 3),
    "inertia": (_data, _assignments, _centroids),
    "silhouette_score": (_data, _assignments),
    "best_k_by_silhouette": (_data, [2, 3, 4]),
    "dbscan": (_data, 1.0, 5),
}
