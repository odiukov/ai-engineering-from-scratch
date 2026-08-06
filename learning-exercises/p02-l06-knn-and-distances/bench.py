"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N, _D = 3000, 20
_X = [[random.gauss(0, 1) for _ in range(_D)] for _ in range(_N)]
_y_class = [random.randint(0, 2) for _ in range(_N)]
_y_value = [random.uniform(0, 100) for _ in range(_N)]
_query = [random.gauss(0, 1) for _ in range(_D)]

_points2d = [[random.uniform(0, 100), random.uniform(0, 100)] for _ in range(500)]
_query2d = [50.0, 50.0]

_a = [random.gauss(0, 1) for _ in range(300)]
_b = [random.gauss(0, 1) for _ in range(300)]


def _build(points, depth=0, indices=None):
    """Локальная сборка KD-дерева — bench.py не должен зависеть от exercise.py."""
    if indices is None:
        indices = list(range(len(points)))
    if not indices:
        return None
    axis = depth % len(points[indices[0]])
    indices = sorted(indices, key=lambda i: points[i][axis])
    mid = len(indices) // 2
    return {"point": points[indices[mid]], "index": indices[mid], "axis": axis,
            "left": _build(points, depth + 1, indices[:mid]),
            "right": _build(points, depth + 1, indices[mid + 1:])}


_kdtree = _build(_points2d)


BENCH = {
    "l2_distance": (_a, _b),
    "l1_distance": (_a, _b),
    "cosine_distance": (_a, _b),
    "minkowski_distance": (_a, _b, 3),
    "k_nearest": (_X, _query, 5),
    "knn_classify": (_X, _y_class, _query, 5),
    "knn_regress": (_X, _y_value, _query, 5),
    "standardize": (_X,),
    "build_kdtree": (_points2d,),
    "kdtree_nearest": (_kdtree, _query2d),
}
