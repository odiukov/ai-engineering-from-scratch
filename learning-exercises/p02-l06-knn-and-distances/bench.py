"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N, _D = 3000, 20
_X = [[random.gauss(0, 1) for _ in range(_D)] for _ in range(_N)]
_y_class = [random.randint(0, 2) for _ in range(_N)]
_y_value = [random.uniform(0, 100) for _ in range(_N)]
_query = [random.gauss(0, 1) for _ in range(_D)]

_a = [random.gauss(0, 1) for _ in range(300)]
_b = [random.gauss(0, 1) for _ in range(300)]

BENCH = {
    "l2_distance": (_a, _b),
    "l1_distance": (_a, _b),
    "cosine_distance": (_a, _b),
    "minkowski_distance": (_a, _b, 3),
    "k_nearest": (_X, _query, 5),
    "knn_classify": (_X, _y_class, _query, 5),
    "knn_regress": (_X, _y_value, _query, 5),
    "standardize": (_X,),
}
