"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 128
_a = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
_b = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
# 300 точек по 128 координат: полный перебор соседа — десятки миллисекунд
_points = [[random.uniform(-1.0, 1.0) for _ in range(_DIM)] for _ in range(300)]
_inv_cov = [[1.0 if i == j else 0.05 for j in range(_DIM)] for i in range(_DIM)]
_set_a = set(random.sample(range(5000), 800))
_set_b = set(random.sample(range(5000), 800))
_word_a = "".join(random.choice("abcdefgh") for _ in range(160))
_word_b = "".join(random.choice("abcdefgh") for _ in range(160))

# метрика для nearest_neighbor задаётся здесь, а не берётся из решения:
# замер должен быть одинаковым для эталона и для твоего файла
_l2 = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

BENCH = {
    "dot": (_a, _b),
    "lp_norm": (_a, 2),
    "distance": (_a, _b, 1),
    "cosine_similarity": (_a, _b),
    "mahalanobis": (_a, _b, _inv_cov),
    "jaccard_similarity": (_set_a, _set_b),
    "edit_distance": (_word_a, _word_b),
    "nearest_neighbor": (_a, _points, _l2),
}
