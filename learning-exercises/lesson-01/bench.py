"""Входные данные для замера скорости. Аргументы каждой функции."""

import random

random.seed(0)

_v1 = [random.uniform(-1, 1) for _ in range(300)]
_v2 = [random.uniform(-1, 1) for _ in range(300)]
_M = [[random.uniform(-1, 1) for _ in range(300)] for _ in range(50)]
_corpus = [[random.uniform(-1, 1) for _ in range(64)] for _ in range(40)]

BENCH = {
    "magnitude": (_v1,),
    "dot": (_v1, _v2),
    "cosine_similarity": (_v1, _v2),
    "angle_between": (_v1, _v2),
    "project": (_v1, _v2),
    "matvec": (_M, _v1),
    "is_invertible_2x2": ([[1.0, 2.0], [3.0, 4.0]],),
    "most_similar_pair": (_corpus,),
}
