"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_dim = 256
_query = [random.uniform(-1, 1) for _ in range(_dim)]
_corpus = [[random.uniform(-1, 1) for _ in range(_dim)] for _ in range(400)]

BENCH = {
    "dot": (_query, _corpus[0]),
    "norm": (_query,),
    "cosine_similarity": (_query, _corpus[0]),
    "euclidean_distance": (_query, _corpus[0]),
    "normalize": (_query,),
    "truncate_embedding": (_query, 64),
    "binary_quantize": (_query,),
    "search": (_query, _corpus, 10, "cosine"),
}
