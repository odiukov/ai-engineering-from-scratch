"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 64


def _vec():
    return [random.gauss(0.0, 1.0) for _ in range(_DIM)]


_a, _b = _vec(), _vec()
_negatives = [_vec() for _ in range(400)]

_gallery = [_vec() for _ in range(600)]
_gallery_labels = [i % 10 for i in range(600)]
_queries = [_vec() for _ in range(60)]
_query_labels = [i % 10 for i in range(60)]

BENCH = {
    "l2_normalize": (_a,),
    "cosine_similarity": (_a, _b),
    "euclidean_distance": (_a, _b),
    "triplet_loss": (_a, _b, _negatives[0], 0.2),
    "semi_hard_negative": (_a, _b, _negatives, 0.2),
    "rank_gallery": (_a, _gallery),
    "recall_at_k": (_queries, _query_labels, _gallery, _gallery_labels, 5),
    "precision_at_k": (_queries, _query_labels, _gallery, _gallery_labels, 5),
}
