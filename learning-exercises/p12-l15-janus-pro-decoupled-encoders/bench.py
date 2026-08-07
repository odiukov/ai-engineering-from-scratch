"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 32
_N_CODES = 128   # размер codebook: наивный поиск ближайшего это N_CODES * DIM
_N_PATCHES = 144  # столько патчей у картинки 12x12

_CODEBOOK = [[random.gauss(0.0, 1.0) for _ in range(_DIM)] for _ in range(_N_CODES)]
_VECTORS = [[random.gauss(0.0, 1.0) for _ in range(_DIM)] for _ in range(_N_PATCHES)]
_INDICES = [random.randrange(_N_CODES) for _ in range(_N_PATCHES)]

# для semantic_margin квадратичное число пар, поэтому выборка поменьше
_LABELLED = _VECTORS[:64]
_LABELS = ["a" if i % 2 == 0 else "b" for i in range(64)]

BENCH = {
    "cosine_similarity": (_VECTORS[0], _VECTORS[1]),
    "nearest_code": (_VECTORS[0], _CODEBOOK),
    "vq_encode": (_VECTORS, _CODEBOOK),
    "vq_reconstruct": (_INDICES, _CODEBOOK),
    "reconstruction_error": (_VECTORS, _CODEBOOK),
    "semantic_margin": (_LABELLED, _LABELS),
    "route": ("Describe the photo and then render a sketch of it",),
    "encoder_for": ("generate",),
}
