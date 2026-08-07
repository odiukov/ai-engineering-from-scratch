"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 64
_BATCH = 48
_IMAGES = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(_BATCH)]
_TEXTS = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(_BATCH)]
_S = [[random.gauss(0, 2) for _ in range(_BATCH)] for _ in range(_BATCH)]
_CLASSES = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(200)]
_NAMES = [f"class_{i}" for i in range(200)]

BENCH = {
    "l2_normalize": (_IMAGES[0],),
    "cosine_similarity": (_IMAGES[0], _TEXTS[0]),
    "similarity_matrix": (_IMAGES, _TEXTS, 0.07),
    "infonce_loss": (_S,),
    "infonce_grad": (_S,),
    "sigmoid_pairwise_loss": (_S, -10.0),
    "zero_shot_classify": (_IMAGES[0], _CLASSES, _NAMES),
    "prompt_ensemble": (_CLASSES,),
}
