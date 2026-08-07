"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [f"w{i}" for i in range(400)]
_DOCS = [[random.choice(_WORDS) for _ in range(60)] for _ in range(120)]
_VOCAB = {w: i for i, w in enumerate(_WORDS)}
_DIM = 32
_W = [[random.gauss(0, 0.1) for _ in range(_DIM)] for _ in _WORDS]
_W_PRIME = [[random.gauss(0, 0.1) for _ in range(_DIM)] for _ in _WORDS]
_VEC_A = [random.random() for _ in range(_DIM)]
_VEC_B = [random.random() for _ in range(_DIM)]

BENCH = {
    "build_vocab": (_DOCS,),
    "skipgram_pairs": (_DOCS, 3),
    "sigmoid": (0.7,),
    "negative_samples": (len(_VOCAB), {0, 1}, 500, random.Random(0)),
    "train_pair": (_W, _W_PRIME, 3, 7, [11, 12, 13, 14, 15], 0.05),
    "cosine_similarity": (_VEC_A, _VEC_B),
    "nearest": (_VOCAB, _W, _VEC_A, 5),
    "analogy": (_VOCAB, _W, "w0", "w1", "w2", 5),
}
