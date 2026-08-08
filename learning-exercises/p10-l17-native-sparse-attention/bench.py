"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 1024  # длина контекста из урока: l=32, k=4, w=128
_D = 32


def _matrix(rows, cols):
    return [[random.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


_K = _matrix(_N, _D)
_V = _matrix(_N, _D)
_q = [random.gauss(0.0, 1.0) for _ in range(_D)]

_row = [random.gauss(0.0, 1.0) for _ in range(_N)]
_weights = [1.0 / _N] * _N
_block_scores = [random.random() for _ in range(_N // 32)]

BENCH = {
    "softmax": (_row,),
    "attention_weights": (_q, _K),
    "attend": (_weights, _V),
    "compress_blocks": (_K, 32),
    "top_k_blocks": (_block_scores, 4),
    "selected_branch": (_q, _K, _V, 32, 4),
    "nsa_attention": (_q, _K, _V, 32, 4, 64, 128, (0.4, 0.4, 0.2)),
    "keys_per_query": (64000, 64, 16, 64, 512),
}
