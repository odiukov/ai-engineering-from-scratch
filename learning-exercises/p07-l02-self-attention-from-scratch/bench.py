"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _matrix(rows, cols):
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


_N, _D, _DK = 48, 32, 16

_X = _matrix(_N, _D)
_Q = _matrix(_N, _DK)
_K = _matrix(_N, _DK)
_V = _matrix(_N, _DK)
_WQ = _matrix(_D, _DK)
_WK = _matrix(_D, _DK)
_WV = _matrix(_D, _DK)
_MASK = [[j <= i for j in range(_N)] for i in range(_N)]

BENCH = {
    "softmax": ([random.uniform(-3.0, 3.0) for _ in range(512)],),
    "transpose": (_X,),
    "matmul": (_X, _WQ),
    "attention_scores": (_Q, _K),
    "causal_mask": (_N,),
    "scaled_dot_product_attention": (_Q, _K, _V, _MASK),
    "self_attention": (_X, _WQ, _WK, _WV),
}
