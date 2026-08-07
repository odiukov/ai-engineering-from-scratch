"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 512  # длина контекста: шумовой пол виден только на длинном контексте
_D = 32  # размерность головы
_QUERIES = 8


def _matrix(rows, cols):
    return [[random.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


_K1 = _matrix(_N, _D)
_K2 = _matrix(_N, _D)
_V = _matrix(_N, _D)
_Q1 = _matrix(_QUERIES, _D)
_Q2 = _matrix(_QUERIES, _D)

_row = [random.gauss(0.0, 1.0) for _ in range(_N)]
_w1 = [1.0 / _N] * _N
_w2 = [1.0 / _N] * _N

_grid = [i / 100 for i in range(201)]

BENCH = {
    "softmax": (_row,),
    "attention_weights": (_Q1[0], _K1),
    "attend": (_w1, _V),
    "diff_weights": (_w1, _w2, 0.8),
    "diff_attention": (_Q1, _K1, _Q2, _K2, _V, 0.8),
    "signal_to_noise": (_w1, 7),
    "best_lambda": (_Q1[0], _K1, _Q2[0], _K2, 7, _grid),
    "attention_param_count": (4096, 32, 128, "v2"),
}
