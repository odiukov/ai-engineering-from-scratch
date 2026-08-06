"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _matrix(rows, cols):
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


_N, _D, _HEADS = 48, 32, 4
_D_HEAD = _D // _HEADS

_X = _matrix(_N, _D)
_WQ, _WK, _WV, _WO = _matrix(_D, _D), _matrix(_D, _D), _matrix(_D, _D), _matrix(_D, _D)
_Q = _matrix(_N, _D_HEAD)
_K = _matrix(_N, _D_HEAD)
_V = _matrix(_N, _D_HEAD)
_HEAD_LIST = [_matrix(_N, _D_HEAD) for _ in range(_HEADS)]

BENCH = {
    "matmul": (_X, _WQ),
    "softmax": ([random.uniform(-3.0, 3.0) for _ in range(512)],),
    "split_heads": (_X, _HEADS),
    "combine_heads": (_HEAD_LIST,),
    "head_attention": (_Q, _K, _V),
    "repeat_kv_heads": (_HEAD_LIST, 16),
    "multi_head_attention": (_X, _WQ, _WK, _WV, _WO, _HEADS),
    "kv_cache_cells": (4096, 8, 128, 32),
}
