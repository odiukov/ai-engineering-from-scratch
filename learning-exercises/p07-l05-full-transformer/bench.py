"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_D, _HIDDEN, _N = 64, 256, 32

_x = [random.uniform(-2.0, 2.0) for _ in range(_D)]
_rows = [[random.uniform(-2.0, 2.0) for _ in range(_D)] for _ in range(_N)]
_W1 = [[random.uniform(-0.2, 0.2) for _ in range(_D)] for _ in range(_HIDDEN)]
_W3 = [[random.uniform(-0.2, 0.2) for _ in range(_D)] for _ in range(_HIDDEN)]
_W2 = [[random.uniform(-0.2, 0.2) for _ in range(_HIDDEN)] for _ in range(_D)]


def _identity(rows):
    return [[0.0] * len(row) for row in rows]


BENCH = {
    "layer_norm": (_x,),
    "rms_norm": (_x,),
    "silu": (0.7,),
    "ffn_swiglu": (_x, _W1, _W2, _W3),
    "pre_norm_sublayer": (_rows, _identity),
    "post_norm_sublayer": (_rows, _identity),
    "transformer_block": (_rows, [_identity, _identity]),
    "block_params": (4096, 2.6, True, False),
}
