"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_D_IN, _D_OUT, _RANK = 96, 96, 8

_W = [[random.gauss(0, 0.1) for _ in range(_D_IN)] for _ in range(_D_OUT)]
_A = [[random.gauss(0, 0.3) for _ in range(_D_IN)] for _ in range(_RANK)]
_B = [[random.gauss(0, 0.3) for _ in range(_RANK)] for _ in range(_D_OUT)]
_X = [random.gauss(0, 1) for _ in range(_D_IN)]
_TARGET = [random.gauss(0, 1) for _ in range(_D_OUT)]

_SMALL_W = [[random.gauss(0, 0.1) for _ in range(8)] for _ in range(8)]
_DATA = [
    ([random.gauss(0, 1) for _ in range(8)], [random.gauss(0, 1) for _ in range(8)])
    for _ in range(20)
]

BENCH = {
    "linear": (_W, _X),
    "init_lora": (_D_IN, _D_OUT, _RANK, 0),
    "lora_forward": (_W, _A, _B, 16.0, _X),
    "merge_lora": (_W, _A, _B, 16.0),
    "count_trainable": (4096, 4096, 16),
    "quantize_dequantize": (_W, 64),
    "lora_grads": (_W, _A, _B, 16.0, _X, _TARGET),
    "train_lora": (_SMALL_W, _DATA, 2, 4.0, 0.05, 10, 0),
}
