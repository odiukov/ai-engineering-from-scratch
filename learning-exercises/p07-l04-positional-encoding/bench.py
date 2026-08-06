"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N, _D = 128, 64

_X = [[random.uniform(-1.0, 1.0) for _ in range(_D)] for _ in range(_N)]
_Q = [random.uniform(-1.0, 1.0) for _ in range(_D)]
_K = [random.uniform(-1.0, 1.0) for _ in range(_D)]

BENCH = {
    "sinusoidal_encoding": (_N, _D),
    "add_positional_encoding": (_X,),
    "apply_rope": (_Q, 77),
    "rope_dot": (_Q, _K, 77, 12),
    "scale_rope_base": (10000, 32.0, 128),
    "alibi_slopes": (32,),
    "alibi_bias": (8, 64),
}
