"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 32
_SEQ = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(400)]
_KERNEL = [[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(3)]
_FILTERS = [
    ([[random.gauss(0, 1) for _ in range(_DIM)] for _ in range(k)], 0.0)
    for k in (2, 3, 4, 5)
]
_FEATURE_MAP = [random.random() for _ in range(50_000)]

_H = 16
_W_X = [[random.gauss(0, 0.5) for _ in range(_DIM)] for _ in range(_H)]
_W_H = [[random.gauss(0, 0.5) for _ in range(_H)] for _ in range(_H)]
_B = [0.0] * _H
_STATES = [[random.gauss(0, 1) for _ in range(_H)] for _ in range(4000)]
_GATES = {
    key: (
        [[random.gauss(0, 0.5) for _ in range(_DIM)] for _ in range(_H)],
        [[random.gauss(0, 0.5) for _ in range(_H)] for _ in range(_H)],
        [0.0] * _H,
    )
    for key in ("f", "i", "g", "o")
}

BENCH = {
    "conv1d": (_SEQ, _KERNEL, 0.0),
    "global_max_pool": (_FEATURE_MAP,),
    "textcnn_features": (_SEQ, _FILTERS),
    "rnn_step": (_SEQ[0], [0.0] * _H, _W_X, _W_H, _B),
    "rnn_forward": (_SEQ, _W_X, _W_H, _B),
    "pool_hidden": (_STATES, "max"),
    "vanishing_factor": (100, 0.9),
    "lstm_step": (_SEQ[0], [0.0] * _H, [0.0] * _H, _GATES),
}
