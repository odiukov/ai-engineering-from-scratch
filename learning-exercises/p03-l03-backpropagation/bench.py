"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N_IN, _N_HIDDEN = 20, 30
_scale = (2.0 / _N_IN) ** 0.5

_params = {
    "w1": [[random.uniform(-_scale, _scale) for _ in range(_N_IN)] for _ in range(_N_HIDDEN)],
    "b1": [0.0] * _N_HIDDEN,
    "w2": [random.uniform(-_scale, _scale) for _ in range(_N_HIDDEN)],
    "b2": 0.0,
}
_grads = {
    "w1": [[0.01] * _N_IN for _ in range(_N_HIDDEN)],
    "b1": [0.01] * _N_HIDDEN,
    "w2": [0.01] * _N_HIDDEN,
    "b2": 0.01,
}
_x = [random.uniform(-1.0, 1.0) for _ in range(_N_IN)]

BENCH = {
    "sigmoid": (0.7,),
    "init_params": (_N_IN, _N_HIDDEN, 0),
    "forward": (_params, _x),
    "loss_for_params": (_params, _x, 1.0),
    "backward": (_params, _x, 1.0),
    "numeric_gradient": (_params, _x, 1.0),
    "sgd_step": (_params, _grads, 0.1),
    "train_xor": (0, 4, 1.0, 200),
}
