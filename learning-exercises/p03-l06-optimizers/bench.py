"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 2000
_params = [random.gauss(0.0, 1.0) for _ in range(_N)]
_grads = [random.gauss(0.0, 1.0) for _ in range(_N)]
_zeros = [0.0] * _N

_quadratic = lambda p: [2.0 * w for w in p]

BENCH = {
    "sgd_step": (_params, _grads, 0.01),
    "momentum_step": (_params, _grads, _zeros, 0.01),
    "bias_correct": (0.1, 0.9, 7),
    "adam_step": (_params, _grads, _zeros, _zeros, 7),
    "adamw_step": (_params, _grads, _zeros, _zeros, 7),
    "run_sgd": (_quadratic, [1.0] * 50, 0.01, 200),
    "run_momentum": (_quadratic, [1.0] * 50, 0.01, 200),
    "run_adam": (_quadratic, [1.0] * 50, 0.01, 200),
    "noisy_grad": (_quadratic, _params, 0.1, 0),
}
