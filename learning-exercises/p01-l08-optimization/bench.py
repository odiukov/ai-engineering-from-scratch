"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 500
_params = [random.uniform(-1, 1) for _ in range(_DIM)]
_grads = [random.uniform(-1, 1) for _ in range(_DIM)]
_zeros = [0.0] * _DIM

# квадратичная задача: градиент считается быстро, вся цена — в шагах
_quadratic_grad = lambda p: [2 * x for x in p]

BENCH = {
    "rosenbrock": ([-1.0, 1.0],),
    "rosenbrock_gradient": ([-1.0, 1.0],),
    "sgd_momentum_step": (_params, _grads, _zeros, 0.01, 0.9),
    "adam_step": (_params, _grads, _zeros, _zeros, 1),
    "minimize_momentum": (_quadratic_grad, _params[:50], 0.01, 0.9, 200),
    "minimize_adam": (_quadratic_grad, _params[:50], 0.01, 200),
    "exponential_decay": (0.1, 1000),
    "cosine_annealing": (0.1, 0.0, 500, 1000),
}
