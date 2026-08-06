"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_pred = [random.uniform(0.0, 1.0) for _ in range(2000)]
_target = [random.choice([0.0, 1.0]) for _ in range(2000)]

# плоский список троек (values, index, grads) — ровно то, что видит оптимизатор
_values = [random.gauss(0.0, 1.0) for _ in range(5000)]
_grads = [random.gauss(0.0, 1.0) for _ in range(5000)]
_params = [(_values, i, _grads) for i in range(5000)]

BENCH = {
    "mse_loss": (_pred, _target),
    "mse_grad": (_pred, _target),
    "zero_grads": (_params,),
    "sgd_step": (_params, 0.01),
    "xor_dataset": (),
    "train_xor": (0, 8, 200, 0.3),
}
