"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_D = 512
_values = [random.gauss(0.0, 1.0) for _ in range(_D)]
_gamma = [1.0] * _D
_beta = [0.0] * _D
_mask = [random.randint(0, 1) for _ in range(_D)]
_batch = [[random.gauss(0.0, 1.0) for _ in range(_D)] for _ in range(32)]
_losses = [1.0 / (i + 1) + 0.001 * i for i in range(500)]

BENCH = {
    "dropout_mask": (_D, 0.5, 0),
    "apply_dropout": (_values, _mask, 0.5),
    "l2_penalty": (_values, 0.01),
    "l2_gradient": (_values, 0.01),
    "batch_norm": (_batch, _gamma, _beta),
    "layer_norm": (_values, _gamma, _beta),
    "rms_norm": (_values, _gamma),
    "generalization_gap": (0.99, 0.65),
    "early_stop_epoch": (_losses, 10),
}
