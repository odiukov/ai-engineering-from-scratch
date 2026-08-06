"""Входные данные для замера скорости."""

import math
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_f = lambda x: math.sin(1.5 * x) + 0.5 * x
_xs = [random.uniform(-3.0, 3.0) for _ in range(400)]
_ys = [_f(x) + random.gauss(0, 0.5) for x in _xs]
_coeffs = [0.1, 0.5, -0.2, 0.05, 0.01]
_pred = [_f(x) for x in _xs]

BENCH = {
    "polyval": (_coeffs, 2.5),
    "fit_polynomial": (_xs, _ys, 5),
    "mean_squared_error": (_ys, _pred),
    "make_dataset": (_f, 400, 0.5, 1),
    "bias_variance_decomposition": (_f, 4, 30, 40),
    "best_degree": (_f, [1, 3, 5], 30, 30),
    "learning_curve": (_f, 3, [20, 40, 80], 10),
    "diagnose": (0.2, 0.9),
}
