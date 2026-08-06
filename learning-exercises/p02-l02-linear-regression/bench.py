"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N, _D = 400, 6
_X = [[random.uniform(-3, 3) for _ in range(_D)] for _ in range(_N)]
_w = [1.5, -2.0, 0.5, 3.0, -1.0, 0.25]
_y = [
    sum(wj * xj for wj, xj in zip(_w, row)) + 4.0 + random.gauss(0, 0.5) for row in _X
]

_x_flat = [row[0] for row in _X]
_y_pred = [v + random.gauss(0, 0.3) for v in _y]

BENCH = {
    "predict": (_X, _w, 4.0),
    "mse": (_y, _y_pred),
    "gradients": (_X, _y, _w, 4.0),
    "fit_gradient_descent": (_X, _y, 0.01, 60),
    "fit_closed_form": (_x_flat, _y),
    "r_squared": (_y, _y_pred),
    "standardize": (_X,),
    "fit_ridge": (_X, _y, 0.01, 60, 0.1),
}
