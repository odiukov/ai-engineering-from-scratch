"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N, _D = 400, 6
_X = []
_y = []
for _ in range(_N // 2):
    _X.append([random.gauss(-1.5, 1) for _ in range(_D)])
    _y.append(-1)
for _ in range(_N // 2):
    _X.append([random.gauss(1.5, 1) for _ in range(_D)])
    _y.append(1)

_w = [0.4] * _D
_a = [random.gauss(0, 1) for _ in range(_D)]
_b_vec = [random.gauss(0, 1) for _ in range(_D)]

BENCH = {
    "decision_function": (_X, _w, 0.0),
    "hinge_loss": (_X, _y, _w, 0.0),
    "hinge_gradients": (_X, _y, _w, 0.0, 0.01),
    "fit_linear_svm": (_X, _y, 0.05, 40, 0.01),
    "svm_predict": (_X, _w, 0.0),
    "find_support_vectors": (_X, _y, _w, 0.0),
    "polynomial_kernel": (_a, _b_vec, 3, 1.0),
    "rbf_kernel": (_a, _b_vec, 0.5),
}
