"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 150
_N_INFORMATIVE = 3
_N_NOISE = 7

_X, _y = [], []
for _ in range(_N):
    _signal = [random.gauss(0, 1) for _ in range(_N_INFORMATIVE)]
    _noise = [random.gauss(0, 1) for _ in range(_N_NOISE)]
    _X.append(_signal + _noise)
    _y.append(1 if 2 * _signal[0] - _signal[1] + _signal[2] > 0 else 0)

_column = [row[0] for row in _X]
_scores = [random.random() for _ in range(_N_INFORMATIVE + _N_NOISE)]

BENCH = {
    "variance_threshold": (_X, 0.01),
    "correlation": (_column, [float(v) for v in _y]),
    "discretize": (_column, 10),
    "mutual_information": (_X, _y, 10),
    "select_k_best": (_scores, 4),
    "logistic_weights": (_X, _y, 0.0, 0.1, 40),
    "rfe": (_X, _y, 4, 0.1, 20),
    "l1_select": (_X, _y, 0.5, 0.1, 40),
}
