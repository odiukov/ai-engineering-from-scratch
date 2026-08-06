"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N, _D = 400, 5
_X = []
_y = []
for _ in range(_N // 2):
    _X.append([random.gauss(-1, 1) for _ in range(_D)])
    _y.append(0)
for _ in range(_N // 2):
    _X.append([random.gauss(1, 1) for _ in range(_D)])
    _y.append(1)

_w = [0.5] * _D
_probs = [random.random() for _ in range(_N)]
_scores = [random.uniform(-5, 5) for _ in range(200)]

BENCH = {
    "sigmoid": (0.7,),
    "predict_proba": (_X, _w, 0.0),
    "predict_labels": (_probs, 0.5),
    "binary_cross_entropy": (_y, _probs),
    "logistic_gradients": (_X, _y, _w, 0.0),
    "fit_logistic": (_X, _y, 0.1, 40),
    "softmax": (_scores,),
}
