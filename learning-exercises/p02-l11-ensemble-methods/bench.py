"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_X = [[random.uniform(-3, 3), random.uniform(-3, 3)] for _ in range(120)]
_y = [1 if 0.0 < x[0] + x[1] < 3.0 else -1 for x in _X]
_weights = [1.0 / len(_X)] * len(_X)
_ensemble = [
    ({"feature": i % 2, "threshold": float(i) / 4, "polarity": 1 if i % 3 else -1}, 1.0)
    for i in range(20)
]

BENCH = {
    "majority_vote": ([1, -1] * 50, [1.0] * 100),
    "vote_accuracy": (0.6, 101),
    "bootstrap_indices": (2000, 0),
    "fit_stump": (_X, _y, _weights),
    "predict_stump": (_ensemble[0][0], _X[0]),
    "predict_ensemble": (_ensemble, _X[0]),
    "fit_bagging": (_X, _y, 5, 0),
    "fit_adaboost": (_X, _y, 5),
}
