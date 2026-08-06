"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 1000
_N_MINORITY = 60

_y = [1 if i < _N_MINORITY else 0 for i in range(_N)]
_probs = [
    random.betavariate(5, 2) if label else random.betavariate(2, 6) for label in _y
]
_y_pred = [1 if p >= 0.5 else 0 for p in _probs]
_X = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(_N)]
_minority = [row for row, label in zip(_X, _y) if label == 1]

BENCH = {
    "confusion_counts": (_y, _y_pred),
    "precision_recall_f1": (_y, _y_pred),
    "matthews_corrcoef": (_y, _y_pred),
    "class_weights": (_y,),
    "k_nearest": (_minority, 0, 5),
    "smote": (_minority, 200, 5, 0),
    "random_oversample": (_X, _y, 0),
    "best_threshold": (_y, _probs, 0.01),
}
