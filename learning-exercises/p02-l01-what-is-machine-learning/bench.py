"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 20000
_X = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(_N)]
_y_true = [random.randint(0, 1) for _ in range(_N)]
_y_pred = [random.randint(0, 1) for _ in range(_N)]

BENCH = {
    "train_test_split": (_X, _y_true, 0.2, 0),
    "confusion_counts": (_y_true, _y_pred),
    "accuracy": (_y_true, _y_pred),
    "precision": (_y_true, _y_pred),
    "recall": (_y_true, _y_pred),
    "f1": (_y_true, _y_pred),
    "majority_baseline": (_y_true, _N),
}
