"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_labels = [random.randint(0, 3) for _ in range(20000)]

_N, _D = 200, 4
_X = [[round(random.uniform(0, 10), 2) for _ in range(_D)] for _ in range(_N)]
_y = [1 if row[0] + row[1] > 10 else 0 for row in _X]

_left = _labels[:9000]
_right = _labels[9000:]

BENCH = {
    "gini_impurity": (_labels,),
    "entropy": (_labels,),
    "information_gain": (_labels, _left, _right, "gini"),
    "split_dataset": (_X, _y, 0, 5.0),
    "best_split": (_X, _y),
    "build_tree": (_X, _y, 4),
    "tree_predict": ({"leaf": False, "feature": 0, "threshold": 5.0,
                      "left": {"leaf": True, "value": 0},
                      "right": {"leaf": True, "value": 1}}, _X),
    "bootstrap_sample": (_X, _y, 0),
}
