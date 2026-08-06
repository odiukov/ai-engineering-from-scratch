"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_KEY = 1442695040888963407
_THETA = [random.gauss(0.0, 1.0) for _ in range(200)]
_TREE = {"w": [random.gauss(0.0, 1.0) for _ in range(5000)], "b": 0.0}
_BATCH = [[random.gauss(0.0, 1.0) for _ in range(8)] for _ in range(4000)]
_PARAMS = {"w": [1.0] * 8, "b": 0.5}
_XS = [[random.gauss(0.0, 1.0) for _ in range(4)] for _ in range(120)]
_YS = [sum(row) + 1.0 for row in _XS]

_quadratic = lambda p: sum(v * v for v in p)
_double = lambda v: v * 2.0

BENCH = {
    "prng_key": (12345,),
    "split_key": (_KEY, 64),
    "normal": (_KEY, 20000, 1.0),
    "tree_map": (_double, _TREE),
    "grad": (_quadratic,),
    "value_and_grad": (_quadratic,),
    "vmap": (sum,),
    "predict": (_PARAMS, [1.0] * 8),
    "mse": (_PARAMS, _BATCH, [0.0] * 4000),
    "train_linear": (_KEY, _XS, _YS, 30, 0.1),
}
