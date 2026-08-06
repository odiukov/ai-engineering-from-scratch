"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N_ROWS = 300
_N_COLS = 4

# 3% строк — настоящие выбросы, остальное обычное облако
_rows = [
    [random.gauss(0, 1) for _ in range(_N_COLS)]
    if random.random() > 0.03
    else [random.gauss(0, 1) + 12 for _ in range(_N_COLS)]
    for _ in range(_N_ROWS)
]
_column = [row[0] for row in _rows]
_rng = random.Random(0)
_tree = {
    "feature": 0,
    "threshold": 0.0,
    "left": {"size": 100},
    "right": {
        "feature": 1,
        "threshold": 0.0,
        "left": {"size": 60},
        "right": {"size": 40},
    },
}

BENCH = {
    "zscore_flags": (_rows, 3.0),
    "percentile": (_column, 25),
    "iqr_bounds": (_column, 1.5),
    "iqr_flags": (_rows, 1.5),
    "expected_path_length": (256,),
    "build_isolation_tree": (_rows, 8, _rng),
    "path_length": (_tree, _rows[0]),
    "isolation_scores": (_rows, 20, 32, 0),
}
