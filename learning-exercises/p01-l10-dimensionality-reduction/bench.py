"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 300
_D = 20

# данные с настоящей структурой: 20 признаков, порождённых 5 факторами
_loadings = [[random.gauss(0, 1) for _ in range(5)] for _ in range(_D)]
_X = []
for _ in range(_N):
    factors = [random.gauss(0, 3) for _ in range(5)]
    _X.append(
        [
            sum(f * w for f, w in zip(factors, row)) + random.gauss(0, 0.1)
            for row in _loadings
        ]
    )

_means = [sum(col) / _N for col in zip(*_X)]
_cols = [[row[i] - _means[i] for row in _X] for i in range(_D)]
_cov = [
    [sum(a * b for a, b in zip(_cols[i], _cols[j])) / (_N - 1) for j in range(_D)]
    for i in range(_D)
]

_components = [[1.0 if i == j else 0.0 for j in range(_D)] for i in range(3)]

BENCH = {
    "column_means": (_X,),
    "center": (_X,),
    "covariance_matrix": (_X,),
    "power_iteration": (_cov,),
    "top_components": (_cov, 3),
    "explained_variance_ratio": (_cov, 3),
    "project": (_X, _components),
    "reconstruction_error": (_X, 3),
}
