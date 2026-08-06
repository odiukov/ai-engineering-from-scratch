"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 2000

# тренд + недельная сезонность + шум: похоже на настоящие дневные метрики
_series = [
    0.02 * i + 3.0 * math.sin(2 * math.pi * i / 7) + random.gauss(0, 0.5)
    for i in range(_N)
]

_X, _y = [], []
for _t in range(10, _N):
    _X.append(_series[_t - 10 : _t])
    _y.append(_series[_t])

BENCH = {
    "difference": (_series,),
    "is_stationary": (_series,),
    "autocorrelation": (_series, 20),
    "rolling_mean": (_series, 14),
    "make_lag_features": (_series, 10),
    "time_split": (_X, _y, 400),
    "walk_forward_splits": (_N, 5, 500),
    "seasonal_naive_forecast": (_series, 7, 500),
}
