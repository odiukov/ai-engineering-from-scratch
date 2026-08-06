"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_D = 8

_A = [[_rng.gauss(0, 1) for _ in range(_D)] for _ in range(_D)]
_B = [[_rng.gauss(0, 1) for _ in range(_D)] for _ in range(_D)]
# симметричная положительно определённая: A * A^T + сдвиг диагонали
_PSD = [[sum(_A[i][k] * _A[j][k] for k in range(_D)) + (_D if i == j else 0.0)
         for j in range(_D)] for i in range(_D)]

_real = [[_rng.gauss(0.0, 1.0) for _ in range(_D)] for _ in range(1500)]
_gen = [[_rng.gauss(0.4, 1.0) for _ in range(_D)] for _ in range(1500)]

BENCH = {
    "matmul": (_A, _B),
    "matrix_inverse": (_PSD,),
    "matrix_sqrt": (_PSD,),
    "mean_vector": (_real,),
    "covariance": (_real,),
    "fid": (_real, _gen),
    "clip_score": (_real[0], _gen[0]),
    "elo_update": (1000.0, 1000.0, "a"),
}
