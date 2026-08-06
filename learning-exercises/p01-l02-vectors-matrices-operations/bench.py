"""Входные данные для замера скорости."""

import random

random.seed(0)

_A = [[random.uniform(-1, 1) for _ in range(60)] for _ in range(60)]
_B = [[random.uniform(-1, 1) for _ in range(60)] for _ in range(60)]
_S = [[1.0] * 60 for _ in range(60)]

BENCH = {
    "transpose": (_A,),
    "matmul": (_A, _B),
    "identity": (60,),
    "trace": (_A,),
    "is_symmetric": (_S,),
    "hadamard": (_A, _B),
}
