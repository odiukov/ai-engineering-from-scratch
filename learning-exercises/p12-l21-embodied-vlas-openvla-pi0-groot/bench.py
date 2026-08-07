"""Входные данные для замера скорости."""

import math
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# 10-DOF рука, горизонт 64 шага — как раз то, что жмёт FAST
_dof = 10
_horizon = 64
_trajectory = [
    [0.9 * math.sin((i + 3 * d) / 9.0) for d in range(_dof)]
    for i in range(_horizon)
]
_action = _trajectory[0]
_tokens = [random.randrange(256) for _ in range(_dof * 4)]
_series = [0.9 * math.sin(i / 7.0) for i in range(_horizon)]
_coeffs = [random.gauss(0, 1) for _ in range(_horizon)]

BENCH = {
    "discretize": (_action, 256),
    "undiscretize": (_tokens, 256),
    "dct": (_series,),
    "idct": (_coeffs,),
    "fast_tokens": (_trajectory, 4),
    "fast_reconstruct": (_tokens, _horizon, _dof, 4),
    "format_token_budget": (_dof, _horizon),
    "cofinetune_mix": (list("abcdefgh"), list("XYZ"), 1.0, 5000, random.Random(0)),
}
