"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_d = 120                                  # игрушечный аналог d = 640 у SDXL
_r = 8

_W = [[_rng.gauss(0, 0.1) for _ in range(_d)] for _ in range(_d)]
_A = [[_rng.gauss(0, 0.2) for _ in range(_d)] for _ in range(_r)]
_B = [[_rng.gauss(0, 0.2) for _ in range(_r)] for _ in range(_d)]
_x = [_rng.gauss(0, 1) for _ in range(_d)]
_target = [_rng.gauss(0, 1) for _ in range(_d)]

_rank_M = [[_rng.gauss(0, 1) for _ in range(80)] for _ in range(80)]

_base = [_rng.gauss(0, 1) for _ in range(4000)]
_sides = [[_rng.gauss(0, 1) for _ in range(4000)] for _ in range(3)]
_gates = [0.4, 0.3, 0.3]

BENCH = {
    "matvec": (_W, _x),
    "lora_delta": (_A, _B, 1.0),
    "lora_forward": (_W, _A, _B, _x, 1.0),
    "merge_lora": (_W, _A, _B, 1.0),
    "matrix_rank": (_rank_M,),
    "lora_param_count": (640, 640, 16),
    "lora_grads": (_W, _A, _B, _x, _target, 1.0),
    "apply_controls": (_base, _sides, _gates),
}
