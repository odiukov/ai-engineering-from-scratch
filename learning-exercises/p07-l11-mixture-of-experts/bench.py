"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_E, _K, _D, _H, _N = 16, 2, 32, 64, 200

_hidden = [random.gauss(0, 1) for _ in range(_D)]
_W_router = [[random.gauss(0, 1) for _ in range(_D)] for _ in range(_E)]
_experts = [
    [[random.gauss(0, 0.1) for _ in range(_H)] for _ in range(_D)]
    for _ in range(_E)
]
_tokens = [[random.gauss(0, 1) for _ in range(_D)] for _ in range(_N)]
_scores = [random.gauss(0, 1) for _ in range(_E)]
_bias = [0.0] * _E
_usage = [random.randrange(0, 60) for _ in range(_E)]

BENCH = {
    "router_scores": (_hidden, _W_router),
    "select_experts": (_scores, _bias, _K),
    "gate_weights": (_scores, list(range(_K))),
    "apply_expert": (_hidden, _experts[0]),
    "moe_forward": (_hidden, _experts, _W_router, _K, _bias),
    "expert_usage": (_tokens, _W_router, _K, _bias),
    "update_bias": (_bias, _usage, _N * _K / _E, 0.1),
    "moe_params": (256, 44_000_000, 8, 1),
}
