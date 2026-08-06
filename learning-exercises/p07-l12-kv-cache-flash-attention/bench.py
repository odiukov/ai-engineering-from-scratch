"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_d = 32
_n = 1500
_q = [random.gauss(0, 1) for _ in range(_d)]
_K = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n)]
_V = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n)]
_scores = [random.gauss(0, 3) for _ in range(_n)]

_steps = 70
_states = [0.01 * i for i in range(1, _steps + 1)]
_queries = [[random.gauss(0, 1) for _ in range(2)] for _ in range(_steps)]


def _project(state):
    return [state, state * state], [state + 1.0, 1.0 - state]


BENCH = {
    "softmax": (_scores,),
    "attention_full": (_q, _K, _V),
    "tiled_softmax_dot": (_q, _K, _V, 64),
    "decode_naive": (_states, _project, _queries),
    "decode_cached": (_states, _project, _queries),
    "kv_cache_bytes": (131072, 80, 8, 128, 2),
}
