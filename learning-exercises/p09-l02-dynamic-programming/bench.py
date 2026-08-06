"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_STATES = [(r, c) for r in range(4) for c in range(4)]
_ZEROS = {s: 0.0 for s in _STATES}
_NOISY = {s: _rng.uniform(-20.0, 20.0) for s in _STATES}
_uniform = lambda _state: {"up": 0.25, "down": 0.25, "left": 0.25, "right": 0.25}

BENCH = {
    "transitions": ((1, 1), "down", 0.1),
    "sup_norm": (_ZEROS, _NOISY),
    "q_value": ((1, 1), "down", _NOISY, 0.99, 0.1),
    "bellman_sweep": (_NOISY, 0.99, 0.1),
    "policy_evaluation": (_uniform, 0.99, 0.1),
    "greedy_policy": (_NOISY, 0.99, 0.1),
    "value_iteration": (0.99, 0.1),
    "policy_iteration": (0.99, 0.1),
}
