"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_ACTIONS = ("up", "down", "left", "right")
_uniform = lambda _state: {a: 0.25 for a in _ACTIONS}
_V = {(r, c): -float(6 - r - c) for r in range(4) for c in range(4)}
_Q = {s: {a: -float(i) for i, a in enumerate(_ACTIONS)} for s in _V}
_rewards = [-1.0] * 2000

BENCH = {
    "grid_step": ((1, 2), "down"),
    "discounted_return": (_rewards, 0.99),
    "effective_horizon": (0.99,),
    "sample_action": ({a: 0.25 for a in _ACTIONS}, _rng),
    "rollout": (_uniform, _rng),
    "policy_evaluation": (_uniform, 0.99),
    "q_from_v": (_V, 0.99),
    "greedy_from_q": (_Q,),
}
