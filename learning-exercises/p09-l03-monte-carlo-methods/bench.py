"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_ACTIONS = ("up", "down", "left", "right")
_optimal = lambda state, _rng: "down" if state[0] < 3 else "right"
_uniform = lambda _state, rng: rng.choice(_ACTIONS)
_traj = [((0, 0), "down", -1.0)] * 2000
_row = {"up": -9.0, "down": -1.0, "left": -5.0, "right": -3.0}

BENCH = {
    "grid_step": ((1, 2), "down"),
    "returns_from": (_traj, 0.99),
    "incremental_mean": (1.0, 2.0, 10),
    "rollout": (_uniform, _rng),
    "mc_evaluate": (_uniform, 300, 0.99, _rng),
    "constant_alpha_mc": (_uniform, 300, 0.1, 0.99, _rng),
    "epsilon_greedy_action": (_row, _rng, 0.1),
    "mc_control": (300, 0.99, 0.2, _rng),
}
