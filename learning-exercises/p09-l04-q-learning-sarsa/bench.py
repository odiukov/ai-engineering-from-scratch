"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_ACTIONS = ("up", "down", "left", "right")
_row = {"up": -9.0, "down": -1.0, "left": -5.0, "right": -3.0}
_probs = {a: 0.25 for a in _ACTIONS}

BENCH = {
    "grid_step": ((1, 2), "down"),
    "td_error": (-1.0, 0.99, -5.0, -6.0),
    "epsilon_greedy_action": (_row, _rng, 0.1),
    "bootstrap_q_learning": (_row, False),
    "bootstrap_sarsa": (_row, "down", False),
    "bootstrap_expected_sarsa": (_row, _probs, False),
    "q_learning": (300, 0.1, 0.99, 0.1, _rng),
    "sarsa": (300, 0.1, 0.99, 0.1, _rng),
}
