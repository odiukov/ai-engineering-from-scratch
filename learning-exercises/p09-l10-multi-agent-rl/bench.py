"""Входные данные для замера скорости."""

import itertools
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ACTIONS = ("up", "down", "left", "right")
_row = {a: random.gauss(0.0, 1.0) for a in _ACTIONS}
_next_row = {a: random.gauss(0.0, 1.0) for a in _ACTIONS}
_rng = random.Random(0)

_joint = list(itertools.product(_ACTIONS, repeat=2))
_joint_row = {ja: random.gauss(0.0, 1.0) for ja in _joint}
_agent_probs = {a: 0.25 for a in _ACTIONS}

BENCH = {
    "move": ((1, 1), "up"),
    "joint_step": (((0, 0), (3, 0)), ("down", "right")),
    "joint_actions": (_ACTIONS, 3),
    "epsilon_greedy": (_row, _rng, 0.15),
    "q_learning_update": (_row, "up", -1.0, _next_row),
    "train_independent_q": (400,),
    "train_joint_q": (400,),
    "counterfactual_advantage": (_joint_row, ("up", "down"), 0, _agent_probs),
}
