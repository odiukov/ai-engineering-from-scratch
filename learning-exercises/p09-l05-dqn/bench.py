"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_N = 6


def _one_hot(i, size):
    v = [0.0] * size
    v[i] = 1.0
    return v


_net = {
    "W1": [[_rng.gauss(0.0, 0.2) for _ in range(_N)] for _ in range(12)],
    "b1": [0.0] * 12,
    "W2": [[_rng.gauss(0.0, 0.2) for _ in range(12)] for _ in range(2)],
    "b2": [0.0, 0.0],
}
_target = {
    "W1": [row[:] for row in _net["W1"]],
    "b1": _net["b1"][:],
    "W2": [row[:] for row in _net["W2"]],
    "b2": _net["b2"][:],
}
_batch = [
    (_one_hot(p, _N), a, -1.0, _one_hot(min(_N - 1, p + 1) if a else max(0, p - 1), _N),
     (min(_N - 1, p + 1) if a else max(0, p - 1)) == _N - 1)
    for p in range(_N - 1)
    for a in (0, 1)
]
_q_next = [-5.0, -2.0]

BENCH = {
    "one_hot": (3, 64),
    "init_net": (64, 32, 4, _rng),
    "clone_net": (_net,),
    "forward": (_net, _one_hot(0, _N)),
    "dqn_target": (-1.0, 0.99, _q_next, False),
    "double_dqn_target": (-1.0, 0.99, _q_next, _q_next, False),
    "train_step": (_net, _target, _batch, 0.9, 0.0),
}
