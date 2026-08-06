"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N = 40  # состояний в цепи


def _random_stochastic_matrix(n):
    """Матрица со случайными строками, каждая нормирована в сумму 1."""
    rows = []
    for _ in range(n):
        row = [random.random() + 0.01 for _ in range(n)]
        total = sum(row)
        rows.append([x / total for x in row])
    return rows


_P = _random_stochastic_matrix(_N)
_dist = [1.0 / _N] * _N
_small_P = [[0.9, 0.1], [0.5, 0.5]]
_states = [random.randrange(_N) for _ in range(20000)]
_grad = lambda x: x - 3.0  # noqa: E731 — U(x) = (x - 3)^2 / 2

BENCH = {
    "step_distribution": (_dist, _P),
    "distribution_after_n_steps": (_dist, _P, 30),
    "stationary_distribution": (_P,),
    "empirical_distribution": (_states, _N),
    "simulate_markov_chain": (_small_P, 0, 20000, random.Random(0)),
    "random_walk_1d": (50000, random.Random(0)),
    "brownian_motion": (20000, 0.001, random.Random(0)),
    "langevin_dynamics": (_grad, 0.0, 0.01, 1.0, 20000, random.Random(0)),
}
