"""Входные данные для замера скорости."""

import math
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_cities = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(12)]
_dist = [[math.dist(a, b) for b in _cities] for a in _cities]
_tour = list(range(12))

_swarm = [
    {
        "x": [random.uniform(-5, 5), random.uniform(-5, 5)],
        "v": [random.uniform(-1, 1), random.uniform(-1, 1)],
        "p_best": [0.0, 0.0],
        "p_best_fit": 1e9,
    }
    for _ in range(60)
]

_rastrigin = lambda p: 20.0 + sum(x * x - 10.0 * math.cos(2 * math.pi * x) for x in p)

_pheromone = [[0.0 if i == j else 1.0 for j in range(12)] for i in range(12)]
_deposits = [(_tour, 0.5)] * 20

BENCH = {
    "rastrigin": ([0.31] * 40,),
    "pso_velocity": ([0.1] * 40, [0.2] * 40, [0.3] * 40, [0.4] * 40,
                     0.729, 1.494, 1.494, 0.3, 0.7),
    "pso_step": (_swarm, [0.0, 0.0], _rastrigin, [(-5.12, 5.12)] * 2,
                 0.729, 1.494, 1.494, random.Random(0)),
    "run_pso": (_rastrigin, [(-5.12, 5.12)] * 2, 20, 40, random.Random(0)),
    "tour_length": (_tour, _dist),
    "transition_probabilities": (_pheromone[0], _dist[0], list(range(1, 12))),
    "update_pheromone": (_pheromone, _deposits, 0.5),
    "run_aco": (_dist, 10, 15, random.Random(0)),
}
