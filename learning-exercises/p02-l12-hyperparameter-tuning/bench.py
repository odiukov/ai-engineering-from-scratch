"""Входные данные для замера скорости."""

import math
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_space = {
    "learning_rate": ("log_float", 0.001, 1.0),
    "max_depth": ("int", 2, 8),
    "subsample": ("float", 0.5, 1.0),
}
_grid = {
    "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
    "subsample": [0.5, 0.75, 1.0],
}


def _objective(config):
    return (
        -(math.log10(config["learning_rate"]) + 2) ** 2
        - (config["max_depth"] - 4) ** 2
        - 4 * (config["subsample"] - 0.8) ** 2
        + 10
    )


_history = [
    ({"learning_rate": 10 ** -random.uniform(0, 3), "max_depth": random.randint(2, 8)}, 0.0)
    for _ in range(2000)
]
_observations = [
    ([random.random(), random.random()], random.random()) for _ in range(200)
]

BENCH = {
    "log_uniform": (0.001, 1.0, 0.42),
    "sample_config": (_space, 7),
    "grid_search": (_objective, _grid),
    "random_search": (_objective, _space, 200, 0),
    "count_unique": (_history, "max_depth"),
    "expected_improvement": (4.0, 1.5, 5.0),
    "surrogate": (_observations, [0.5, 0.5]),
    "bayes_search": (_objective, _space, 20),
}
