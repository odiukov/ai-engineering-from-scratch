"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_budgets = [10.0 ** (18.0 + 0.5 * i) for i in range(16)]

BENCH = {
    "chinchilla_loss": (70e9, 1400e9),
    "compute_flops": (70e9, 1400e9),
    "tokens_for_budget": (5.88e23, 70e9),
    "compute_optimal": (1e23, 2000),
    "optimal_exponents": (0.34, 0.28),
    "overtraining_tradeoff": (1e24, 10.0),
    "min_compute_for_loss": (2.0,),
    "emergence_curves": (_budgets, 2.0),
}
