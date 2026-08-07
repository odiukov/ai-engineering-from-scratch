"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# 1865 задач — размер SWE-bench Pro; 8 агентов в одиночных прогонах.
_n = 1865
_seen = [random.random() < 0.72 for _ in range(_n)]
_held = [random.random() < 0.23 for _ in range(_n)]
_milestones = [random.random() < 0.6 for _ in range(_n)]
_weights = [random.uniform(0.5, 3.0) for _ in range(_n)]
_team = [1 if random.random() < 0.5 else 0 for _ in range(_n)]
_solos = [[1 if random.random() < 0.4 else 0 for _ in range(_n)] for _ in range(8)]
_scores = [random.gauss(0.5, 0.2) for _ in range(_n)]

_system = {
    "seen": _seen,
    "held": _held,
    "milestones": _milestones,
    "milestone_weights": _weights,
    "tokens": 1_200_000,
    "price_per_1k": 0.01,
    "n_options": 4,
    "team": _team,
    "solos": _solos,
}

BENCH = {
    "accuracy": (_held,),
    "milestone_score": (_milestones, _weights),
    "lift_over_random": (0.23, 4),
    "coordination_gain": (_team, _solos),
    "cost_per_milestone": (1_200_000, 0.6, 0.01),
    "contamination_gap": (_seen, _held),
    "mean_confidence_interval": (_scores,),
    "scorecard": (_system,),
}
