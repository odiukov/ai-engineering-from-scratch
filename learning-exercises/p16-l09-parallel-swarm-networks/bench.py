"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_N_AGENTS = 120
_EDGES = [(i, j) for i in range(_N_AGENTS) for j in range(i + 1, _N_AGENTS)]

_DURATIONS = [random.choice((0.1, 0.4, 1.0, 2.5)) for _ in range(4000)]
_ASSIGNMENT = [random.randrange(8) for _ in _DURATIONS]
_COUNTS = {w: _ASSIGNMENT.count(w) for w in range(8)}

_TASKS = [
    {"id": f"t-{i}", "priority": random.randrange(1, 6), "arrival": random.randrange(0, 500)}
    for i in range(3000)
]

BENCH = {
    "build_topology": (_N_AGENTS, "mesh"),
    "channel_count": (_N_AGENTS, "mesh"),
    "is_connected": (_N_AGENTS, _EDGES),
    "simulate_fixed": (_DURATIONS, _ASSIGNMENT, 8),
    "simulate_swarm": (_DURATIONS, 8),
    "speedup": (2.0, 0.5),
    "hot_spot_ratio": (_COUNTS,),
    "aging_order": (_TASKS, 600, 0.05),
}
