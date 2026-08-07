"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_pool = [
    "role_ambiguity", "task_underspecified", "implicit_success_criteria",
    "state_drift", "lost_message", "unsynchronized_write",
    "unchecked_output", "memory_poisoning", "missing_regression_tests",
]

# 4000 трасс по 1-4 симптома — размер квартального аудита из урока
_incidents = [
    random.sample(_pool, random.randint(1, 4)) for _ in range(4000)
]
_rates = {"spec": 0.4177, "coord": 0.3694, "verify": 0.2130}
_mitigations = [
    ("m-%d" % i, random.sample(["spec", "coord", "verify"], random.randint(1, 3)))
    for i in range(300)
]
_opinions = [random.uniform(0, 1) for _ in range(20000)]
_rounds = [
    ([random.gauss(0.5, 0.4 / (r + 1)) for _ in range(50)],
     [0.4 + 0.01 * r] * 50)
    for r in range(400)
]
_results = [random.random() > 0.08 for _ in range(20000)]

BENCH = {
    "classify_incident": (_incidents[0],),
    "category_rates": (_incidents,),
    "rank_mitigations": (_rates, _mitigations),
    "opinion_spread": (_opinions,),
    "detect_groupthink": (_rounds,),
    "circuit_state": (_results, 0.05, 100),
    "cascade_load": (100.0, 0.1, 3, 2000),
    "audit": (_incidents, _mitigations, 3),
}
