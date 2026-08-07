"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_answers = ["4.2% improvement", "the study reports 4.2%", "42", "42.0", "7"]
_votes = [
    (random.choice(_answers), round(random.uniform(0.1, 0.99), 2))
    for _ in range(4000)
]
_sample = [random.gauss(0.0, 1.0) for _ in range(2000)] + [500.0]

BENCH = {
    "max_faulty": (100,),
    "quorum_size": (100,),
    "canonicalize": ("the study reports 4.2% improvement",),
    "cluster_votes": (_votes,),
    "plurality": (_votes,),
    "weighted_consensus": (_votes, 0.5),
    "geometric_median": (_sample,),
    "simulate_bft": (7, 2, random.Random(0), 200, 0.1),
}
