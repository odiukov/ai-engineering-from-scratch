"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_flags = (
    "has_typed_state",
    "has_roles",
    "has_dialogue",
    "has_parallel_fanout",
    "needs_resume",
    "needs_human_approval",
)

_problem = {
    "llm_calls": 8,
    "has_typed_state": True,
    "has_roles": False,
    "has_dialogue": False,
    "has_parallel_fanout": True,
    "needs_resume": True,
    "needs_human_approval": False,
}

_cases = [
    dict({f: random.random() < 0.4 for f in _flags}, llm_calls=random.randint(1, 12))
    for _ in range(200)
]

BENCH = {
    "normalize_problem": (_problem,),
    "shape_of": (_problem,),
    "hard_filter": (_problem,),
    "score": ("langgraph", _problem),
    "pick_framework": (_problem,),
    "routing_cost_per_run": ("crewai", 20, 5.0, 15.0),
    "compare_run_cost": (_cases[0], 20, 5.0, 15.0),
}
