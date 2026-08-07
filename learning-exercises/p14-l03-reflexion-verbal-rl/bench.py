"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_actions = [random.choice("abcdef") for _ in range(2000)]
_memory = [
    {"trial": i, "attempt": [1, 1, 1], "delta": -17, "text": f"промах {i}"}
    for i in range(2000)
]

BENCH = {
    "binary_evaluator": ([1, 2, 3], 20),
    "heuristic_evaluator": (_actions, 5),
    "reflect": ([1, 1, 1], -17),
    "add_reflection": (_memory, {"trial": 9999}, 6),
    "expire_reflections": (_memory, 2000, 50),
    "memory_prompt": (_memory,),
    "actor": (_memory,),
    "run_reflexion": (20, 8, True),
}
