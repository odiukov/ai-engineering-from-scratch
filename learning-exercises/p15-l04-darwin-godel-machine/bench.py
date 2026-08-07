"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_ops = ["collapse", "trim", "lower", "upper", "nop", "reverse"]
_agent = {"ops": _ops, "hack_bonus": 0.3}
_archive = {(i % 8 + 1, round(i / 200, 2)): {"ops": _ops[: i % 6 + 1],
                                             "hack_bonus": 0.0}
            for i in range(150)}

BENCH = {
    "apply_ops": (_ops, "  Some   MESSY   text  " * 20),
    "true_score": (_ops,),
    "reported_score": (_agent, False),
    "archive_key": (_agent,),
    "passes_gate": (_agent, 0.1),
    "archive_accept": (_archive, _agent, 0.0),
    "mutate_agent": (_rng, _agent, False),
    "run_dgm": (_rng, 200, False),
}
