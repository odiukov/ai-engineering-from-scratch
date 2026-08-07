"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# Цепочка из 60 операторов: каждый включает следующий. Наивная реализация,
# пересобирающая состояние с нуля на каждом шаге, здесь заметно просядет.
_CHAIN = 60

_operators = {
    f"step_{i}": {
        "pre": () if i == 0 else (f"done_{i - 1}",),
        "add": (f"done_{i}",),
        "remove": (),
    }
    for i in range(_CHAIN)
}

_methods = {
    "run_all": (
        {"name": "m1", "pre": (), "subtasks": tuple(f"step_{i}" for i in range(_CHAIN))},
    ),
}

_domain = {"operators": _operators, "methods": _methods}
_state = frozenset()
_full_plan = [f"step_{i}" for i in range(_CHAIN)]

_samples = tuple((x, 3 * x + 7) for x in range(-40, 41))
_fitness = lambda ind: sum((ind[0] * x + ind[1] - y) ** 2 for x, y in _samples) * 1.0

BENCH = {
    "applicable": (_operators["step_30"], frozenset({"done_29"})),
    "apply_operator": (_operators["step_0"], _state),
    "decompose": (_methods, "run_all", _state),
    "plan": (_domain, "run_all", _state),
    "execute_plan": (_domain, _full_plan, _state),
    "fitness_linear": ((3, 7), _samples),
    "mutate": ((3, 7), random.Random(0)),
    "evolve": ([(0, 0)], _fitness, lambda ind, rng: tuple(
        v + rng.randint(-2, 2) for v in ind), random.Random(0), 40),
}
