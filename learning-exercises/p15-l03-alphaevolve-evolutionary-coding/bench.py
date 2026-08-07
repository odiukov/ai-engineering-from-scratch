"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_target = lambda x: 2.0 * x * x + 3.0 * x - 1.0
_train = [i / 4.0 for i in range(-20, 21)]
_holdout = [i / 4.0 + 0.125 for i in range(-20, 21)]

# Дерево поглубже, чтобы рекурсия была заметна на фоне накладных расходов.
_deep = ("x",)
for _ in range(12):
    _deep = ("add", ("mul", _deep, ("num", 1.0)), ("x",))

_archive = {}
_expr = ("x",)
for _i in range(200):
    _archive[(min(1 + _i % 6, 6), _i % 5)] = (_expr, float(_i))

BENCH = {
    "evaluate_expr": (_deep, 1.5),
    "depth": (_deep,),
    "mse": (_deep, _train, _target),
    "cell_key": (_deep,),
    "mutate": (_rng, _deep),
    "archive_insert": (_archive, _deep, 0.5),
    "best_of": (_archive,),
    "evolve": (_rng, ("x",), 300, _train, _holdout, _target),
}
