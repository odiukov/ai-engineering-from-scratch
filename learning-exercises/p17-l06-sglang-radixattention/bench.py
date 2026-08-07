"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)          # обязательно: замер должен быть воспроизводим

_system = tuple(range(1000, 1600))          # 600 токенов общего системного промпта
_contexts = [tuple(range(10000 * (c + 1), 10000 * (c + 1) + 400)) for c in range(6)]
_a = _system + _contexts[0] + tuple(range(90000, 90040))
_b = _system + _contexts[0] + tuple(range(91000, 91040))

# 300 запросов: общий системный промпт, шесть веток документов, уникальный хвост.
_workload = tuple(
    _system + _contexts[_rng.randrange(6)] + tuple(range(50000 + 100 * i, 50040 + 100 * i))
    for i in range(300)
)

_parts = {"system": _system, "context": _contexts[0], "question": _a[-40:]}

BENCH = {
    "common_prefix_len": (_a, _b),
    "render_prompt": (("system", "context", "question"), _parts),
    "prefill_speedup": (0.844,),
    "run_workload": (_workload, 20000, "cache_aware"),
}
