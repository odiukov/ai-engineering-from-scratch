"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_TASKS = ("chat", "summarize", "code", "math", "planning", "rewrite")

# Двадцать тысяч запросов: run_workload обязана быть линейной. Наивная
# реализация, которая на каждом запросе заново считает базовую линию через
# полный проход по списку, на таком размере уже не вернётся.
_stream = [
    {
        "req_id": f"q{i:05d}",
        "task": _TASKS[i % len(_TASKS)],
        "prompt_tokens": _rng.randrange(120, 9000),
        "output_tokens": _rng.randrange(20, 1200),
        "similarity": _rng.random(),
        "cheap_confidence": _rng.random(),
        "cheap_correct": _rng.random() > 0.2,
    }
    for i in range(20000)
]

_report = {
    "requests": 20000,
    "total_cost_usd": 120.0,
    "baseline_cost_usd": 300.0,
    "saving_usd": 180.0,
    "saving_pct": 60.0,
    "escalation_rate": 0.41,
    "cheap_share": 0.59,
    "accuracy": 0.965,
    "quality_loss_pct": 3.5,
}

BENCH = {
    "make_request": ("q", "chat", 1000, 200, 0.3, 0.6, True),
    "call_cost": ("frontier", 4000, 300),
    "pre_route": (_stream[0],),
    "cascade": (_stream[0], 0.75),
    "cascade_break_even_rate": (4000, 300),
    "route_request": (_stream[0], "cascade", 0.75),
    "run_workload": (_stream, "cascade", 0.75),
    "drift_alarm": (_report,),
}
