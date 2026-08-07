"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# Тысяча заданий: submit из наивного цикла квадратичен по проверке дублей,
# и на таком размере это уже заметно.
_queue = [
    {"job_id": f"job_{i:04d}",
     "n_requests": _rng.randrange(1_000, 60_000),
     "submitted_h": _rng.uniform(0.0, 48.0)}
    for i in range(1000)
]

_completions = [
    {"job_id": f"job_{i:04d}", "finished_h": 20.0 + i % 30, "wait_h": float(i % 40)}
    for i in range(20000)
]

BENCH = {
    "sync_cost": (50_000, 4000, 2000, 200),
    "cached_cost": (50_000, 4000, 2000, 200),
    "batch_cost": (50_000, 4000, 2000, 200, True),
    "submit": (_queue, "new-job", 10_000, 12.0),
    "drain_window": (_queue, 0, 6, 10_000),
    "sla_report": (_completions,),
    "triage": (600,),
    "lane_decision": (50_000, 4000, 200, 100, 86_400),
}
