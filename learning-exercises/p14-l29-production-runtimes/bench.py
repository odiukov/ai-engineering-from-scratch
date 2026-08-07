"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_N = 400

_pending = [
    {"id": f"j{i:04d}", "payload": f"job payload {i}", "attempt": 0} for i in range(_N)
]

_store = {
    "counters": {"emails": _N},
    "applied": [f"j{i:04d}" for i in range(_N)],
}

_records = [
    {
        "id": f"j{i:04d}",
        "enqueued_at": i,
        "finished_at": i + random.randint(1, 40) if i % 4 else None,
        "status": "done" if i % 4 else "pending",
    }
    for i in range(_N)
]

_latencies = [random.randint(1, 5000) for _ in range(5000)]
_steps = [f"step{i:04d}" for i in range(_N)]
_checkpoint = {
    "completed": _steps[: _N // 2],
    "results": {name: name.upper() for name in _steps[: _N // 2]},
}
_schedule = {f"cron{i:03d}": random.randint(5, 240) for i in range(_N)}
_last_run = {name: random.randint(0, 1000) for name in _schedule}

BENCH = {
    "pick_runtime_shape": (600, False, False, False, True),
    "enqueue": (_pending, "j0007", "duplicate payload"),
    "run_worker": (_pending, str.upper, 3),
    "apply_once": (_store, "j0399", "emails", 1),
    "percentile": (_latencies, 95),
    "queue_metrics": (_records, _N + 100),
    "resume_from_checkpoint": (_steps, _checkpoint, str.upper),
    "cron_due": (_schedule, _last_run, 2000),
}
