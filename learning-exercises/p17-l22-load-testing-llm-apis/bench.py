"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_samples = [random.gauss(800.0, 250.0) for _ in range(100000)]

_workload = tuple(
    {
        "prompt_tokens": max(50, int(random.gauss(500, 180))),
        "output_tokens": 100,
        "prefix": f"prefix-{i % 80}",
    }
    for i in range(3000)
)
_arrivals = tuple(i / 20.0 for i in range(3000))

_records = tuple(
    {
        "arrival": i / 20.0,
        "start": i / 20.0,
        "ttft_ms": 80.0 if i % 3 else 800.0,
        "total_ms": 1580.0,
        "wait_ms": 0.0,
        "cache_hit": bool(i % 3),
        "rejected": i % 50 == 0,
    }
    for i in range(100000)
)

_summary = {"ttft_p50": 80.0, "ttft_p95": 800.0, "ttft_p99": 800.0, "reject_rate": 0.02}

BENCH = {
    "percentile": (_samples, 0.99),
    "prompt_lengths": (100000, 500, 150, random.Random(0)),
    "make_workload": (100000, 500, 150, 80, random.Random(0)),
    "arrival_schedule": ("soak", 20000.0, 5.0),
    "run_load": (_workload, _arrivals, 8, 20),
    "summarize": (_records,),
    "apparent_itl": (10.0, 0.5, 50, 8),
    "ci_gate": (_summary, {"ttft_p95": 800.0, "reject_rate": 0.05}),
}
