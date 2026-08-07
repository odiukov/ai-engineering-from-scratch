"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_latencies = [random.expovariate(1 / 200.0) for _ in range(20000)]
_requests = [
    {
        "ttft_ms": random.expovariate(1 / 200.0),
        "tpot_ms": max(1.0, random.gauss(8.0, 2.0)),
        "output_tokens": random.randint(50, 300),
    }
    for _ in range(20000)
]
_slo = {"ttft_ms": 500.0, "tpot_ms": 15.0, "e2e_ms": 2000.0}

BENCH = {
    "ttft_ms": (40.0, 12.0, 110.0),
    "e2e_ms": (162.0, 7.33, 127),
    "itl_ms": (500.0, 700.0, 100, "genai-perf"),
    "percentile": (_latencies, 99),
    "latency_summary": (_latencies,),
    "throughput_tokens_per_s": (_requests, 60.0),
    "goodput": (_requests, _slo),
    "slo_breakdown": (_requests, _slo),
}
