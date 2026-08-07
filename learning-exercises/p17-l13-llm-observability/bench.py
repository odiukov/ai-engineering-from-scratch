"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим


def _span(span_id, parent_id, start, end, cost, status="ok"):
    return {
        "span_id": span_id,
        "parent_id": parent_id,
        "name": "llm.chat",
        "start_ms": start,
        "end_ms": end,
        "duration_ms": end - start,
        "cost_usd": cost,
        "status": status,
    }


# Широкое дерево: наивная сборка «для каждого спана пройти весь список»
# заметно медленнее словаря по parent_id.
_wide = [_span("root", None, 0, 20_000, 0.0)]
_wide += [
    _span(f"s{i}", "root", i * 2, i * 2 + 40, _rng.uniform(0.0001, 0.01))
    for i in range(3000)
]

# Глубокая последовательная цепочка на одном уровне: критический путь обязан
# пройти по всем звеньям, а не спуститься в самое позднее и остановиться.
_chain = [_span("root", None, 0, 4000, 0.0)]
_chain += [_span(f"c{i}", "root", i * 2, i * 2 + 2, 0.0) for i in range(2000)]

_traces = [
    [_span(f"t{i}", None, 0, 10, _rng.uniform(0.0, 0.05),
           "error" if i % 50 == 0 else "ok")]
    for i in range(4000)
]

BENCH = {
    "make_span": ("a", None, "agent", 0, 100, 0.001, "ok"),
    "index_by_parent": (_wide,),
    "trace_cost": (_wide,),
    "critical_path": (_chain, "root"),
    "critical_path_ms": (_chain, "root"),
    "keep_trace": (_traces[0], random.Random(1), 0.05, 0.02),
    "sample_traces": (_traces, random.Random(1), 0.05, 0.02),
    "retention_cost": (1_000_000, 0.05, 0.50),
}
