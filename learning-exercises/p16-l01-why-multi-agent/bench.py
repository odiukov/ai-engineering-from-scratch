"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_messages = [
    {"from": f"a{i % 7}", "to": f"a{(i + 1) % 7}", "content": "x" * 20}
    for i in range(4000)
]
_outputs = [random.randint(100, 900) for _ in range(2000)]
_seconds = [random.uniform(0.1, 2.0) for _ in range(2000)]
_sizes = [i * 50 for i in range(2000)]

BENCH = {
    "inbox": (_messages, "a3"),
    "single_agent_context": (1000, _outputs),
    "multi_agent_contexts": (1000, _outputs),
    "first_overflow": (_sizes, 10**9),
    "pipeline_seconds": (_seconds, 0.1),
    "fanout_seconds": (_seconds, 0.1),
    "coordination_overhead": (1000, _outputs, 100),
    "recommend_topology": (1000, _outputs, 5, False),
}
