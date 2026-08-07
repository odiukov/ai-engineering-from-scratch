"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

MODEL_PRICING = {
    "claude-sonnet-5": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
FALLBACK_CHAIN = ("claude-sonnet-5", "gpt-4o", "gpt-4o-mini")

_text = " ".join(f"word{i}" for i in range(2000))
_latencies = [random.uniform(50, 900) for _ in range(5000)]
_logs = [
    {
        "model": random.choice(("gpt-4o", "gpt-4o-mini", "cache")),
        "input_tokens": random.randint(200, 4000),
        "output_tokens": random.randint(20, 800),
        "latency_ms": random.uniform(50, 900),
        "cache_hit": random.random() < 0.3,
        "error": None,
    }
    for _ in range(3000)
]
# журнал строится случайно, но кэш-попадание обязано быть согласовано с моделью
for _e in _logs:
    if _e["cache_hit"]:
        _e["model"] = "cache"
    elif _e["model"] == "cache":
        _e["cache_hit"] = True

_ok = lambda model, attempt: "ok"

BENCH = {
    "estimate_tokens": (_text,),
    "request_cost": ("gpt-4o", 1500, 400, MODEL_PRICING),
    "backoff_delay": (4, 1.0, 10.0, random.Random(0)),
    "retry_with_backoff": (lambda attempt: "ok",),
    "call_with_fallback": (FALLBACK_CHAIN, _ok),
    "ab_bucket": ("user_12345", "chat_v2", 10),
    "percentiles": (_latencies, (50, 90, 99)),
    "summarize_requests": (_logs, MODEL_PRICING),
}
