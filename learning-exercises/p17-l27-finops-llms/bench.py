"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_routes = ("haiku", "sonnet", "opus")

_calls = [
    {
        "trace_id": "t%05d" % i,
        "user_id": "u_%d" % (i % 500),
        "tenant_id": "t_%d" % (i % 40),
        "task_id": "task_%d" % (i % 12),
        "route": _routes[i % 3],
        "day": "2026-08-%02d" % (1 + i % 31),
        "layers": {
            "prompt": random.randint(200, 4000),
            "tool": random.randint(0, 900),
            "memory": random.randint(0, 700),
            "response": random.randint(20, 800),
        },
        "cached_input": i % 7 == 0,
        "batch": i % 11 == 0,
    }
    for i in range(20000)
]

# год ежедневного расхода: аномалии ищутся по всей истории, а прогноз — по месяцу
_daily = {
    "%04d-%02d-%02d" % (2025 + (i // 336), 1 + (i // 28) % 12, 1 + i % 28): 100.0
    + random.random() * 20
    for i in range(336)
}
_month = {"2026-08-%02d" % d: 80.0 + random.random() * 40 for d in range(1, 21)}
_history = [100.0 + random.random() * 20 for _ in range(365)]

_policy = {
    "contracted_daily_usd": 100.0,
    "spend_cap_multiplier": 2.0,
    "kill_z": 4.0,
    "min_history": 5,
}

BENCH = {
    "call_cost": (_calls[0],),
    "layer_shares": (_calls,),
    "attribute": (_calls, "tenant_id"),
    "daily_totals": (_calls,),
    "zscore": (900.0, _history),
    "anomaly_days": (_daily,),
    "forecast_month": (_month, "2026-08-20", 31),
    "enforcement_action": (900.0, _history, _policy),
}
