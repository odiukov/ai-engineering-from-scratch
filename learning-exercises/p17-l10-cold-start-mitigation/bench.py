"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_phases = {f"phase-{i}": 1.0 + i for i in range(50)}
# сутки по минутам: ночной штиль, утренний всплеск, дневное плато
_day = [
    max(0, int(random.gauss(120 if 420 <= _m <= 1200 else 8, 25)))
    for _m in range(1440)
]

BENCH = {
    "weights_load_seconds": (140.0, 7.0),
    "cold_start_seconds": (_phases, ["pre_seeded", "streamer"]),
    "mitigation_savings": (_phases, ["pre_seeded", "streamer"]),
    "ready_at": (600.0, 328.0),
    "available_replicas": ([float(i) for i in range(20000)], 10000.0),
    "simulate_arrivals": (_day, 10, 4, 328.0, 60.0),
    "warm_pool_monthly_cost": (5, 4.50),
    "min_warm_pool_for": (0.05, _day, 10, 328.0, 60.0),
}
