"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_bids = [
    {
        "bidder": f"worker-{i:03d}",
        "price": round(random.uniform(40.0, 160.0), 2),
        "quality": round(random.random(), 3),
    }
    for i in range(2000)
]

BENCH = {
    "zopa": (100.0, 60.0),
    "concede": (60.0, 100.0, 0.3),
    "accepts": ("seller", 72.0, 60.0, 71.76),
    "bargain": (100.0, 60.0, None, 0.3, 200),
    "naive_bargain": (100.0, 60.0, random.Random(0), 0.3, 200),
    "narrate": (72.0, "desperate"),
    "contract_net": (_bids, 100.0, "best_quality"),
}
