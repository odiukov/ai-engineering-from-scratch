"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_skus = ["sku-001", "sku-002", "sku-003"]

_actions = []
for _ in range(3000):
    _actions.append(("add_to_cart", random.choice(_skus)))
    if random.random() < 0.3:
        _actions.append(("remove_from_cart", random.choice(_skus)))
    if random.random() < 0.2:
        _actions.append(("checkout",))

_records = [
    {"intended": "buy", "clicked": random.choice(["buy", "cart", None]),
     "plan_ok": random.random() < 0.8}
    for _ in range(20000)
]

_results = [
    {"task_id": f"t{i:05d}", "success": random.random() < 0.4,
     "steps": random.randint(1, 20), "gold_steps": random.randint(1, 8)}
    for i in range(20000)
]

BENCH = {
    "new_state": (),
    "apply_action": ({"cart": {"sku-001": 1}, "orders": []}, ("checkout",)),
    "run_trajectory": (_actions,),
    "task_succeeded": ({"cart": {}, "orders": [{"oid": "ord-001",
                        "items": {"sku-001": 1}, "total": 199}] * 5000},
                       {"sku-002": 1}),
    "trajectory_efficiency": (7, 3),
    "classify_step": ({"intended": "buy", "clicked": "cart", "plan_ok": True},),
    "failure_breakdown": (_records,),
    "benchmark_report": (_results,),
}
