"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)

_load = [_rng.randrange(0, 200) for _ in range(4000)]
_series = [_rng.randrange(1, 16) for _ in range(4000)]
_nodes = {f"n{i:04d}": _rng.randrange(0, 9) for i in range(2000)}
_cluster = {
    f"n{i:04d}": {
        "running_requests": _rng.randrange(0, 50),
        "empty_since": 0.0 if i % 3 == 0 else None,
        "utilization": float(_rng.randrange(0, 100)),
    }
    for i in range(2000)
}

BENCH = {
    "duty_cycle_util": (100, 4, 4),
    "queue_depth_per_replica": (100, 4),
    "desired_replicas": (4, 40.0, 10.0, 1, 16),
    "stabilize": (_series, 300),
    "run_autoscaler": (_load, 10.0, 5, 1, 64),
    "count_scale_events": (_series,),
    "gang_schedule": (_nodes, 64),
    "consolidation_plan": (_cluster, "WhenEmptyOrUnderutilized", 7200.0, 3600.0),
}
