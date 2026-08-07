"""Входные данные для замера скорости."""

import random

_replicas = [True] * 7 + [False] * 3
_rng_seed = 11

_scenario = {"error_rate": 0.015}

BENCH = {
    "burn_rate": (0.015, 0.0005),
    "should_abort": (30.0, 0.3),
    "inject_failures": (20000, 0.25, random.Random(_rng_seed)),
    "route_request": (_replicas, 12345),
    "serve_request": (_replicas, 1, False, True),
    "run_scenario": (_replicas, 20000, 0.25, True, random.Random(_rng_seed)),
    "experiment_report": ("provider 429", _scenario, 0.30),
}
