"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

BENCH = {
    "horizon_at": (36.0,),
    "months_to_cross": (720.0,),
    "end_to_end_reliability": (0.999, 5000),
    "max_steps_for_target": (0.9995, 0.5),
    "deployment_horizon": (14.0, 0.4),
    "simulate_run": (_rng, 0.999, 2000, 10_000.0, 1.0),
    "horizon_verdict": (4.0, 14.0, 0.3, 2.0),
}
