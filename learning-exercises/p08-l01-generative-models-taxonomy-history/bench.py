"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_samples = [
    _rng.gauss(-2.0, 0.6) if _rng.random() < 0.4 else _rng.gauss(2.0, 0.9)
    for _ in range(600)
]


def _kde(samples, x):
    """Обёртка с фиксированной шириной ядра: integrate_density зовёт fn(samples, x)."""
    import math

    norm = 1.0 / math.sqrt(2 * math.pi)
    total = 0.0
    for s in samples:
        u = (x - s) / 0.3
        total += norm * math.exp(-0.5 * u * u)
    return total / (len(samples) * 0.3)


BENCH = {
    "model_family": ("StyleGAN",),
    "has_explicit_density": (3,),
    "histogram_density": (_samples, 0.0),
    "kde_density": (_samples, 0.0),
    "integrate_density": (_kde, _samples, -0.5, 0.5, 100),
    "implicit_generator": (_samples, 400, random.Random(0)),
    "sampling_cost": (50, 0.06),
    "speedup_source": (50, 0.06, 4, 0.01),
}
