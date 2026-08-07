"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)          # обязательно: замер должен быть воспроизводим

_target = [0.40, 0.25, 0.20, 0.10, 0.05]
_draft = [0.30, 0.30, 0.20, 0.15, 0.05]
_mix = tuple((0.125, 0.30 + 0.05 * i) for i in range(8))
_weights = [_rng.random() for _ in range(20000)]

BENCH = {
    "expected_speedup": (0.7, 5, 0.1),
    "breakeven_alpha": (5, 0.15),
    "blended_alpha": (_mix,),
    "normalize": (_weights,),
    "sample_index": (_target, _rng),
    "residual_distribution": (_target, _draft),
    "speculative_step": (_target, _draft, 5, _rng),
    "run_speculative": (_target, _draft, 5, 4000, _rng),
}
