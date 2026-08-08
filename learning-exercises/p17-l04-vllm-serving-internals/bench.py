"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)

_lens = [_rng.randrange(64, 8192) for _ in range(4000)]
_workload = tuple(
    (i * 0.01, _rng.choice([128, 256, 512, 2048, 8192]), _rng.randrange(50, 300))
    for i in range(60)
)
_samples = [_rng.random() for _ in range(20000)]

BENCH = {
    "blocks_for": (8192, 16),
    "contiguous_waste": (_lens, 8192),
    "paged_waste": (_lens, 16),
    "chunk_plan": (65536, 512),
    "percentile": (_samples, 99),
    "schedule_static": (_workload, 16),
    # 7104 blocks fit the whole synthetic workload; 8000 leaves headroom so
    # the benchmark measures scheduling rather than the admission guard.
    "schedule_continuous": (_workload, 8000, 512),
}
