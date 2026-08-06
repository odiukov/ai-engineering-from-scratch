"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_v = 256
_raw = [random.random() for _ in range(_v)]
_total = sum(_raw)
_q = [x / _total for x in _raw]
_p = [(0.5 * x + 0.5 / _v) for x in _q]

_cache = [(i, i * 2) for i in range(4096)]

BENCH = {
    "sample_from": (_q, random.Random(1)),
    "residual_dist": (_q, _p),
    "accept_draft": (0.01, 0.02, 0.4),
    "acceptance_probability": (_q, _p),
    "spec_step": (_q, _p, 8, random.Random(2)),
    "expected_tokens_per_verify": (0.85, 5),
    "kl_divergence": (_q, _p),
    "rollback_kv": (_cache, 4000, 3),
}
