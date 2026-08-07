"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_V = 512  # словарь: спекулятивный шаг обязан быть линейным по нему
_raw_q = [random.random() for _ in range(_V)]
_total_q = sum(_raw_q)
_Q = [w / _total_q for w in _raw_q]

# черновик — тот же q, слегка сдвинутый: alpha получается высоким
_raw_p = [w + 0.15 * random.random() for w in _raw_q]
_total_p = sum(_raw_p)
_P = [w / _total_p for w in _raw_p]

_rng = random.Random(0)

BENCH = {
    "normalize": (_raw_q,),
    "sample_index": (_Q, 0.73),
    "accept": (0.3, 0.25, 0.5),
    "residual": (_Q, _P),
    "spec_step": (_P, _Q, 8, _rng),
    "expected_emitted": (0.9, 8),
    "time_per_token": (0.9, 8, 0.04),
    "best_draft_length": (0.9, 0.04, 40),
}
