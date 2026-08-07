"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# кривая recall из 200 точек: наивный линейный поиск по ней и меряется
_CURVE = []
_prev = 0.0
for _i in range(1, 201):
    _thresh = _i / 200.0
    _CURVE.append((_thresh, 1.0 - 0.3 * _thresh))

_RNG = random.Random(0)

BENCH = {
    "token_budget": (7200, 2, 729),
    "max_duration": (10_000_000, 2, 81),
    "summary_token_budget": (7200, 4, 16),
    "compression_gain": (583200, 145),
    "ring_chunk": (10_000_000, 1024),
    "recall_at": (_CURVE, 0.995),
    "needle_trial": (7200.0, _CURVE, _RNG),
    "pick_strategy": (120, "specific"),
}
