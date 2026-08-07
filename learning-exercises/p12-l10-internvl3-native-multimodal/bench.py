"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_mix = {"text": 40, "interleaved": 35, "caption": 20, "video": 5}

# 50 000 запросов — примерно час трафика продакшена; на таком объёме
# видно разницу между одним проходом и пересчётом порогов на каждый запрос.
_details = [random.random() for _ in range(50000)]

BENCH = {
    "normalize_mix": (_mix,),
    "sample_modalities": (_mix, 50000, random.Random(0)),
    "route_resolution": (0.55,),
    "routed_tokens": (_details,),
    "routing_speedup": (_details,),
    "dvd_speedup": (37.0, 51.0),
    "alignment_debt": (80.0, 74.0, 12.0),
}
