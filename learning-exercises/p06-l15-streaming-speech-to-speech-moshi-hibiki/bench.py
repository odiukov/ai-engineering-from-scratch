"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
# 40 000 кадров по 80 мс — это примерно 53 минуты разговора
_stages = [_rng.uniform(10.0, 200.0) for _ in range(40000)]
_acoustic = [_rng.randrange(2048) for _ in range(8)]
_heads = [(lambda c, p: len(p)) for _ in range(2000)]

BENCH = {
    "frame_ms_from_rate": (12.5,),
    "tokens_per_second": (12.5, 8, 2),
    "theoretical_latency_ms": (80.0, 1),
    "pipeline_latency_ms": (_stages,),
    "build_frame": ("привет", _acoustic, 8),
    "depth_decode": ("ctx", _heads),
}
