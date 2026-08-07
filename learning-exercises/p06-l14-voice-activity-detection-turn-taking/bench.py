"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
# 60 000 отсчётов при 16 кГц — это ~3.75 с речи, реальный размер буфера VAD
_chunk = [0.4 * math.sin(2 * math.pi * i / 50) + 0.05 * _rng.gauss(0, 1) for i in range(60000)]
_probs = [0.5 + 0.45 * math.sin(2 * math.pi * i / 37) for i in range(60000)]
_buffer = [[0.0] * 320 for _ in range(3000)]

BENCH = {
    "rms": (_chunk,),
    "dbfs": (_chunk,),
    "energy_vad": (_chunk, -40.0),
    "hysteresis_flags": (_probs, 0.5, 0.35),
    "pre_roll": (_buffer, 400.0, 20),
    "flush_latency_ms": (500.0, 4.0),
}
