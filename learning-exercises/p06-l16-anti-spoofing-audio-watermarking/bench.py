"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
# 80 000 отсчётов при 16 кГц — пять секунд, реальная длина одной реплики TTS
_host = [0.5 * math.sin(2 * math.pi * 440 * i / 16000) for i in range(80000)]
_payload = [(i * 7) % 2 for i in range(16)]
_spec = [1.0 / (k + 1) for k in range(4096)]
_marked = [x + 0.05 * _rng.choice((-1.0, 1.0)) for x in _host]
_real = [_rng.gauss(0.7, 0.15) for _ in range(600)]
_fake = [_rng.gauss(0.3, 0.15) for _ in range(600)]

BENCH = {
    "spectral_rolloff": (_spec, 0.85),
    "is_suspicious": (_spec, 0.92),
    "chip_sequence": (80000, 0),
    "embed_watermark": (_host, _payload, 0.05, 0),
    "add_noise": (_host, 10.0, random.Random(0)),
    "detect_watermark": (_marked, 16, 0.05, 0),
    "bit_recovery_accuracy": (_payload, _payload[::-1]),
    "eer": (_real, _fake),
}
