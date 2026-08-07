"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_signal = [0.7 * math.sin(2 * math.pi * i / 40) + 0.2 * _rng.gauss(0, 1.0) for i in range(4000)]
_codebook = [-1.2 + 2.4 * i / 63 for i in range(64)]
_cascade = [[c / (4 ** k) for c in _codebook] for k in range(6)]
_indices = [[i % 64 for i in range(len(_signal))] for _ in range(6)]
_frames = [[(i * 7 + k) % 1024 for k in range(8)] for i in range(4000)]

BENCH = {
    "nearest_code": (_codebook, 0.31),
    "uniform_codebook": (1024, 1.0),
    "quantize_layer": (_signal, _codebook),
    "rvq_encode": (_signal, _cascade),
    "rvq_decode": (_indices, _cascade, len(_signal)),
    "reconstruction_mse": (_signal, _signal[::-1]),
    "codec_cost": (10.0, 12.5, 8, 1024),
    "split_semantic_acoustic": (_frames,),
}
