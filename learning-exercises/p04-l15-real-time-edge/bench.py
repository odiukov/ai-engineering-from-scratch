"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_times = [_rng.expovariate(0.15) for _ in range(20000)]
_values = [_rng.gauss(0.0, 2.0) for _ in range(20000)]
_qs, _scale, _zp = (
    [_rng.randint(-128, 127) for _ in range(20000)],
    0.0117647,
    -128,
)

_layers = []
for _i in range(300):
    _layers.append(
        {"type": "conv", "c_in": 64, "c_out": 64, "k": 3, "h_out": 56, "w_out": 56}
    )
    _layers.append({"type": "linear", "in_features": 512, "out_features": 512})

BENCH = {
    "drop_warmup": (_times, 100),
    "latency_stats": (_times,),
    "throughput_fps": (12.5, 8),
    "conv2d_flops": (64, 128, 3, 56, 56, 1),
    "linear_flops": (1024, 1000),
    "model_flops": (_layers,),
    "quantize_int8": (_values,),
    "dequantize_int8": (_qs, _scale, _zp),
}
