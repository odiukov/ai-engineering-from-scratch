"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# «Веса слоя»: гауссов шум плюс редкие выбросы — ровно та картина, ради
# которой придуманы AWQ и микромасштабирование.
_weights = [random.gauss(0.0, 0.05) for _ in range(50000)]
for _i in range(0, 50000, 977):
    _weights[_i] *= 60.0

_codes = list(range(256)) * 200

BENCH = {
    "quant_params": (_weights, 4),
    "quantize": (_weights, 0.01, 8, 4),
    "dequantize": (_codes, 0.01, 8),
    "roundtrip": (_weights, 4),
    "quantization_error": (_weights, 4),
    "blockwise_roundtrip": (_weights, 4, 32),
    "format_memory_gb": (405, 4, 8, 256, 8192),
    "pick_format": ("blackwell", False, False),
}
