"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ROWS, _COLS = 256, 256

_matrix = [[random.gauss(0.0, 0.02) for _ in range(_COLS)] for _ in range(_ROWS)]
_matrix[7] = [v * 15 for v in _matrix[7]]  # строка-выброс, как в реальных весах

_flat = [v for row in _matrix for v in row]
_q_flat, _scale = (
    [min(max(round(v / 0.002), -128), 127) for v in _flat],
    0.002,
)
_reconstructed = [q * _scale for q in _q_flat]

_positive = [abs(v) + 0.5 for v in _flat]
_q_asym = [min(max(round(v / 0.004), 0), 255) for v in _positive]

_q_rows = [_q_flat[i * _COLS:(i + 1) * _COLS] for i in range(_ROWS)]
_scales = [0.002] * _ROWS

BENCH = {
    "quantize_symmetric": (_flat, 8),
    "dequantize": (_q_flat, _scale),
    "quantize_asymmetric": (_positive, 8),
    "dequantize_asymmetric": (_q_asym, 0.004, 0),
    "quantize_per_channel": (_matrix, 8),
    "dequantize_per_channel": (_q_rows, _scales),
    "quantization_error": (_flat, _reconstructed),
    "model_memory_gb": (70, 4),
}
