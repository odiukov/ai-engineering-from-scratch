"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_DIM = 8
_K = 128

# Квантование в лоб — это len(vectors) * K сравнений. 256 на 128 даёт
# десятки миллисекунд: разница между sqrt и квадратом расстояния видна,
# а один вызов всё ещё мгновенный.
_codebook = [[random.uniform(-1.0, 1.0) for _ in range(_DIM)] for _ in range(_K)]
_vectors = [[random.uniform(-1.0, 1.0) for _ in range(_DIM)] for _ in range(256)]
_indices = [random.randrange(_K) for _ in range(1024)]

_parts = [("text", [random.randrange(32) for _ in range(200)])]
for _ in range(8):
    _parts.append(("image", [random.randrange(16) for _ in range(256)]))
    _parts.append(("text", [random.randrange(32) for _ in range(200)]))
_ids = []
for _kind, _values in _parts:
    if _kind == "text":
        _ids.extend(_values)
    else:
        _ids.append(48)
        _ids.extend(32 + v for v in _values)
        _ids.append(49)

BENCH = {
    "nearest_code": (_vectors[0], _codebook),
    "quantize": (_vectors, _codebook),
    "dequantize": (_indices, _codebook),
    "reconstruction_mse": (_vectors, _codebook),
    "compression_ratio": (512, 512, 1024, 8192),
    "encode_document": (_parts,),
    "decode_document": (_ids,),
    "qk_norm": ([random.uniform(-100.0, 100.0) for _ in range(4096)],),
}
