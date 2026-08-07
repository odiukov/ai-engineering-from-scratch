"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_SIZES = [
    (random.choice([224, 336, 448, 588]), random.choice([224, 336, 448, 588]))
    for _ in range(24)
]
_SPANS = []
_offset = 0
for _h, _w in _SIZES:
    _n = (_h // 14) * (_w // 14)
    _SPANS.append((_offset, _offset + _n))
    _offset += _n

_SMALL = [(0, 120), (120, 300), (300, 460), (460, 700)]

BENCH = {
    "patch_count": (1920, 1080, 14),
    "pack_batch": (_SIZES, 14),
    "block_diagonal_mask": (_SMALL,),
    "mask_density": (_SPANS,),
    "padded_batch_cost": (_SIZES, 14),
    "square_resize_cost": (_SIZES, 336, 14),
    "drop_patches": (_SPANS, 0.5, random.Random(0)),
    "fit_to_token_budget": (4096, 4096, 14, 1024),
}
