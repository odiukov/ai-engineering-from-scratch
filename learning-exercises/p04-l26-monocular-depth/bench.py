"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_H, _W = 120, 160

# синтетическая сцена: пол уходит вдаль сверху вниз, посередине ближний ящик
_target = []
for v in range(_H):
    row = []
    for u in range(_W):
        d = 1.0 + (v / _H) * 4.0
        if abs(u - _W / 2) < _W / 6 and abs(v - _H * 0.6) < _H / 6:
            d = 2.0
        row.append(d)
    _target.append(row)

# «предсказание» относительной модели: тот же рельеф в другом масштабе и с шумом
_pred = [[3.0 * d + 0.5 + _rng.uniform(-0.05, 0.05) for d in row] for row in _target]

_intr = (320.0, 320.0, _W / 2, _H / 2)

BENCH = {
    "flatten_valid": (_pred, _target),
    "abs_rel_error": (_pred, _target),
    "delta_accuracy": (_pred, _target),
    "align_scale_shift": (_pred, _target),
    "aligned_abs_rel": (_pred, _target),
    "pixel_to_camera": (80, 60, 2.0, _intr),
    "depth_to_point_cloud": (_target, _intr),
    "lift_box_to_3d": (_target, (30, 40, 130, 110), _intr),
}
