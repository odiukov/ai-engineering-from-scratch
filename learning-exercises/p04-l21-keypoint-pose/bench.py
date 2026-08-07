"""Входные данные для замера скорости."""

import math
import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

_SIZE = 96  # карта 96x96 на сустав — типичный выход HRNet до апсемплинга


def _blob(cx, cy, sigma=2.0):
    d = 2.0 * sigma * sigma
    return [
        [math.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / d)) for x in range(_SIZE)]
        for y in range(_SIZE)
    ]


# 17 суставов COCO
_centres = [(_rng.uniform(4, _SIZE - 5), _rng.uniform(4, _SIZE - 5)) for _ in range(17)]
_heatmaps = [_blob(cx, cy) for cx, cy in _centres]

_pred = [(x + _rng.gauss(0.0, 1.5), y + _rng.gauss(0.0, 1.5)) for x, y in _centres]
_true = list(_centres)
_kappas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
           0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]

# PAF на той же сетке: в каждой клетке единичный вектор
_paf = [
    [(math.cos(0.01 * (x + y)), math.sin(0.01 * (x + y))) for x in range(_SIZE)]
    for y in range(_SIZE)
]

BENCH = {
    "gaussian_heatmap": (_SIZE, 48.3, 61.7, 2.0),
    "argmax_coords": (_heatmaps[0],),
    "subpixel_offset": (_heatmaps[0], 48, 48),
    "heatmaps_to_keypoints": (_heatmaps,),
    "mean_l2_error": (_pred, _true),
    "pck": (_pred, _true, 0.2, 80.0),
    "oks": (_pred, _true, 80.0, _kappas),
    "paf_line_integral": (_paf, (2, 2), (90, 90), 200),
}
