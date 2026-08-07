"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_N = 8                  # перебор перестановок ограничен 8x8


def _box(rng):
    x = rng.uniform(0, 300)
    y = rng.uniform(0, 300)
    return (x, y, x + 20, y + 20)


_rng = random.Random(0)
_tracks = [_box(_rng) for _ in range(_N)]
_dets = [_box(_rng) for _ in range(_N)]
_cost = [[_rng.random() for _ in range(_N)] for _ in range(_N)]

# 3 объекта, едущих по прямым, 120 кадров — типичный синтетический прогон
_frames = [
    [
        (10 + 2 * f, 10, 30 + 2 * f, 30),
        (150, 20 + f, 170, 40 + f),
        (300 - 2 * f, 200, 320 - 2 * f, 220),
    ]
    for f in range(120)
]

_track_state = [
    {"id": i + 1, "bbox": box, "last_frame": 0, "hits": 1}
    for i, box in enumerate(_tracks)
]

_pred_per_frame = [[(i + 1, box) for i, box in enumerate(frame)] for frame in _frames]
_gt_per_frame = [[(i + 100, box) for i, box in enumerate(frame)] for frame in _frames]

BENCH = {
    "iou": (_tracks[0], _dets[0]),
    "iou_matrix": (_tracks, _dets),
    "optimal_assignment": (_cost,),
    "associate": (_tracks, _dets, 0.3),
    "update_tracks": (_track_state, _dets, 1, _N + 1),
    "run_tracker": (_frames,),
    "count_id_switches": (_pred_per_frame, _gt_per_frame),
    "mota": (12, 7, 3, 500),
}
