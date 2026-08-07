"""Входные данные для замера скорости."""

import random

random.seed(0)

_ANCHORS = [(30, 60), (75, 170), (200, 380)]


def _box(rng):
    x1 = rng.uniform(0, 380)
    y1 = rng.uniform(0, 380)
    return (x1, y1, x1 + rng.uniform(10, 120), y1 + rng.uniform(10, 120))


_rng = random.Random(0)
# 600 боксов: квадратичный NMS здесь заметно медленнее аккуратного
_boxes = [_box(_rng) for _ in range(600)]
_scores = [_rng.random() for _ in range(600)]
_gt = [_box(_rng) for _ in range(200)]

# 13x13 сетка на 3 анкера, 20 классов — раскладка YOLOv2 для VOC
_slots = 13 * 13 * 3
_pred = [[_rng.uniform(-4, 4) for _ in range(25)] for _ in range(_slots)]
_target = [[0.0] * 25 for _ in range(_slots)]
_has_obj = [i % 50 == 0 for i in range(_slots)]

BENCH = {
    "sigmoid": (0.7,),
    "iou": (_boxes[0], _boxes[1]),
    "nms": (_boxes, _scores, 0.45),
    "decode_box": ((0.3, -0.2, 0.1, 0.4), 3, 4, 32, (30, 60)),
    "encode_box": ((100.0, 130.0, 148.0, 218.0), 3, 5, 32, (30, 60)),
    "best_anchor": ((70, 180), _ANCHORS),
    "yolo_loss": (_pred, _target, _has_obj),
    "precision_recall": (_boxes, _gt, 0.5),
}
