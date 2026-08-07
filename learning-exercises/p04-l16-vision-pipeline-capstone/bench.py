"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# 400 боксов на кадре 640x480 — примерно столько отдаёт детектор до NMS
_boxes = []
for _ in range(400):
    x1 = _rng.uniform(-40, 640)
    y1 = _rng.uniform(-40, 480)
    _boxes.append((x1, y1, x1 + _rng.uniform(0, 120), y1 + _rng.uniform(0, 120)))

_clamped, _valid = [], []
for _i, _b in enumerate(_boxes):
    _x1 = min(max(_b[0], 0.0), 640.0)
    _y1 = min(max(_b[1], 0.0), 480.0)
    _x2 = min(max(_b[2], 0.0), 640.0)
    _y2 = min(max(_b[3], 0.0), 480.0)
    _clamped.append((_x1, _y1, _x2, _y2))
    if (_x2 - _x1) >= 32 and (_y2 - _y1) >= 32:
        _valid.append(_i)

_names = [f"class_{i}" for i in range(1000)]
_preds = [(_rng.randrange(1000), _rng.random()) for _ in _valid]

_detections = [
    {"box": b, "score": _rng.random(), "class_id": 0} for b in _clamped
]
_classifications = [
    {"detection_index": i, "class_id": 0, "class_name": "class_0", "score": 0.5}
    for i in _valid
]

_stages = {
    "preprocess": [_rng.uniform(2.0, 5.0) for _ in range(200)],
    "detect": [_rng.uniform(300.0, 500.0) for _ in range(200)],
    "classify": [_rng.uniform(20.0, 40.0) for _ in range(200)],
}

BENCH = {
    "validate_box": (_boxes[0],),
    "validate_detection": (_clamped[0], 0.5, 3),
    "clamp_box": (_boxes[0], 640, 480),
    "is_classifiable": (_clamped[0], 32),
    "select_crops": (_boxes, 640, 480, 32),
    "attach_classifications": (_valid, _preds, _names),
    "build_result": ("bench", _detections, _classifications, 12.5),
    "bottleneck_stage": (_stages,),
}
