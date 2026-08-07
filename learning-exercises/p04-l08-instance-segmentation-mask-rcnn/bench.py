"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_boxes = []
for _ in range(300):
    x = random.uniform(0, 200)
    y = random.uniform(0, 200)
    _boxes.append((x, y, x + random.uniform(5, 60), y + random.uniform(5, 60)))
_scores = [random.random() for _ in _boxes]

_feature = [[random.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(64)]
_small_mask = [[random.random() for _ in range(28)] for _ in range(28)]
_mask_a = [[random.randrange(2) for _ in range(96)] for _ in range(96)]
_mask_b = [[random.randrange(2) for _ in range(96)] for _ in range(96)]

BENCH = {
    "box_iou": (_boxes[0], _boxes[1]),
    "nms": (_boxes, _scores, 0.7),
    "decode_box_delta": ((0.0, 0.0, 32.0, 32.0), (0.1, -0.2, 0.3, 0.05)),
    "bilinear_sample": (_feature, 17.3, 42.9),
    "roi_align": (_feature, (10.3, 12.7, 55.1, 60.4), 14, 1.0),
    "roi_pool": (_feature, (10.3, 12.7, 55.1, 60.4), 14, 1.0),
    "paste_mask": (_small_mask, (5.0, 5.0, 90.0, 90.0), 96, 96),
    "mask_iou": (_mask_a, _mask_b),
}
