"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_H, _W, _C = 64, 64, 4

_logits = [
    [[random.uniform(-4.0, 4.0) for _ in range(_C)] for _ in range(_W)]
    for _ in range(_H)
]
_targets = [[random.randrange(_C) for _ in range(_W)] for _ in range(_H)]
_preds = [[random.randrange(_C) for _ in range(_W)] for _ in range(_H)]
_mask_a = [[random.randrange(2) for _ in range(_W)] for _ in range(_H)]
_mask_b = [[random.randrange(2) for _ in range(_W)] for _ in range(_H)]
_ious = [random.random() for _ in range(200)]

BENCH = {
    "softmax": ([1.0, -2.0, 0.5, 3.0],),
    "pixel_accuracy": (_preds, _targets),
    "pixel_cross_entropy": (_logits, _targets),
    "dice_coefficient": (_mask_a, _mask_b),
    "dice_loss": (_logits, _targets, _C),
    "combined_loss": (_logits, _targets, _C),
    "iou_per_class": (_preds, _targets, _C),
    "mean_iou": (_ious,),
}
