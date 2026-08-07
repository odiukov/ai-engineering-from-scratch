"""Входные данные для замера скорости."""

import random

random.seed(0)

_C = 100
_logits = [random.uniform(-8.0, 8.0) for _ in range(_C)]
_dist = [1.0 / _C] * _C
_batch = [[random.uniform(-8.0, 8.0) for _ in range(_C)] for _ in range(400)]
_targets = [random.randrange(_C) for _ in range(400)]
_images = [[random.random() for _ in range(256)] for _ in range(256)]
_labels = [random.randrange(10) for _ in range(256)]
_true = [random.randrange(_C) for _ in range(5000)]
_pred = [random.randrange(_C) for _ in range(5000)]
_cm = [[random.randrange(50) for _ in range(_C)] for _ in range(_C)]

BENCH = {
    "softmax": (_logits,),
    "cross_entropy": (_logits, 7),
    "one_hot": (7, _C, 0.1),
    "soft_cross_entropy": (_logits, _dist),
    "mixup_batch": (_images, _labels, 10, 0.4, random.Random(0)),
    "confusion_matrix": (_true, _pred, _C),
    "class_report": (_cm,),
    "top_k_accuracy": (_batch, _targets, 5),
}
