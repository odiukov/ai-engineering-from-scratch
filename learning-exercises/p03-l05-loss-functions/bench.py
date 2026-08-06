"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_preds = [random.uniform(0.001, 0.999) for _ in range(2000)]
_targets = [float(random.randint(0, 1)) for _ in range(2000)]
_logits = [random.gauss(0.0, 3.0) for _ in range(256)]

BENCH = {
    "mse": (_preds, _targets),
    "mse_gradient": (_preds, _targets),
    "binary_cross_entropy": (_preds, _targets),
    "bce_gradient": (_preds, _targets),
    "softmax": (_logits,),
    "categorical_cross_entropy": (_logits, 7),
    "cce_gradient": (_logits, 7),
    "label_smoothed_cce": (_logits, 7, 0.1),
}
