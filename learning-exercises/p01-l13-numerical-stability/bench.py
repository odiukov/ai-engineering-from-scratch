"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# 20000 слагаемых: компенсация добавляет три операции на элемент, и разница
# между аккуратным циклом и списковым включением уже видна
_values = [random.uniform(-1.0, 1.0) for _ in range(20000)]
# логиты сдвинуты далеко вверх: наивная реализация на них просто упадёт
_logits = [500.0 + random.uniform(-5.0, 5.0) for _ in range(2000)]
_grads = [random.uniform(-10.0, 10.0) for _ in range(20000)]

BENCH = {
    "kahan_sum": (_values,),
    "stable_variance": (_values,),
    "logsumexp": (_logits,),
    "log_softmax": (_logits,),
    "softmax": (_logits,),
    "cross_entropy": (7, _logits),
    "relative_error": (1.0, 1.0000001),
    "clip_by_norm": (_grads, 1.0),
}
