"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# (8, 12, 16) — 1536 элементов: полный обход формы стоит десятки миллисекунд,
# и лишний пересчёт strides внутри цикла становится заметен
_SHAPE = (8, 12, 16)
_data = [random.random() for _ in range(8 * 12 * 16)]
_bias = [random.random() for _ in range(16)]

BENCH = {
    "strides": (_SHAPE,),
    "flat_index": (_SHAPE, (7, 11, 15)),
    "reshape": (_SHAPE, (-1, 16)),
    "permute": (_data, _SHAPE, (2, 0, 1)),
    "broadcast_shapes": (_SHAPE, (16,)),
    "broadcast_to": (_bias, (16,), _SHAPE),
    "add": (_data, _SHAPE, _bias, (16,)),
    "reduce_sum": (_data, _SHAPE, 1),
}
