"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

IMG_OPEN = -1
IMG_CLOSE = -2
PATCH = -3

# последовательность из текста и четырёх картинок по 64 патча: маска на
# такой длине уже квадратная по памяти, наивная реализация это чувствует
_SEQ = []
for _ in range(4):
    _SEQ += [random.randint(0, 5000) for _ in range(16)]
    _SEQ += [IMG_OPEN] + [PATCH] * 64 + [IMG_CLOSE]
_SEQ += [random.randint(0, 5000) for _ in range(16)]

_DIM = 256
_X0 = [random.gauss(0.0, 1.0) for _ in range(_DIM)]
_EPS = [random.gauss(0.0, 1.0) for _ in range(_DIM)]
_PRED = [random.gauss(0.0, 1.0) for _ in range(_DIM)]

BENCH = {
    "find_image_blocks": (_SEQ,),
    "build_mask": (_SEQ,),
    "flow_interpolate": (_X0, _EPS, 0.37),
    "flow_target": (_X0, _EPS),
    "flow_loss": (_PRED, _X0, _EPS),
    "flow_loss_grad": (_PRED, _X0, _EPS),
    "balanced_weights": (2.3, 19.7),
    "generation_forward_passes": (512, 4096, 24),
}
