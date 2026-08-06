"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_z = (random.uniform(-5, 5), random.uniform(-5, 5))
_w = (random.uniform(-5, 5), random.uniform(-5, 5))

BENCH = {
    "c_mul": (_z, _w),
    "c_conj": (_z,),
    "c_abs": (_z,),
    "c_div": (_z, _w),
    "to_polar": (_z,),
    "from_polar": (2.5, 0.7),
    "c_pow": (_z, 17),
    "roots_of_unity": (20000,),
}
