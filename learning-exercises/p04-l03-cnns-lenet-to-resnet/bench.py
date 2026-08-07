"""Входные данные для замера скорости."""

import random

random.seed(0)

_gains = [random.uniform(0.4, 0.9) for _ in range(2000)]
_x = [random.uniform(-1.0, 1.0) for _ in range(2000)]
_f = lambda v: [t * 0.5 for t in v]

BENCH = {
    "spatial_out": (224, 3, 2, 1),
    "conv_params": (256, 512, 3),
    "dense_params": (25088, 4096),
    "lenet5_shapes": (32,),
    "lenet5_params": (),
    "shortcut_kind": (64, 128, 2),
    "residual_forward": (_x, _f),
    "gradient_scale": (_gains, True),
}
