"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим


def _tensor(dims):
    if len(dims) == 1:
        return [_rng.gauss(0.0, 1.0) for _ in range(dims[0])]
    return [_tensor(dims[1:]) for _ in range(dims[0])]


_plane = _tensor([32, 32])
_x = _tensor([3, 32, 32])
_w = _tensor([8, 3, 3, 3])
_b = _tensor([8])
_kernel = _tensor([3, 3])

BENCH = {
    "conv_output_size": (224, 3, 1, 2),
    "conv_params": (3, 64, 3),
    "pad2d": (_plane, 1),
    "conv2d": (_plane, _kernel, 1, 1),
    "conv2d_multichannel": (_x, _w, _b, 1, 1),
    "im2col": (_x, 3, 3, 1, 1),
    "conv2d_im2col": (_x, _w, _b, 1, 1),
    "receptive_field": ([(3, 1), (3, 2)] * 8,),
}
