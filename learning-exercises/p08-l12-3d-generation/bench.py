"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_SIZE = 24


def _splat():
    return {
        "pos": [_rng.uniform(2, _SIZE - 2), _rng.uniform(2, _SIZE - 2)],
        "sigma": _rng.uniform(0.8, 2.5),
        "color": _rng.uniform(0.2, 0.8),
    }


_gaussians = [_splat() for _ in range(40)]
_target = [[_rng.uniform(0.0, 1.0) for _ in range(_SIZE)] for _ in range(_SIZE)]
_image = [[_rng.uniform(0.0, 1.0) for _ in range(_SIZE)] for _ in range(_SIZE)]
_layers = [(_rng.random(), _rng.random()) for _ in range(20000)]

BENCH = {
    "gaussian_value": (3, 4, _gaussians[0]),
    "render": (_SIZE, _gaussians),
    "image_mse": (_image, _target),
    "color_gradients": (_SIZE, _gaussians, _target),
    "fit_colors": (_SIZE, _gaussians[:6], _target, 0.5, 5),
    "alpha_composite": (_layers,),
    "prune_gaussians": (_gaussians * 500, 0.5),
    "split_gaussian": (_gaussians[0], [1.0, 0.5]),
}
