"""Входные данные для замера скорости."""

import math
import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим


def _cov(sx, sy, angle):
    c, s = math.cos(angle), math.sin(angle)
    a = sx * sx * c * c + sy * sy * s * s
    b = (sx * sx - sy * sy) * s * c
    d = sx * sx * s * s + sy * sy * c * c
    return [[a, b], [b, d]]


# один тайл 16x16 обычно накрывают сотни сплатов; берём 2000
_splats = []
for _ in range(2000):
    _splats.append(
        {
            "mean": (_rng.uniform(-20, 20), _rng.uniform(-20, 20)),
            "cov": _cov(_rng.uniform(0.5, 4.0), _rng.uniform(0.5, 4.0), _rng.uniform(0, math.pi)),
            "colour": (_rng.random(), _rng.random(), _rng.random()),
            "opacity": _rng.random(),
            "depth": _rng.random(),
        }
    )

_layers = [(_rng.random() * 0.5, (_rng.random(), _rng.random(), _rng.random())) for _ in range(50000)]
_sh = [(_rng.random(), _rng.random(), _rng.random()) for _ in range(4)]

BENCH = {
    "covariance_2d": (2.5, 0.7, 0.9),
    "inverse_2x2": (_splats[0]["cov"],),
    "gaussian_density": ((0.0, 0.0), _splats[0]["cov"], (1.5, -2.0)),
    "alpha_composite": (_layers,),
    "render_pixel": (_splats, (0.0, 0.0)),
    "eval_sh_degree_1": (_sh, (0.0, 0.6, 0.8)),
    "densify_decision": (0.001, 0.5, 0.9),
    "gaussian_float_count": (3,),
}
