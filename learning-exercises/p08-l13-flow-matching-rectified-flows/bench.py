"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)
_x0s = [_rng.gauss(2.0, 0.3) for _ in range(4000)]
_x1s = [_rng.gauss(0.0, 1.0) for _ in range(4000)]
_ts = [_rng.random() for _ in range(4000)]
_noise = [_rng.gauss(0.0, 1.0) for _ in range(200)]

_field = lambda x, t: 0.6 * x - 0.4 * t + 0.1

BENCH = {
    "interpolate": (2.0, -1.0, 0.35),
    "flow_target": (2.0, -1.0),
    "flow_matching_loss": (_field, _x0s, _x1s, _ts),
    "euler_sample": (_field, 1.0, 2000),
    "path_curvature": (_field, 1.0, 2000),
    "reflow_pairs": (_field, _noise, 200),
    "logit_normal_t": (random.Random(1),),
    "cfg_velocity": (2.0, 1.0, 3.0),
}
