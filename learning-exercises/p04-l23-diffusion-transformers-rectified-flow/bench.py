"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_SIZE = 64
_PATCH = 4
_CHANNELS = 3

_image = [
    [[random.uniform(-1.0, 1.0) for _ in range(_SIZE)] for _ in range(_SIZE)]
    for _ in range(_CHANNELS)
]
_tokens = [
    [random.uniform(-1.0, 1.0) for _ in range(_CHANNELS * _PATCH * _PATCH)]
    for _ in range((_SIZE // _PATCH) ** 2)
]

_DIM = 512
_x = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
_scale = [random.uniform(-0.1, 0.1) for _ in range(_DIM)]
_shift = [random.uniform(-0.1, 0.1) for _ in range(_DIM)]
_branch = lambda h: [v * 0.5 for v in h]

_x0 = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
_eps = [random.gauss(0.0, 1.0) for _ in range(_DIM)]
_batch = [[random.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(64)]
_model = lambda x_t, t: [v * t for v in x_t]

BENCH = {
    "patchify": (_image, _PATCH),
    "unpatchify": (_tokens, _CHANNELS, _PATCH),
    "adaln_zero_block": (_x, _branch, _scale, _shift, 0.5),
    "rectified_flow_path": (_x0, _eps, 0.3),
    "velocity_target": (_x0, _eps),
    "flow_matching_loss": (_model, _batch, random.Random(0)),
    "classifier_free_guidance": (_x0, _eps, 3.5),
    "euler_sample": (_model, _eps, 20),
}
