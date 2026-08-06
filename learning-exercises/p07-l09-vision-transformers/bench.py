"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_H = _W = 48
_P = 6
_C = 3
_D = 64

_image = [[[random.random() for _ in range(_C)] for _ in range(_W)] for _ in range(_H)]
_patches = [
    [random.random() for _ in range(_P * _P * _C)]
    for _ in range((_H // _P) * (_W // _P))
]
_proj = [[random.gauss(0, 0.1) for _ in range(_D)] for _ in range(_P * _P * _C)]
_tokens = [[random.gauss(0, 1) for _ in range(_D)] for _ in _patches]
_pos = [[0.01] * _D for _ in _patches]

BENCH = {
    "patch_grid": (_H, _W, _P),
    "patchify": (_image, _P),
    "unpatchify": (_patches, _P, _H, _W),
    "linear_project": (_patches, _proj),
    "pos_2d": (28, 28, 256),
    "add_cls_and_pos": (_tokens, [0.0] * _D, _pos),
    "attention_pairs": (224, 224, 16),
    "vit_param_count": (768, 12, 196, 16),
}
