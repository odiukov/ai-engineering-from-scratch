"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)

_z = [_rng.gauss(0, 1) for _ in range(64)]
_W = [[_rng.gauss(0, 0.2) for _ in range(64 + 8)] for _ in range(64)]
_b = [0.0] * 64

_y = [_rng.gauss(0, 1) for _ in range(4096)]
_y_hat = [_rng.gauss(0, 1) for _ in range(4096)]

_targets = [_rng.gauss(0, 1) for _ in range(4000)]

# 64x64 «картинка»: PatchGAN 8x8 со шагом 4 даёт 15x15 = 225 патчей
_image = [[_rng.random() for _ in range(64)] for _ in range(64)]
_patch_mean = lambda patch: sum(sum(row) for row in patch) / (len(patch) * len(patch[0]))

_double = lambda v: [a * 2.0 for a in v]
_halve = lambda v: [a / 2.0 for a in v]

BENCH = {
    "one_hot": (3, 8),
    "conditioned_input": (_z, 3, 8),
    "linear_generator": (_z, 3, _W, _b, 8),
    "l1_loss": (_y, _y_hat),
    "best_constant": (_targets, "l1"),
    "pix2pix_generator_loss": (0.5, _y, _y_hat, 100.0),
    "patchgan_score": (_image, 8, 4, _patch_mean),
    "cycle_consistency_loss": (_y, _double, _halve),
}
