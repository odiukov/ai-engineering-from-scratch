"""Входные данные для замера скорости."""

import math
import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим


def _unit(dim):
    v = [_rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


# батч SimCLR: 128 пар видов по 64 измерения -> матрица косинусов 256x256
_z1 = [_unit(64) for _ in range(128)]
_z2 = [_unit(64) for _ in range(128)]

_logits = [_rng.gauss(0.0, 1.0) for _ in range(65536)]
_center = [_rng.gauss(0.0, 0.1) for _ in range(65536)]
_teacher = [1.0 / 65536] * 65536

_weights_old = [_rng.random() for _ in range(200000)]
_weights_new = [_rng.random() for _ in range(200000)]

# 196 патчей ViT-B/16 по 768 пикселей
_original = [[_rng.random() for _ in range(768)] for _ in range(196)]
_reconstructed = [[_rng.random() for _ in range(768)] for _ in range(196)]
_masked = list(range(49, 196))

BENCH = {
    "l2_normalize": (_z1[0],),
    "cosine_similarity_matrix": (_z1,),
    "info_nce": (_z1, _z2, 0.1),
    "ema_update": (_weights_old, _weights_new, 0.996),
    "center_and_sharpen": (_logits, _center, 0.04),
    "dino_loss": (_logits, _teacher, 0.1),
    "random_mask_indices": (100000, 0.75, random.Random(0)),
    "masked_reconstruction_loss": (_original, _reconstructed, _masked),
}
