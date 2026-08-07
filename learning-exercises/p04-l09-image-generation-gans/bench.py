"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)

_real_logits = [_rng.gauss(2.0, 1.0) for _ in range(2000)]
_fake_logits = [_rng.gauss(-2.0, 1.0) for _ in range(2000)]
_matrix = [[_rng.gauss(0.0, 1.0) for _ in range(60)] for _ in range(60)]
_samples = [[_rng.gauss(0.0, 1.0) for _ in range(64)] for _ in range(120)]

BENCH = {
    "sigmoid": (1.5,),
    "bce_with_logits": (1.5, 1),
    "discriminator_loss": (_real_logits, _fake_logits),
    "generator_loss": (_fake_logits,),
    "generator_loss_grad": (-3.0,),
    "conv_transpose_output_size": (16, 4, 2, 1),
    "power_iteration_sigma": (_matrix, random.Random(0), 50),
    "mode_collapse_score": (_samples,),
}
