"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)

# вероятности от «почти обученного» дискриминатора: настоящие ближе к 1,
# подделки ближе к 0, но перекрытие есть — как в реальном логе
_d_real = [min(max(_rng.gauss(0.75, 0.15), 1e-6), 1 - 1e-6) for _ in range(2000)]
_d_fake = [min(max(_rng.gauss(0.30, 0.15), 1e-6), 1 - 1e-6) for _ in range(2000)]

_samples = [
    _rng.gauss(-2.0, 0.4) if _rng.random() < 0.5 else _rng.gauss(2.0, 0.4)
    for _ in range(4000)
]

BENCH = {
    "sigmoid": (0.7,),
    "binary_cross_entropy": (0.73, 1.0),
    "binary_cross_entropy_grad": (0.73, 1.0),
    "discriminator_loss": (_d_real, _d_fake),
    "generator_loss": (_d_fake,),
    "generator_loss_grad": (_d_fake,),
    "optimal_discriminator": (0.3, 0.3),
    "is_mode_collapse": (_samples,),
}
