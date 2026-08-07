"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)

_betas = [1e-4 + (2e-2 - 1e-4) * i / 999 for i in range(1000)]
_bars = []
_running = 1.0
for _b in _betas:
    _running *= 1.0 - _b
    _bars.append(_running)

_image = [_rng.gauss(0.0, 1.0) for _ in range(3 * 64 * 64)]
_noise = [_rng.gauss(0.0, 1.0) for _ in range(3 * 64 * 64)]

BENCH = {
    "linear_beta_schedule": (1000, 1e-4, 2e-2),
    "alphas_cumprod": (_betas,),
    "q_step": (_image, 0.02, _noise),
    "q_sample": (_image, _noise, 0.3),
    "predict_x0": (_image, _noise, 0.3),
    "timestep_embedding": (500, 256),
    "ddpm_step": (_image, _noise, 500, _betas, _bars, _noise),
    "ddim_step": (_image, _noise, 0.3, 0.5),
}
