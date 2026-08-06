"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_T = 4000
_betas = [1e-4 + (0.02 - 1e-4) * t / (_T - 1) for t in range(_T)]
_bars, _cum = [], 1.0
for _b in _betas:
    _cum *= (1.0 - _b)
    _bars.append(_cum)

_eps = [_rng.gauss(0, 1) for _ in range(5000)]
_eps_hat = [_rng.gauss(0, 1) for _ in range(5000)]

_short_betas = _betas[:1500]
_short_bars = _bars[:1500]
_zero_model = lambda x, t: 0.0

BENCH = {
    "linear_beta_schedule": (_T,),
    "alpha_bars_from_betas": (_betas,),
    "forward_sample": (1.0, _T - 1, _bars, _rng),
    "sinusoidal_embedding": (137, 256),
    "ddpm_loss": (_eps, _eps_hat),
    "predict_x0": (1.0, 10, 0.5, _bars),
    "reverse_step": (0.5, 10, 0.2, _betas, _bars, _rng),
    "sample_chain": (_zero_model, _short_betas, _short_bars, _rng),
}
