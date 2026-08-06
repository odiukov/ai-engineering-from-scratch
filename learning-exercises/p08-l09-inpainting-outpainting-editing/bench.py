"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_N = 4096                                # аналог латента 64x64 в одну строку

_T = 1000
_bars, _cum = [], 1.0
for _t in range(_T):
    _beta = 1e-4 + (0.02 - 1e-4) * _t / (_T - 1)
    _cum *= (1.0 - _beta)
    _bars.append(_cum)

_mask = [_rng.random() < 0.3 for _ in range(_N)]
_clean = [_rng.gauss(0, 1) for _ in range(_N)]
_x_t = [_rng.gauss(0, 1) for _ in range(_N)]

BENCH = {
    "invert_mask": (_mask,),
    "dilate_mask": (_mask, 8),
    "build_inpaint_input": (_x_t, _clean, _mask),
    "reinject_known": (_x_t, _mask, _clean, 500, _bars, _rng),
    "restore_known": (_x_t, _mask, _clean),
    "sdedit_start_step": (0.6, _T),
    "sdedit_noise": (_clean, 600, _bars, _rng),
    "repaint_timesteps": (_T, 10, 5),
}
