"""Входные данные для замера скорости."""

import math

_TOTAL = 20000


def _cosine(step, lr=0.05, total_steps=_TOTAL, lr_min=0.0):
    """Копия расписания: bench не должен зависеть от того, чей модуль загружен."""
    if step >= total_steps:
        return lr_min
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * step / total_steps))


_curve = [_cosine(s) for s in range(_TOTAL)]

BENCH = {
    "constant_schedule": (12345, 0.05),
    "step_decay_schedule": (12345, 0.1, 100, 0.1),
    "cosine_schedule": (12345, 0.05, _TOTAL, 1e-5),
    "linear_warmup": (12345, 0.05, 2000),
    "warmup_cosine_schedule": (12345, 0.05, _TOTAL, 2000, 1e-5),
    "one_cycle_schedule": (12345, 0.05, _TOTAL),
    "lr_curve": (_cosine, _TOTAL),
    "peak_step": (_curve,),
    "descend": (_cosine, 10.0, _TOTAL),
}
