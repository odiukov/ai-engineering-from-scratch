"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ACTIONS = ("up", "down", "left", "right")
_rng = random.Random(0)
_row = {a: random.gauss(0.0, 1.0) for a in _ACTIONS}

# таблица «как после обучения»: все клетки 4x6, кроме обрыва
_Q = {
    (r, c): {a: random.gauss(0.0, 5.0) for a in _ACTIONS}
    for r in range(4)
    for c in range(6)
    if not (r == 3 and 1 <= c <= 4)
}

_ranges = {
    "slip": (0.0, 0.3),
    "mass": (0.8, 1.2),
    "friction": (0.5, 1.5),
    "motor_delay": (0.0, 0.05),
}

BENCH = {
    "perpendicular": ("up",),
    "slip_step": ((2, 2), "right", 0.2, _rng),
    "randomize": (_rng, _ranges),
    "epsilon_greedy": (_row, _rng, 0.15),
    "train_q": ((0.0, 0.3), 600),
    "evaluate": (_Q, 0.2, _rng, 100),
    "sweep": (_Q, [0.0, 0.1, 0.2, 0.3, 0.5], _rng, 40),
    "widen_range": ((0.0, 0.1), -9.0, -12.0),
}
