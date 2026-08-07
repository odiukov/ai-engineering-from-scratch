"""Входные данные для замера скорости."""

import random

_traj = [(i, 1.05 ** i, 1.02 ** i, 1.05 ** i - 1.02 ** i) for i in range(400)]

BENCH = {
    "next_cycle": (1.0, 1.0, 1.15, 1.08),
    "race": (400, 1.05, 1.02),
    "crossing_cycle": (_traj, 1e9),
    "noisy_race": (400, 1.05, 1.02, 0.02, 0.02, random.Random(0)),
    "crossing_share": (40, 60, 1.10, 1.05, 0.02, 0.02, 1.5, random.Random(0)),
    "self_improve": (lambda x: x + 1, float, 0, 2000),
    "audit_cycles": (5000, 7),
}
