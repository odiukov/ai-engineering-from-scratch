"""Входные данные для замера скорости."""

import math
import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_relu = lambda t: max(0.0, t)
_matrix = [[random.gauss(0.0, 1.0) for _ in range(120)] for _ in range(120)]
_vector = [random.gauss(0.0, 1.0) for _ in range(120)]
_values = [random.gauss(0.0, 1.0) for _ in range(20000)]
_identical = [[1.0] * 120 for _ in range(120)]


def _kaiming(fan_in, fan_out, seed=0):
    """Обёртка нужна, чтобы bench не зависел от того, чей модуль загружен."""
    std = math.sqrt(2.0 / fan_in)
    rng = random.Random(seed)
    return [[rng.gauss(0.0, std) for _ in range(fan_in)] for _ in range(fan_out)]


BENCH = {
    "zero_init": (300, 300),
    "random_init": (200, 200, 1.0, 0),
    "xavier_init": (200, 200, 0),
    "kaiming_init": (200, 200, 0),
    "variance": (_values,),
    "matvec": (_matrix, _vector),
    "is_symmetry_broken": (_identical,),
    "forward_magnitudes": (_kaiming, _relu, 12, 48, 0),
    "recommend_init": ("leaky_relu",),
}
