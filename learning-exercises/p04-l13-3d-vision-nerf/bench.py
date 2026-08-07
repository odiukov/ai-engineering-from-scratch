"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)   # обязательно: замер должен быть воспроизводим

# облако из 3000 точек в R^3 и общий MLP 3 -> 64 (как первый слой PointNet)
_cloud = [[_rng.uniform(-1.0, 1.0) for _ in range(3)] for _ in range(3000)]
_weights = [[_rng.uniform(-1.0, 1.0) for _ in range(3)] for _ in range(64)]
_biases = [_rng.uniform(-0.1, 0.1) for _ in range(64)]

# один луч: 20000 отсчётов вдоль него
_n = 20000
_t_vals = [2.0 + 4.0 * i / (_n - 1) for i in range(_n)]
_sigmas = [_rng.uniform(0.0, 2.0) for _ in range(_n)]
_colors = [[_rng.random(), _rng.random(), _rng.random()] for _ in range(_n)]
_alphas = [_rng.uniform(0.0, 0.05) for _ in range(_n)]

# длинный вектор координат, чтобы кодирование считалось десятки миллисекунд
_long_point = [_rng.uniform(-2.0, 2.0) for _ in range(2000)]

BENCH = {
    "shared_mlp": (_cloud[0], _weights, _biases),
    "pointnet_global_feature": (_cloud, _weights, _biases),
    "positional_encoding": (_long_point, 10),
    "sample_ray": ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], _t_vals),
    "segment_deltas": (_t_vals,),
    "alpha_from_density": (1.5, 0.25),
    "transmittance": (_alphas,),
    "volumetric_render": (_sigmas, _colors, _t_vals),
}
