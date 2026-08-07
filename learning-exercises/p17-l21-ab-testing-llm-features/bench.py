"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_p_values = [random.random() ** 3 for _ in range(50000)]

BENCH = {
    "normal_cdf": (1.3,),
    "z_quantile": (0.975,),
    "sample_size": (0.03, 0.05),
    "wilson_interval": (45000, 50000),
    "proportion_test": (3000, 100000, 3600, 100000),
    "srm_check": (49700, 50300),
    "benjamini_hochberg": (_p_values,),
    "run_experiment": (0.10, 0.12, 40000, random.Random(0)),
}
