"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

# батч 400 образцов по 128 нейронов, примерно треть из них молчит
_activations = [
    [max(0.0, random.gauss(-0.3, 1.0)) for _ in range(128)]
    for _ in range(400)
]
_clean = [[random.gauss(0.0, 1.0) for _ in range(200)] for _ in range(200)]
_named = {f"layer{i}": _clean for i in range(20)}
_losses = [1.0 / (i + 1) for i in range(5000)]
_point = [random.gauss(0.0, 1.0) for _ in range(150)]

_quadratic = lambda p: sum(v * v for v in p)
_analytic = [2 * v for v in _point]

BENCH = {
    "has_nan_or_inf": (_clean,),
    "find_bad_gradients": (_named,),
    "dead_relu_fractions": (_activations,),
    "dead_neurons": (_activations, 1.0),
    "numeric_gradient": (_quadratic, _point),
    "relative_difference": (1.0, 1.001),
    "gradient_check": (_quadratic, _analytic, _point),
    "can_overfit_one_batch": (lambda p: (p[0] - 3.0) ** 2, [0.0], 200, 0.1, 1e-3),
    "diagnose_loss_curve": (_losses,),
}
