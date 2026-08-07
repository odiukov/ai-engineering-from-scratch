"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_LAYERS = 24
_WIDTH = 64

_x = [random.uniform(-1.0, 1.0) for _ in range(_WIDTH)]
_params = [
    (
        [random.uniform(0.6, 1.2) for _ in range(_WIDTH)],
        [random.uniform(-0.5, 0.6) for _ in range(_WIDTH)],
        [random.uniform(0.6, 1.2) for _ in range(_WIDTH)],
        [random.uniform(-0.2, 0.2) for _ in range(_WIDTH)],
    )
    for _ in range(_LAYERS)
]
_grad_y = [1.0] * _WIDTH


def _forward(x, params):
    """Копия прямого прохода: bench не должен зависеть от твоего exercise.py."""
    acts = [x]
    cur = x
    for w1, b1, w2, b2 in params:
        cur = [w2[j] * max(0.0, w1[j] * cur[j] + b1[j]) + b2[j] for j in range(len(cur))]
        acts.append(cur)
    return acts


_activations = _forward(_x, _params)
_segment = 5
_saved = [_activations[i] for i in range(0, _LAYERS, _segment)]

BENCH = {
    "layer_forward": (_x, _params[0]),
    "layer_backward": (_x, _params[0], _grad_y),
    "forward_store_all": (_x, _params),
    "backward_store_all": (_grad_y, _activations, _params),
    "forward_checkpointed": (_x, _params, _segment),
    "backward_checkpointed": (_grad_y, _saved, _params, _segment),
    "checkpoint_budget": (64, 1000, 8),
    "optimal_segment": (256, 3),
}
