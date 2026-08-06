"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_sizes = [32, 64, 64, 8]
_layers = []
for _n_in, _n_out in zip(_sizes, _sizes[1:]):
    _w = [[random.uniform(-1.0, 1.0) for _ in range(_n_in)] for _ in range(_n_out)]
    _layers.append((_w, [0.0] * _n_out))

_inputs = [random.uniform(-1.0, 1.0) for _ in range(32)]

BENCH = {
    "sigmoid": (0.7,),
    "layer_forward": (_layers[0][0], _layers[0][1], _inputs),
    "network_forward": (_layers, _inputs),
    "predict_binary": (0.7,),
    "xor_forward": (1, 0),
    "layer_shapes": (_layers,),
    "count_parameters": (_sizes,),
    "init_network": (_sizes, 0),
}
