"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

# линейно разделимый набор: метка = знак суммы координат
_data = []
for _ in range(400):
    _point = [random.uniform(-1.0, 1.0) for _ in range(8)]
    _data.append((_point, 1 if sum(_point) > 0 else 0))

_weights = [1.0] * 8

BENCH = {
    "step": (0.5,),
    "perceptron_output": (_weights, -0.5, _data[0][0]),
    "update_once": (_weights, -0.5, _data[0][0], 1, 0.1),
    "train_perceptron": (_data, 0.1, 20),
    "accuracy": (_weights, 0.0, _data),
    "perceptron_converged": (_data, 20),
    "xor_network": (1, 0),
}
