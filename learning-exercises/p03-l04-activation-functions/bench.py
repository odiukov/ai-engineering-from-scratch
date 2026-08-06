"""Входные данные для замера скорости."""

import math
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_logits = [random.gauss(0.0, 3.0) for _ in range(512)]

BENCH = {
    "sigmoid": (0.7,),
    "d_sigmoid": (0.7,),
    "tanh_act": (0.7,),
    "d_tanh": (0.7,),
    "relu": (0.7,),
    "d_relu": (0.7,),
    "leaky_relu": (-0.7,),
    "d_leaky_relu": (-0.7,),
    "gelu": (0.7,),
    "d_gelu": (0.7,),
    "swish": (0.7,),
    "d_swish": (0.7,),
    "softmax": (_logits,),
    "dead_zone_fraction": (math.cos, -5.0, 5.0, 2000),
    "max_derivative": (math.cos, -5.0, 5.0, 2000),
}
