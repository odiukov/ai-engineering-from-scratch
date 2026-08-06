"""Входные данные для замера скорости."""

import random

random.seed(0)

_rng = random.Random(0)
_Z_DIM, _W_DIM, _HIDDEN, _DEPTH, _NUM_BLOCKS = 32, 32, 64, 8, 6


def _mat(rows, cols):
    return [[_rng.gauss(0, 0.3) for _ in range(cols)] for _ in range(rows)]


_z = [_rng.gauss(0, 1) for _ in range(_Z_DIM)]
_layers = [(_mat(_W_DIM, _Z_DIM if i == 0 else _W_DIM), [0.0] * _W_DIM) for i in range(_DEPTH)]

_w = [_rng.gauss(0, 1) for _ in range(_W_DIM)]
_ws = [[_rng.gauss(0, 1) for _ in range(_W_DIM)] for _ in range(4000)]
_w_mean = [0.0] * _W_DIM

_features = [_rng.gauss(0, 1) for _ in range(4000)]

_const = [_rng.gauss(0, 0.3) for _ in range(_HIDDEN)]
_blocks = [
    {
        "W": _mat(_HIDDEN, _HIDDEN),
        "b": [0.0] * _HIDDEN,
        "scale_w": [_rng.gauss(0, 0.3) for _ in range(_W_DIM)],
        "bias_w": [_rng.gauss(0, 0.3) for _ in range(_W_DIM)],
    }
    for _ in range(_NUM_BLOCKS)
]
_w_per_layer = [_w] * _NUM_BLOCKS

BENCH = {
    "leaky_relu": (-0.7,),
    "mapping_network": (_z, _layers),
    "adain": (_features, 1.5, -0.5),
    "modulate": (_w, _blocks[0]["scale_w"], _blocks[0]["bias_w"]),
    "average_w": (_ws,),
    "truncate_w": (_w, _w_mean, 0.7),
    "style_mixing": (_w, _w_mean, _NUM_BLOCKS, 3),
    "synthesis": (_const, _blocks, _w_per_layer, 0.0, None, True),
}
