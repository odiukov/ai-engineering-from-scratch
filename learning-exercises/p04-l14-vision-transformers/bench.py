"""Входные данные для замера скорости."""

import random

_rng = random.Random(0)  # обязательно: замер должен быть воспроизводим

# картинка 96x96, патч 8 -> сетка 12x12 = 144 патча по 64 числа
_H = _W = 96
_P = 8
_image = [[_rng.uniform(-1.0, 1.0) for _ in range(_W)] for _ in range(_H)]
_patches = [
    [_rng.uniform(-1.0, 1.0) for _ in range(_P * _P)]
    for _ in range((_H // _P) * (_W // _P))
]

# проекция патча 64 -> токен 64 (в уроке это 256 -> 192)
_DIM = 64
_Wproj = [[_rng.uniform(-0.1, 0.1) for _ in range(_P * _P)] for _ in range(_DIM)]
_bias = [_rng.uniform(-0.1, 0.1) for _ in range(_DIM)]

# последовательность из 144 токенов + [CLS]
_tokens = [[_rng.uniform(-1.0, 1.0) for _ in range(_DIM)] for _ in range(144)]
_cls = [_rng.uniform(-1.0, 1.0) for _ in range(_DIM)]
_pos = [[_rng.uniform(-0.1, 0.1) for _ in range(_DIM)] for _ in range(145)]

# внимание на 96 токенах размерности 64: 96*96*64 умножений на каждую из двух матриц
_QKV = [[_rng.uniform(-1.0, 1.0) for _ in range(64)] for _ in range(96)]

_scores = [_rng.uniform(-10.0, 10.0) for _ in range(4096)]

_vec = [_rng.uniform(-3.0, 3.0) for _ in range(768)]
_gamma = [1.0] * 768
_beta = [0.0] * 768
_sublayer = lambda v: [x * 0.5 for x in v]

BENCH = {
    "patchify": (_image, _P),
    "unpatchify": (_patches, _P, _H, _W),
    "patch_embed": (_image, _P, _Wproj, _bias),
    "add_cls_and_positions": (_tokens, _cls, _pos),
    "softmax": (_scores,),
    "scaled_dot_product_attention": (_QKV, _QKV, _QKV),
    "layer_norm": (_vec, _gamma, _beta),
    "prenorm_residual": (_vec, _gamma, _beta, _sublayer),
}
