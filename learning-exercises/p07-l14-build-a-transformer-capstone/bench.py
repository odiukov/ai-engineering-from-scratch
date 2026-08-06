"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_d = 32
_hidden = 2 * _d
_vocab = 65
_n = 24


def _mat(rows, cols):
    return [[random.gauss(0, 0.2) for _ in range(cols)] for _ in range(rows)]


_x = [random.gauss(0, 1) for _ in range(_d)]
_W = _mat(_d, _d)
_X = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n)]

_block = {
    "n_heads": 4,
    "norm1": [1.0] * _d,
    "wq": _mat(_d, _d), "wk": _mat(_d, _d), "wv": _mat(_d, _d), "wo": _mat(_d, _d),
    "norm2": [1.0] * _d,
    "w1": _mat(_hidden, _d), "w3": _mat(_hidden, _d), "w2": _mat(_d, _hidden),
}

_params = {
    "tok_emb": _mat(_vocab, _d),
    "pos_emb": _mat(_n, _d),
    "blocks": [_block, _block],
    "norm_f": [1.0] * _d,
}
_tokens = [random.randrange(_vocab) for _ in range(_n)]
_logits = [[random.gauss(0, 1) for _ in range(_vocab)] for _ in range(_n)]

BENCH = {
    "softmax": ([random.gauss(0, 3) for _ in range(2000)],),
    "linear": (_x, _mat(512, _d)),
    "rms_norm": (_x, [1.0] * _d),
    "swiglu_ffn": (_x, _block["w1"], _block["w3"], _block["w2"]),
    "multi_head_attention": (_X, _W, _W, _W, _W, 4),
    "transformer_block": (_X, _block),
    "init_params": (_vocab, _d, 4, 2, _n, random.Random(1)),
    "gpt_forward": (_tokens, _params),
    "cross_entropy_next_token": (_logits, _tokens),
}
