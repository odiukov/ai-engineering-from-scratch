"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 512
_SEQ = 48
_DIM = 32

_logits = [random.gauss(0.0, 1.0) for _ in range(_VOCAB)]
_probs = [1.0 / _VOCAB] * _VOCAB

_rows = [[random.gauss(0.0, 1.0) for _ in range(_VOCAB)] for _ in range(_SEQ)]
_targets = [random.randrange(_VOCAB) for _ in range(_SEQ)]

_Q = [[random.gauss(0.0, 1.0) for _ in range(_DIM)] for _ in range(_SEQ)]
_K = [[random.gauss(0.0, 1.0) for _ in range(_DIM)] for _ in range(_SEQ)]
_V = [[random.gauss(0.0, 1.0) for _ in range(_DIM)] for _ in range(_SEQ)]

_vec = [random.gauss(0.0, 1.0) for _ in range(768)]
_gamma = [1.0] * 768
_beta = [0.0] * 768

BENCH = {
    "softmax": (_logits,),
    "layer_norm": (_vec, _gamma, _beta),
    "causal_attention": (_Q, _K, _V),
    "cross_entropy": (_rows, _targets),
    "d_cross_entropy": (_rows, _targets),
    "count_parameters": (50257, 768, 12, 1024, 3072),
    "top_k_top_p": (_probs, 50, 0.95),
    "sample_next_token": (_logits, random.Random(0), 0.8, 50, 0.95),
}
