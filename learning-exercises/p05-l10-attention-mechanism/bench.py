"""Входные данные для замера скорости."""

import random

random.seed(0)

_T_ENC = 400
_D_H = 64
_D_S = 64
_D_ATTN = 32

_encoder = [[random.uniform(-1, 1) for _ in range(_D_H)] for _ in range(_T_ENC)]
_query = [random.uniform(-1, 1) for _ in range(_D_S)]
_scores = [random.uniform(-4, 4) for _ in range(_T_ENC)]
_mask = [i % 5 != 0 for i in range(_T_ENC)]

_W = [[random.uniform(-0.3, 0.3) for _ in range(_D_H)] for _ in range(_D_S)]
_W_a = [[random.uniform(-0.3, 0.3) for _ in range(_D_S)] for _ in range(_D_ATTN)]
_U_a = [[random.uniform(-0.3, 0.3) for _ in range(_D_H)] for _ in range(_D_ATTN)]
_v_a = [random.uniform(-1, 1) for _ in range(_D_ATTN)]

_decoder = [[random.uniform(-1, 1) for _ in range(_D_H)] for _ in range(30)]

BENCH = {
    "softmax": (_scores,),
    "masked_softmax": (_scores, _mask),
    "dot_score": (_query, _encoder),
    "general_score": (_query, _encoder, _W),
    "additive_score": (_query, _encoder, _W_a, _U_a, _v_a),
    "attend": (_scores, _encoder),
    "alignment_matrix": (_decoder, _encoder),
    "multi_head_dot_attention": (_query, _encoder, _encoder, 8),
}
