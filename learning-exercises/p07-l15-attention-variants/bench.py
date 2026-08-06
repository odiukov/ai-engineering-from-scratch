"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_n = 900
_d = 24
_q1 = [random.gauss(0, 1) for _ in range(_d)]
_q2 = [random.gauss(0, 1) for _ in range(_d)]
_K1 = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n)]
_K2 = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n)]
_V = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n)]

# маска последней строки: половина позиций закрыта окном
_row = [0.0 if j >= _n - 256 else float("-inf") for j in range(_n)]

_mask = [[0.0 if j <= i else float("-inf") for j in range(160)] for i in range(160)]

BENCH = {
    "causal_mask": (160,),
    "swa_mask": (160, 32),
    "strided_mask": (160, 32, 8),
    "count_attended": (_mask,),
    "effective_receptive_field": (32, 1024),
    "masked_attention_row": (_q1, _K1, _V, _row),
    "diff_attention_row": (_q1, _q2, _K1, _K2, _V, _row, 0.5),
    "kv_cache_bytes": (80, 8, 128, 131072, 2, 1024),
}
