"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_D = 32
_SEQ = 60  # наивный декодинг здесь квадратичен, длиннее брать незачем

_w_k = [[random.gauss(0.0, 0.2) for _ in range(_D)] for _ in range(_D)]
_w_v = [[random.gauss(0.0, 0.2) for _ in range(_D)] for _ in range(_D)]
_tokens = [[random.gauss(0.0, 1.0) for _ in range(_D)] for _ in range(_SEQ)]

_scores = [random.gauss(0.0, 3.0) for _ in range(4000)]
_query = _tokens[0]
_keys = [[random.gauss(0.0, 1.0) for _ in range(_D)] for _ in range(2000)]
_values = [[random.gauss(0.0, 1.0) for _ in range(_D)] for _ in range(2000)]

_output_lens = [random.randint(1, 400) for _ in range(4000)]

BENCH = {
    "softmax": (_scores,),
    "matvec": (_query, _w_k),
    "attention": (_query, _keys, _values),
    "generate_no_cache": (_tokens, _w_k, _w_v),
    "generate_with_cache": (_tokens, _w_k, _w_v),
    "kv_cache_bytes": (80, 8, 128, 131072, 2),
    "batching_steps": (_output_lens, 32),
    "speculative_speedup": (5, 0.78),
}
