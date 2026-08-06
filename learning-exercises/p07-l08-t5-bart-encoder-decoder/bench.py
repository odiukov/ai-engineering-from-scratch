"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_n_src, _n_tgt, _d = 64, 64, 32
_tokens = [f"t{i}" for i in range(400)]
_ids = list(range(400))
_Q = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n_tgt)]
_K = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n_src)]
_V = [[random.gauss(0, 1) for _ in range(_d)] for _ in range(_n_src)]
_spans = [(i * 8 + 1, 3) for i in range(40)]
_source, _target = None, None

BENCH = {
    "softmax": ([random.gauss(0, 1) for _ in range(2000)],),
    "cross_attention": (_Q, _K, _V),
    "shift_right": (_ids, 0),
    "pick_spans": (400, random.Random(0), 0.15, 3.0),
    "corrupt_spans": (_tokens, _spans),
    "round_trip": (
        _tokens[:1] + [f"<extra_id_{i}>" for i in range(40)] + _tokens[1:],
        [x for i in range(40) for x in (f"<extra_id_{i}>", *_tokens[i * 3:i * 3 + 3])]
        + ["<extra_id_40>"],
    ),
    "text_infill": (_tokens, _spans),
    "document_rotate": (_tokens, 137),
}
