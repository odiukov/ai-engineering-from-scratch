"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_scores = [_rng.gauss(0, 3) for _ in range(4000)]

_d = 64
_n_tokens = 77                      # ровно столько токенов у CLIP-L в SD 1.x
_query = [_rng.gauss(0, 1) for _ in range(_d)]
_keys = [[_rng.gauss(0, 1) for _ in range(_d)] for _ in range(_n_tokens)]
_values = [[_rng.gauss(0, 1) for _ in range(_d)] for _ in range(_n_tokens)]

_eps_cond = [_rng.gauss(0, 1) for _ in range(4096)]
_eps_uncond = [_rng.gauss(0, 1) for _ in range(4096)]

BENCH = {
    "encode": (10.0,),
    "decode": (1.8215,),
    "latent_compression_ratio": (512, 512, 3, 8, 4),
    "softmax": (_scores,),
    "cross_attention": (_query, _keys, _values),
    "drop_label_for_cfg": (1, 2, 0.1, _rng),
    "classifier_free_guidance": (_eps_cond, _eps_uncond, 3.0),
}
