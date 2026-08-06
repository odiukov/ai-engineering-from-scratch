"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_vocab = 512
_n = 96

_logits = [random.gauss(0, 1) for _ in range(_vocab)]
_probs = [1.0 / _vocab] * _vocab
_scores = [[random.gauss(0, 1) for _ in range(_n)] for _ in range(_n)]
_logits_per_pos = [[random.gauss(0, 1) for _ in range(_vocab)] for _ in range(_n)]
_tokens = [random.randrange(_vocab) for _ in range(_n)]

BENCH = {
    "softmax": (_logits, 0.7),
    "causal_mask": (_n * 4,),
    "prefix_average_matrix": (_n * 4,),
    "causal_attention_weights": (_scores,),
    "cross_entropy_shifted": (_logits_per_pos, _tokens),
    "top_k_filter": (_probs, 50),
    "top_p_filter": (_probs, 0.9),
    "min_p_filter": (_probs, 0.05),
}
