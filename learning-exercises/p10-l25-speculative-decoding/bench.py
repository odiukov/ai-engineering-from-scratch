"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 256


def _normalized(values):
    total = sum(values)
    return [v / total for v in values]


_p = _normalized([random.random() for _ in range(_VOCAB)])
_q = _normalized([random.random() for _ in range(_VOCAB)])

_K = 4
_p_rows = [_p] * (_K + 1)
_q_rows = [_q] * _K
_draft_tokens = [random.randrange(_VOCAB) for _ in range(_K)]
_rng = random.Random(0)

# бинарное дерево черновика EAGLE: 63 узла, глубина 5
_parents = [-1] + [(i - 1) // 2 for i in range(1, 63)]
_accepted = [random.random() < 0.7 for _ in range(63)]
_accepted[0] = True

BENCH = {
    "sample_from": (_p, _rng),
    "expected_tokens": (0.8, 4),
    "speculative_speedup": (0.8, 4, 0.05),
    "acceptance_rate": (_p, _q),
    "residual_distribution": (_p, _q),
    "speculative_step": (_p_rows, _q_rows, _draft_tokens, _rng),
    "tree_attention_mask": (_parents,),
    "longest_accepted_path": (_parents, _accepted),
}
