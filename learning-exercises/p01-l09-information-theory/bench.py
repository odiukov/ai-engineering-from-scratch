"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _distribution(n):
    raw = [random.random() for _ in range(n)]
    total = sum(raw)
    return [x / total for x in raw]


_p = _distribution(5000)
_q = _distribution(5000)

# таблица 80x80, нормированная до суммы 1
_rows = [[random.random() for _ in range(80)] for _ in range(80)]
_grand = sum(sum(r) for r in _rows)
_joint = [[c / _grand for c in r] for r in _rows]

BENCH = {
    "information_content": (0.001,),
    "entropy": (_p,),
    "cross_entropy": (_p, _q),
    "kl_divergence": (_p, _q),
    "joint_entropy": (_joint,),
    "conditional_entropy": (_joint,),
    "mutual_information": (_joint,),
    "perplexity": (_p, _q),
}
