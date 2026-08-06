"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_values = [random.uniform(-5, 5) for _ in range(2000)]
_raw = [random.random() for _ in range(2000)]
_total = sum(_raw)
_probs = [p / _total for p in _raw]

_logits = [random.uniform(-20, 20) for _ in range(2000)]

# таблица 60x60, нормированная до суммы 1
_rows = [[random.random() for _ in range(60)] for _ in range(60)]
_grand = sum(sum(r) for r in _rows)
_joint = [[c / _grand for c in r] for r in _rows]

BENCH = {
    "expected_value": (_values, _probs),
    "variance": (_values, _probs),
    "normal_pdf": (0.7, 0.0, 1.0),
    "softmax": (_logits,),
    "log_softmax": (_logits,),
    "cross_entropy_loss": (_logits, 0),
    "marginals": (_joint,),
    "is_independent": (_joint,),
}
