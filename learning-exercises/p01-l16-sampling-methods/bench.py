"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = 20000

# логиты "словаря" — как их отдаёт последний слой языковой модели
_logits = [random.gauss(0.0, 3.0) for _ in range(_VOCAB)]

# нормированный вектор вероятностей той же длины
_raw = [random.random() for _ in range(_VOCAB)]
_total = sum(_raw)
_probs = [x / _total for x in _raw]

# отдельный генератор: замер не должен зависеть от глобального состояния
_rng = random.Random(0)

# треугольная плотность на [-1, 1] с верхней границей 1 — принимается ~50%
_tent = lambda x: 1.0 - abs(x)

BENCH = {
    "softmax_with_temperature": (_logits, 0.8),
    "sample_index": (_probs, _rng),
    "sample_exponential": (2.0, _rng),
    "monte_carlo_pi": (20000, _rng),
    "rejection_sample": (_tent, -1.0, 1.0, 1.0, _rng),
    "top_k_filter": (_logits, 50),
    "top_p_filter": (_logits, 0.9),
    "sample_token": (_logits, _rng, 0.8, 50, 0.9),
}
