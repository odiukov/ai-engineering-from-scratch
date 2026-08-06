"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [f"w{i}" for i in range(300)]

_docs = []
_labels = []
for _i in range(400):
    _label = "spam" if _i % 2 else "ham"
    _pool = _WORDS[:150] if _label == "spam" else _WORDS[150:]
    _docs.append(" ".join(random.choice(_pool) for _ in range(12)))
    _labels.append(_label)

_query = " ".join(random.choice(_WORDS) for _ in range(40))

BENCH = {
    "bayes_posterior": (0.0001, 0.99, 0.01),
    "sequential_posterior": (0.0001, 0.99, 0.01, 20000),
    "mle_probability": (7, 10),
    "laplace_probability": (0, 10, 5),
    "beta_update": ((1, 1), 7, 3),
    "beta_mean": ((13, 9),),
    "beta_map": ((13, 9),),
    "naive_bayes_predict": (_docs, _labels, _query),
}
