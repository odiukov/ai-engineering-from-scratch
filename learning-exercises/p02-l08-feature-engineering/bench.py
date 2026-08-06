"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_numbers = [random.uniform(0, 1000) for _ in range(2000)]
_with_holes = [v if random.random() > 0.1 else None for v in _numbers]
_categories = [random.choice("abcdefghij") for _ in range(2000)]
_targets = [random.gauss(100, 30) for _ in range(2000)]
_labels = [1 if v > 500 else 0 for v in _numbers]
_words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
_documents = [
    " ".join(random.choice(_words) for _ in range(30)) for _ in range(200)
]

BENCH = {
    "min_max_scale": (_numbers,),
    "standardize": (_numbers,),
    "bin_values": (_numbers, 10),
    "impute_median": (_with_holes,),
    "one_hot_encode": (_categories,),
    "target_encode": (_categories, _targets, 10),
    "tfidf": (_documents,),
    "mutual_information": (_numbers, _labels, 10),
}
