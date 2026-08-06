"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = [f"w{i:04d}" for i in range(300)]
_N_DOCS = 200

# документы разной длины: тексты в реальности сильно разреженные
_docs = [
    [random.choice(_VOCAB) for _ in range(random.randint(20, 60))]
    for _ in range(_N_DOCS)
]
_X = [
    [0] * len(_VOCAB) for _ in range(_N_DOCS)
]
_position = {w: i for i, w in enumerate(_VOCAB)}
for _row, _doc in zip(_X, _docs):
    for _token in _doc:
        _row[_position[_token]] += 1

_y = ["spam" if i % 3 else "ham" for i in range(_N_DOCS)]
_model = {
    "classes": ["ham", "spam"],
    "log_priors": {"ham": -1.0986, "spam": -0.4055},
    "log_probs": {
        "ham": [-5.7] * len(_VOCAB),
        "spam": [-5.7] * len(_VOCAB),
    },
}

BENCH = {
    "build_vocabulary": (_docs,),
    "bag_of_words": (_docs[0], _VOCAB),
    "class_log_priors": (_y,),
    "feature_log_probs": (_X, _y, 1.0),
    "fit_multinomial_nb": (_X, _y, 1.0),
    "log_scores": (_model, _X[0]),
    "predict": (_model, _X),
    "predict_proba": (_model, _X[0]),
}
