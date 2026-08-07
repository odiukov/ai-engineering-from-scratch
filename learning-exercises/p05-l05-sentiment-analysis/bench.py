"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = [f"w{i}" for i in range(400)]
_TOKENS = [random.choice(_VOCAB + ["not", "never", ".", ","]) for _ in range(20_000)]
_DOCS_POS = [[random.choice(_VOCAB) for _ in range(40)] for _ in range(150)]
_DOCS_NEG = [[random.choice(_VOCAB) for _ in range(40)] for _ in range(150)]
_DOCS_BY_CLASS = {"pos": _DOCS_POS, "neg": _DOCS_NEG}

_PRIORS = {"pos": 0.5, "neg": 0.5}
_PROBS = {
    "pos": {w: random.random() for w in _VOCAB},
    "neg": {w: random.random() for w in _VOCAB},
}
_LONG_DOC = [random.choice(_VOCAB) for _ in range(5000)]

_X = [[random.gauss(0, 1) for _ in range(30)] for _ in range(400)]
_Y = [1 if row[0] > 0 else 0 for row in _X]
_W = [random.gauss(0, 1) for _ in range(30)]

_Y_TRUE = [random.randint(0, 1) for _ in range(20_000)]
_Y_PRED = [random.randint(0, 1) for _ in range(20_000)]

BENCH = {
    "apply_negation": (_TOKENS,),
    "train_nb": (_DOCS_BY_CLASS, _VOCAB),
    "predict_nb": (_LONG_DOC, _PRIORS, _PROBS),
    "sigmoid": (0.7,),
    "train_lr": (_X, _Y, 20, 0.05, 0.01),
    "predict_lr": (_X, _W, 0.0),
    "evaluate": (_Y_TRUE, _Y_PRED),
    "macro_f1": (_Y_TRUE, _Y_PRED),
}
