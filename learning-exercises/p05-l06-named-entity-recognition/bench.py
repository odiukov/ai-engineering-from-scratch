"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_TYPES = ["ORG", "GPE", "PERSON", "PRODUCT"]
_WORDS = [f"Tok{i}" for i in range(500)]
_TOKENS = [random.choice(_WORDS) for _ in range(4000)]

_SPANS = []
_i = 0
while _i < len(_TOKENS) - 3:
    _len = random.randint(1, 3)
    _SPANS.append((_i, _i + _len, random.choice(_TYPES)))
    _i += _len + random.randint(1, 3)

_LABELS = ["O"] * len(_TOKENS)
for _s, _e, _t in _SPANS:
    _LABELS[_s] = f"B-{_t}"
    for _k in range(_s + 1, _e):
        _LABELS[_k] = f"I-{_t}"

_NOISY = [lb if random.random() > 0.2 else "O" for lb in _LABELS]
_GAZ = {t: set(random.sample(_WORDS, 60)) for t in _TYPES}
_SCORES = [
    {lb: random.random() for lb in ["O"] + [f"B-{t}" for t in _TYPES] + [f"I-{t}" for t in _TYPES]}
    for _ in range(600)
]

BENCH = {
    "spans_to_bio": (_TOKENS, _SPANS),
    "bio_to_spans": (_LABELS,),
    "is_valid_bio": (_LABELS,),
    "word_shape": ("Internationalization-2024",),
    "token_features": ("Apple", "the", "sued"),
    "rule_based_ner": (_TOKENS, _GAZ),
    "entity_f1": (_LABELS, _NOISY),
    "constrained_decode": (_SCORES,),
}
