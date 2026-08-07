"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [
    "the", "cats", "were", "running", "caresses", "ponies", "hopping",
    "falling", "watched", "bled", "don't", "3pm", "singing", "better",
]

_TEXT = " ".join(random.choice(_WORDS) for _ in range(4000)) + " see https://example.com ."
_TABLE = {("cats", "NOUN"): "cat", ("running", "VERB"): "run"}

BENCH = {
    "tokenize": (_TEXT,),
    "tokenize_with_urls": (_TEXT,),
    "stem_step_1a": ("caresses",),
    "stem_step_1b": ("hopping",),
    "stem": ("caresses",),
    "lemmatize": ("cats", "NOUN", _TABLE),
    "preprocess": (_TEXT, _TABLE),
}
