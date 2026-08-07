"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [f"w{i}" for i in range(300)]
_DOCS = [[random.choice(_WORDS) for _ in range(50)] for _ in range(80)]
_DIM = 24
_W = [[random.gauss(0, 0.1) for _ in range(_DIM)] for _ in _WORDS]
_W_TILDE = [[random.gauss(0, 0.1) for _ in range(_DIM)] for _ in _WORDS]
_B = [0.0] * len(_WORDS)
_B_TILDE = [0.0] * len(_WORDS)

_ALPHABET = "abcdefghijklmnopqrst"
_CORPUS = {
    "".join(random.choice(_ALPHABET) for _ in range(random.randint(4, 9))): random.randint(1, 50)
    for _ in range(300)
}
_MERGES = [("a", "b"), ("ab", "c"), ("d", "e"), ("abc", "de")]
_NGRAM_TABLE = {g: [random.random() for _ in range(8)] for g in _ALPHABET}
_TOKENS = [random.choice("abc") for _ in range(5000)]

BENCH = {
    "build_cooccurrence": (_DOCS, 5),
    "glove_weight": (37.0,),
    "glove_step": (_W, _W_TILDE, _B, _B_TILDE, 3, 7, 12.0, 0.01),
    "char_ngrams": ("internationalization",),
    "fasttext_vector": ("internationalization", _NGRAM_TABLE),
    "merge_pair": (_TOKENS, ("a", "b")),
    "learn_bpe": (_CORPUS, 40),
    "apply_bpe": ("abcdeabcde", _MERGES),
}
