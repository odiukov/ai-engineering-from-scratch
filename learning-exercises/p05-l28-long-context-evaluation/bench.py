"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ("the quick brown fox jumps over the lazy dog near a quiet river bank "
          "while birds sing and clouds drift slowly across an orange sky").split()

_FILLER = " ".join(random.choice(_WORDS) for _ in range(20000))
_NEEDLE = "the magic word is pineapple"
_NEEDLES = ["the magic word is pineapple",
            "the magic word is compass",
            "the magic word is whisper"]
_HAYSTACK = _FILLER

_CHAIN = []
_CHAIN.append("X0 = 1")
for _i in range(1, 300):
    _CHAIN.append(" ".join(random.choice(_WORDS) for _ in range(30)))
    _CHAIN.append(f"X{_i} = X{_i - 1} + {random.randrange(1, 9)}")
_TRACE_TEXT = " ".join(_CHAIN)


def _model(context, question):
    return "The magic word is pineapple." if "pineapple" in context else "no answer"


def _trial(depth, length):
    return 1 if length <= 8000 and (depth <= 0.25 or depth >= 0.75) else 0


_DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
_LENGTHS = [1000, 4000, 16000, 64000]
_GRID = {(n, d): _trial(d, n) for n in _LENGTHS for d in _DEPTHS}
_RATES = {1000: 1.0, 4000: 1.0, 16000: 0.4, 64000: 0.1}

BENCH = {
    "build_haystack": (_FILLER, _NEEDLE, 0.5, 50000),
    "insert_needles": (_FILLER, _NEEDLES, [0.2, 0.5, 0.8]),
    "score_needle": (_HAYSTACK, "What is the magic word?", "pineapple", _model),
    "score_multi_needle": (_HAYSTACK, "q", ["pineapple", "compass", "whisper"], _model),
    "niah_grid": (_DEPTHS, _LENGTHS, _trial),
    "pass_rates": (_GRID, "depth"),
    "effective_length": (_RATES, 0.9),
    "trace_variables": (_TRACE_TEXT,),
}
