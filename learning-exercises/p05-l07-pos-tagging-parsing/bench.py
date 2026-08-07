"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_TAGS = ["DET", "NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "CCONJ"]
_WORDS = [f"w{i}" for i in range(200)]

_TRAIN = []
for _ in range(200):
    _n = random.randint(6, 14)
    _tokens = [random.choice(_WORDS) for _ in range(_n)]
    _tags = [random.choice(_TAGS) for _ in range(_n)]
    _TRAIN.append((_tokens, _tags))

_WORD_BEST = {w: random.choice(_TAGS) for w in _WORDS}
_SENT = [random.choice(_WORDS) for _ in range(25)]
_COUNTS = {t: random.randint(1, 50) for t in _TAGS}

_TRANSITIONS = {"<BOS>": {t: random.randint(1, 20) for t in _TAGS}}
for _t in _TAGS:
    _TRANSITIONS[_t] = {u: random.randint(1, 20) for u in _TAGS}
_EMISSIONS = {t: {w: random.randint(1, 5) for w in _WORDS} for t in _TAGS}

_ARC_TOKENS = [f"t{i}" for i in range(300)]
_ARCS = [(-1, 0, "ROOT")]
for _i in range(1, 300):
    _ARCS.append((_i - 1, _i, random.choice(["nsubj", "dobj", "det", "prep"])))

BENCH = {
    "ptb_to_ud": ("NNS",),
    "train_mft": (_TRAIN,),
    "predict_mft": (_SENT, _WORD_BEST, "NOUN"),
    "tag_accuracy": ([random.choice(_TAGS) for _ in range(5000)],
                     [random.choice(_TAGS) for _ in range(5000)]),
    "count_hmm": (_TRAIN,),
    "laplace_logprob": (_COUNTS, "NOUN", 12, 0.01),
    "viterbi": (_SENT, _TRANSITIONS, _EMISSIONS, _TAGS, _WORDS, 0.01),
    "extract_svo": (_ARC_TOKENS, _ARCS),
}
