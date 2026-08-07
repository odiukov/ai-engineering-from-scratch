"""Входные данные для замера скорости."""

import random

random.seed(0)

_DIM = 128
_query = [random.gauss(0, 1) for _ in range(_DIM)]
_documents = [
    (f"lang{i % 20}:doc{i}", [random.gauss(0, 1) for _ in range(_DIM)])
    for i in range(500)
]
_labels = {
    f"label{i}": [random.gauss(0, 1) for _ in range(_DIM)] for i in range(20)
}

_PIECES = ["anti", "bio", "tico", "body", "establish", "ment", "ing", "pre", "ion"]
_vocab = set(_PIECES)
_texts = [
    " ".join("".join(random.sample(_PIECES, 3)) for _ in range(30)) for _ in range(50)
]

_FEATURES = ["word_order", "gender", "case", "article", "tone", "plural", "negation"]
_make = lambda: {f: random.choice(["yes", "no", "maybe"]) for f in _FEATURES}
_target = _make()
_candidates = {f"lang{i}": (_make(), 10 ** random.randint(4, 9)) for i in range(100)}

_records = [
    (f"lang{i % 30}", random.randint(0, 3), random.randint(0, 3)) for i in range(20000)
]

BENCH = {
    "cosine_similarity": (_query, _documents[0][1]),
    "cross_lingual_retrieve": (_query, _documents, 5),
    "zero_shot_classify": (_query, _labels),
    "subword_segment": ("antiestablishmentbiotico" * 4, _vocab),
    "tokenization_fertility": (_texts, _vocab),
    "language_similarity": (_target, _candidates["lang0"][0]),
    "rank_source_languages": (_target, _candidates, 0.5),
    "per_language_accuracy": (_records,),
}
