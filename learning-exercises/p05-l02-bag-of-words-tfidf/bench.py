"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = [f"w{i}" for i in range(300)]
_DOCS = [[random.choice(_WORDS) for _ in range(40)] for _ in range(200)]
_VOCAB = {w: i for i, w in enumerate(_WORDS)}
_BOW = [[0] * len(_VOCAB) for _ in _DOCS]
for _i, _doc in enumerate(_DOCS):
    for _t in _doc:
        _BOW[_i][_VOCAB[_t]] += 1
_DF = [sum(1 for r in _BOW if r[j] > 0) for j in range(len(_VOCAB))]
_VEC_A = [random.random() for _ in range(len(_VOCAB))]
_VEC_B = [random.random() for _ in range(len(_VOCAB))]

BENCH = {
    "build_vocab": (_DOCS,),
    "bag_of_words": (_DOCS, _VOCAB),
    "term_frequency": (_BOW[0], sum(_BOW[0])),
    "document_frequency": (_BOW,),
    "inverse_document_frequency": (_DF, len(_DOCS)),
    "tfidf": (_BOW,),
    "l2_normalize": (_BOW,),
    "cosine_similarity": (_VEC_A, _VEC_B),
}
