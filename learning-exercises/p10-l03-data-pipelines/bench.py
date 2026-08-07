"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["model", "data", "token", "text", "learn", "corpus", "web", "clean"]


def _doc(n):
    return " ".join(random.choice(_WORDS) for _ in range(n))


_DIRTY = "<p>" + _doc(400) + " http://example.com/page </p>\n\n\n\n" + _doc(400)
_DOC = _doc(600)

_SET_A = {f"shingle {i}" for i in range(300)}
_SET_B = {f"shingle {i}" for i in range(150, 450)}

_SIG_A = [random.randrange(2 ** 32) for _ in range(64)]
_SIG_B = list(_SIG_A[:32]) + [random.randrange(2 ** 32) for _ in range(32)]

# 24 документа, каждый третий — почти-дубликат предыдущего
_DOCS = []
for i in range(24):
    if i % 3 == 1 and _DOCS:
        _DOCS.append(_DOCS[-1] + " tail")
    else:
        _DOCS.append(_doc(60))

_TOKENS = [random.randrange(256) for _ in range(20000)]

BENCH = {
    "clean_text": (_DIRTY,),
    "quality_filter": (_DOC,),
    "shingles": (_DOC, 5),
    "jaccard": (_SET_A, _SET_B),
    "minhash_signature": (_SET_A, 16, 0),
    "estimate_jaccard": (_SIG_A, _SIG_B),
    "deduplicate": (_DOCS, 0.8, 5, 16, 4, 0),
    "pack_sequences": (_TOKENS, 512),
}
