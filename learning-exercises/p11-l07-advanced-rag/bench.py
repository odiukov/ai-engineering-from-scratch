"""Входные данные для замера скорости.

Индекс собирается здесь локально, а не через импорт из exercise: на пустой
заготовке build_bm25_index ещё бросает NotImplementedError, и compare.py
падал бы на загрузке этого файла.
"""

import random
import re

random.seed(0)  # обязательно: замер должен быть воспроизводим

_VOCAB = [f"term{i}" for i in range(300)]
_DOCS = [
    " ".join(random.choice(_VOCAB) for _ in range(random.randint(40, 120)))
    for _ in range(400)
]


def _build_index(docs):
    tokens = [re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", d.lower()) for d in docs]
    doc_lens = [len(t) for t in tokens]
    freqs = {}
    for toks in tokens:
        for term in set(toks):
            freqs[term] = freqs.get(term, 0) + 1
    return {
        "docs": list(docs),
        "tokens": tokens,
        "doc_lens": doc_lens,
        "avg_dl": sum(doc_lens) / len(docs),
        "doc_freqs": freqs,
        "n_docs": len(docs),
    }


_INDEX = _build_index(_DOCS)
_QUERY = " ".join(random.choice(_VOCAB) for _ in range(8))
_VECTOR = [(i, 1.0 - i / 100) for i in range(30)]
_CANDIDATES = [(i, random.random()) for i in range(60)]
_LISTS = [[(random.randrange(400), random.random()) for _ in range(50)] for _ in range(3)]
_TEXT = " ".join(random.choice(_VOCAB) for _ in range(6000))

BENCH = {
    "tokenize": (" ".join(_DOCS[:20]),),
    "build_bm25_index": (_DOCS,),
    "bm25_score": (_QUERY, 0, _INDEX),
    "bm25_search": (_QUERY, _INDEX, 10),
    "reciprocal_rank_fusion": (_LISTS, 60),
    "hybrid_search": (_QUERY, _INDEX, _VECTOR, 5, 60),
    "rerank": (_QUERY, _CANDIDATES, _DOCS),
    "parent_child_chunks": (_TEXT, 200, 50),
}
