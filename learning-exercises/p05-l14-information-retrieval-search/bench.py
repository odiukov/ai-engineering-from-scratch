"""Входные данные для замера скорости."""

import random
from collections import Counter

random.seed(0)

_VOCAB = [f"term{i}" for i in range(2000)]


def _document(length):
    return " ".join(random.choice(_VOCAB) for _ in range(length))


_corpus = [_document(random.randint(40, 120)) for _ in range(800)]

# индекс собран по контракту из docstring build_bm25_index, а не вызовом
# самой функции: замер не должен зависеть от того, написана она уже или нет
_docs = [d.split() for d in _corpus]
_df = Counter()
for _doc in _docs:
    for _term in set(_doc):
        _df[_term] += 1
_index = {
    "docs": _docs,
    "df": _df,
    "n_docs": len(_docs),
    "avg_dl": sum(len(d) for d in _docs) / len(_docs),
    "k1": 1.5,
    "b": 0.75,
}

_query = " ".join(random.sample(_VOCAB, 8))
_long_text = _document(400)

_dim = 64
_doc_vectors = [[random.uniform(-1, 1) for _ in range(_dim)] for _ in range(3000)]
_query_vector = [random.uniform(-1, 1) for _ in range(_dim)]

_rankings = [
    [(random.random(), random.randrange(3000)) for _ in range(100)] for _ in range(3)
]
_eval_rankings = [
    [(random.random(), doc) for doc in random.sample(range(3000), 50)]
    for _ in range(200)
]
_relevant = [random.randrange(3000) for _ in range(200)]

BENCH = {
    "tokenize": (_long_text,),
    "build_bm25_index": (_corpus,),
    "bm25_idf": (_index, "term7"),
    "bm25_score": (_index, _query, 0),
    "bm25_rank": (_index, _query, 10),
    "dense_rank": (_query_vector, _doc_vectors, 30),
    "reciprocal_rank_fusion": (_rankings, 60),
    "evaluate_rankings": (_eval_rankings, _relevant, 10),
}
