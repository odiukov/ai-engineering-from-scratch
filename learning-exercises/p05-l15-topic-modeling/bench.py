"""Входные данные для замера скорости."""

import random

random.seed(0)

_WORDS_A = ["cat", "dog", "pet", "vet", "leash", "kitten"]
_WORDS_B = ["stock", "bond", "market", "yield", "broker", "fund"]

_documents = []
for _i in range(120):
    _pool = _WORDS_A if _i % 2 == 0 else _WORDS_B
    _documents.append(" ".join(random.choice(_pool) for _ in range(40)))

_vocab = sorted(set(_WORDS_A) | set(_WORDS_B))
_index = {w: i for i, w in enumerate(_vocab)}
_doc_ids = [[_index[w] for w in doc.split()] for doc in _documents]
_assignments = [[random.randrange(4) for _ in doc] for doc in _doc_ids]
_topics = [random.sample(_vocab, 5) for _ in range(4)]
_doc_tokens = [doc.split() for doc in _documents]
_clusters = [[w for doc in _doc_ids[i::4] for w in doc] for i in range(4)]
_topic_word = [[random.random() for _ in _vocab] for _ in range(4)]

BENCH = {
    "build_corpus": (_documents,),
    "count_tables": (_doc_ids, _assignments, 4, len(_vocab)),
    "topic_conditional": ([3, 1, 0, 2], [2, 0, 5, 1], [7, 3, 9, 4], 0.1, 0.01, 12),
    "fit_lda": (_doc_ids, 4, len(_vocab), random.Random(0), 3),
    "top_words": (_topic_word, _vocab, 10),
    "topic_diversity": (_topics,),
    "topic_coherence_npmi": (_topics, _doc_tokens),
    "class_based_tfidf": (_clusters, len(_vocab)),
}
