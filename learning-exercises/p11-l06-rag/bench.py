"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_vocab_words = [f"word{i}" for i in range(400)]
_documents = [
    " ".join(random.choice(_vocab_words) for _ in range(120)) for _ in range(120)
]
_long_text = " ".join(random.choice(_vocab_words) for _ in range(20000))

_vocab = sorted(set(_vocab_words))
_idf = [1.0 + i / len(_vocab) for i in range(len(_vocab))]
_embeddings = [
    [random.random() if random.random() < 0.05 else 0.0 for _ in _vocab]
    for _ in range(300)
]
_query_vec = _embeddings[0]

_retrieved = list(range(500))
random.shuffle(_retrieved)

BENCH = {
    "chunk_text": (_long_text, 200, 50),
    "build_vocabulary": (_documents,),
    "compute_idf": (_documents, _vocab),
    "tfidf_embed": (_documents[0], _vocab, _idf),
    "cosine_similarity": (_embeddings[0], _embeddings[1]),
    "search": (_query_vec, _embeddings, 5),
    "build_rag_prompt": ("what is the refund policy", _documents[:5]),
    "recall_at_k": (_retrieved, list(range(0, 500, 7)), 50),
}
