"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["iphone", "ipod", "android", "google", "apple", "launched", "released",
          "system", "operating", "search", "index", "vector", "passage", "query"]

_DIM = 256
_LONG_TEXT = " ".join(random.choice(_WORDS) for _ in range(3000))
_QUERY_TEXT = " ".join(random.choice(_WORDS) for _ in range(12))

_vec = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
_other = [random.uniform(-1.0, 1.0) for _ in range(_DIM)]
# корпус в 2000 документов: ранжирование обязано быть линейным по корпусу
_corpus = [[random.uniform(-1.0, 1.0) for _ in range(_DIM)] for _ in range(2000)]

# late interaction: 32 токена запроса против 512 токенов документа
_q_tokens = [[random.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)]
_d_tokens = [[random.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(512)]

_sparse_long = None  # заполняется ниже через тот же tf-подсчёт, что и в уроке


def _sparse(text):
    """Разреженный вектор без импорта решения — только для замера."""
    tf = {}
    for token in text.split():
        tf[token] = tf.get(token, 0) + 1
    return {t: float(c) for t, c in tf.items()}


_sparse_long = _sparse(_LONG_TEXT)
_sparse_query = _sparse(_QUERY_TEXT)

_rankings = [
    random.sample(range(5000), 200),
    random.sample(range(5000), 200),
    random.sample(range(5000), 200),
]

BENCH = {
    "embed": (_LONG_TEXT, _DIM),
    "cosine": (_vec, _other),
    "matryoshka_truncate": (_vec, 64),
    "rank": (_vec, _corpus),
    "sparse_embed": (_LONG_TEXT,),
    "sparse_score": (_sparse_query, _sparse_long),
    "maxsim": (_q_tokens, _d_tokens),
    "reciprocal_rank_fusion": (_rankings,),
}
