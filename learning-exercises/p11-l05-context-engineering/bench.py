"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_vocab = ["jwt", "token", "postgres", "vector", "index", "auth", "cache", "queue"]
_doc = lambda n: " ".join(random.choice(_vocab) for _ in range(n))

_text = _doc(6000)
_docs = [_doc(60) for _ in range(300)]
_scores = [random.random() for _ in _docs]
_turns = [("user" if i % 2 == 0 else "assistant", _doc(40)) for i in range(200)]
_components = [(f"c{i}", _doc(300), 400) for i in range(20)]

BENCH = {
    "count_tokens": (_text,),
    "truncate_to_tokens": (_text, 500),
    "score_relevance": ("jwt token auth", _docs),
    "reorder_lost_in_middle": (_docs, _scores),
    "allocate_budget": (_components, 4000, 500),
    "compress_history": (_turns, 300),
    "classify_intent": (_text,),
    "select_tools": ("fix the bug in the database query code", 600),
}
