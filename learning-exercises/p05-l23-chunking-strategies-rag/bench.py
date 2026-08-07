"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["contract", "party", "payment", "fee", "notice", "termination",
          "clause", "secret", "monthly", "arbitration", "dispute", "acme"]


def _sentence():
    return " ".join(random.choice(_WORDS) for _ in range(12)).capitalize() + "."


# документ примерно на 40 КБ: наивные реализации с конкатенацией строк
# в цикле на нём заметно проседают, аккуратные — нет
_PARAGRAPHS = [" ".join(_sentence() for _ in range(8)) for _ in range(60)]
_DOC = "\n\n".join(_PARAGRAPHS)

_MAPPING = [
    {"child": _sentence(), "parent_idx": i // 5, "parent": _PARAGRAPHS[i // 5]}
    for i in range(300)
]

BENCH = {
    "chunk_fixed": (_DOC, 512),
    "split_sentences": (_DOC,),
    "chunk_recursive": (_DOC, 512),
    "chunk_sentence_window": (_DOC, 3, 1),
    "sentence_similarity": (_PARAGRAPHS[0], _PARAGRAPHS[1]),
    "chunk_semantic": (_DOC, 0.3, 200, 800),
    "chunk_parent_child": (_DOC, 2048, 256),
    "retrieve_parents": (_sentence(), _MAPPING, 5),
}
