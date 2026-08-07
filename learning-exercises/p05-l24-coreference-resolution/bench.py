"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_NAMES = ["Mary", "John", "Tim Cook", "Alice", "Steve", "Sarah"]
_PRONOUNS = ["She", "He", "They", "It"]
_NOMINALS = ["the company", "the doctor", "the engineers", "the device"]
_VERBS = ["called", "met", "left", "waved", "answered", "smiled"]


def _sentence():
    subject = random.choice(_NAMES + _PRONOUNS + _NOMINALS)
    return f"{subject} {random.choice(_VERBS)} {random.choice(_NAMES)}."


# документ на несколько тысяч mention-ов: resolve_pronouns квадратичен по
# числу mention-ов, и на таком тексте разница между аккуратным и наивным
# перебором кандидатов уже видна
_DOC = " ".join(_sentence() for _ in range(400))
_MENTIONS = [
    {"text": "x", "start": i * 10, "end": i * 10 + 3,
     "type": "pronoun" if i % 3 == 0 else "ne",
     "gender": random.choice(["m", "f", "u"]), "number": "sg"}
    for i in range(600)
]
_LINKS = [(i, i - 1) for i in range(1, 600, 2)]

_PRED = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(300)]
_GOLD = [[2 * i, 2 * i + 1] for i in range(450)]

BENCH = {
    "extract_mentions": (_DOC,),
    "agreement_score": (_MENTIONS[0], _MENTIONS[1]),
    "recency_score": (_MENTIONS[500], _MENTIONS[0]),
    "resolve_pronouns": (_MENTIONS,),
    "build_clusters": (600, _LINKS),
    "resolve_document": (_DOC,),
    "muc_f1": (_PRED, _GOLD),
}
