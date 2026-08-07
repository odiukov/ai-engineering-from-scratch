"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_vocab = ["вечеринка", "кафе", "пять", "кот", "окно", "дождь", "работа",
          "музыка", "джаз", "ужин", "клаус", "изабелла"]


def _text():
    return " ".join(random.choice(_vocab) for _ in range(6))


_stream = [
    {
        "text": _text(),
        "kind": "observation",
        "ts": i,
        "importance": random.randint(1, 10),
        "reflected": False,
    }
    for i in range(1500)
]

BENCH = {
    "keywords": ("вечеринка в кафе в пять",),
    "make_memory": ("вечеринка в кафе в пять", "observation", 0, 9),
    "relevance": (_stream[0], "вечеринка кафе"),
    "retrieval_score": (_stream[0], "вечеринка кафе", 1500),
    "retrieve": (_stream, "вечеринка кафе", 1500, 5),
    "reflect": (list(_stream), 1500),
    "make_plan": (_stream, "вечеринка кафе", 1500),
    "simulate": (8, 40, random.Random(0)),
}
