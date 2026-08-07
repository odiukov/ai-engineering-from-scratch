"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ("code", "review", "bug", "deadline", "scope", "ship", "test", "spec")
_SPECIALTIES = {
    f"agent-{i}": random.sample(_WORDS, 3) for i in range(30)
}
_NAMES = list(_SPECIALTIES)
_POOL = [
    (random.choice(_NAMES), " ".join(random.choices(_WORDS, k=12))) for _ in range(400)
]
_POLICIES = {name: (lambda pool: "keep working on the code") for name in _SPECIALTIES}


def _selector(pool):
    return _NAMES[len(pool) % len(_NAMES)]


BENCH = {
    "keyword_score": (" ".join(random.choices(_WORDS, k=200)), list(_WORDS)),
    "round_robin_selector": (_POOL, _NAMES),
    "relevance_selector": (_POOL, _SPECIALTIES),
    "auto_selector": (_POOL, _SPECIALTIES),
    "is_terminated": (_POOL, "TERMINATE", None),
    "run_groupchat": (_POLICIES, _selector, 300),
    "speaker_counts": (_POOL,),
    "dominance": (_POOL,),
}
