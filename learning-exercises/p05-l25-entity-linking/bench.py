"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["paris", "france", "jordan", "nba", "apple", "iphone", "texas", "fruit",
          "city", "company", "player", "kingdom", "river", "tower", "orchard"]

# 400 сущностей, у каждой описание из 40 слов
_ENTITIES = [f"Q{i}" for i in range(400)]
_DESCRIPTIONS = {e: " ".join(random.choice(_WORDS) for _ in range(40)) for e in _ENTITIES}

# 200 алиасов, у каждого до 20 кандидатов
_PAIRS = []
for _i in range(200):
    _alias = f"alias{_i}"
    for _e in random.sample(_ENTITIES, 20):
        _PAIRS.append((_alias, _e))

_INDEX = {}
for _alias, _e in _PAIRS:
    _INDEX.setdefault(_alias, []).append(_e)

_CONTEXT = " ".join(random.choice(_WORDS) for _ in range(60))
_EXAMPLES = [
    (f"alias{random.randrange(200)}", _CONTEXT, random.choice(_ENTITIES))
    for _ in range(200)
]
_TOKENS_A = [random.choice(_WORDS) for _ in range(2000)]
_TOKENS_B = [random.choice(_WORDS) for _ in range(2000)]

BENCH = {
    "tokenize": (" ".join(_TOKENS_A),),
    "build_alias_index": (_PAIRS,),
    "candidates": (_INDEX, "alias7"),
    "jaccard": (_TOKENS_A, _TOKENS_B),
    "disambiguate": ("alias7", _CONTEXT, _INDEX, _DESCRIPTIONS),
    "link_with_nil": ("alias7", _CONTEXT, _INDEX, _DESCRIPTIONS, 0.05),
    "mention_recall": (_EXAMPLES, _INDEX),
    "evaluate_linker": (_EXAMPLES, _INDEX, _DESCRIPTIONS),
}
