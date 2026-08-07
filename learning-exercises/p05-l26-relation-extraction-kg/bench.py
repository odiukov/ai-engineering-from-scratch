"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi"]
_ORGS = ["Apple", "Google", "Microsoft", "Amazon", "Netflix", "Tesla"]
_PLACES = ["Alabama", "Oregon", "Vermont", "Kansas", "Nevada"]

_SENTENCES = []
for _ in range(400):
    _kind = random.randrange(3)
    if _kind == 0:
        _SENTENCES.append(f"{random.choice(_NAMES)} founded {random.choice(_ORGS)}.")
    elif _kind == 1:
        _SENTENCES.append(f"{random.choice(_NAMES)} works at {random.choice(_ORGS)}.")
    else:
        _SENTENCES.append(f"{random.choice(_NAMES)} was born in {random.choice(_PLACES)}.")

_TEXT = " ".join(_SENTENCES)

_PATTERNS = [
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
    (r"(?P<s>[A-Z]\w+) works at (?P<o>[A-Z]\w+)", "works at"),
    (r"(?P<s>[A-Z]\w+) was born in (?P<o>[A-Z]\w+)", "was born in"),
]

_RELATION_MAP = {"founded": "P112", "works at": "P108", "was born in": "P19"}

_TRIPLES = []
for _ in range(2000):
    _TRIPLES.append(
        (random.choice(_NAMES), random.choice(["founded", "works at", "was born in"]),
         random.choice(_ORGS + _PLACES))
    )

_EXTRACTIONS = []
for _i in range(1000):
    _start = random.randrange(max(len(_TEXT) - 20, 1))
    _EXTRACTIONS.append({
        "subject": _TEXT[_start:_start + 5],
        "subject_span": (_start, _start + 5),
        "relation": "works at",
        "object": random.choice(_ORGS),
        "object_span": (_start + 6, _start + 11),
    })

_GRAPH = {}
for _s, _r, _o in _TRIPLES:
    _GRAPH.setdefault(_s, []).append((_r, _o))

BENCH = {
    "extract_triples": (_TEXT, _PATTERNS),
    "verify_span": (_TEXT, _TEXT[10:20], (10, 20)),
    "filter_verified": (_TEXT, _EXTRACTIONS),
    "hallucination_rate": (_TEXT, _EXTRACTIONS),
    "canonicalize": ("Works At", _RELATION_MAP),
    "canonicalize_triples": (_TRIPLES, _RELATION_MAP),
    "build_graph": (_TRIPLES,),
    "neighbors": (_GRAPH, "Alice", "P108"),
}
