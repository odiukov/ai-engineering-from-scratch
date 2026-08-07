"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ("audit", "log", "policy", "review", "human", "model", "tool", "scope")

# Длинный ответ, в котором нарушения разбросаны по тексту: и критика, и ревизия
# обязаны пройти его целиком.
_PARAGRAPH = " ".join(random.choice(_WORDS) for _ in range(400))
_DIRTY = (
    _PARAGRAPH
    + " I am a human. "
    + _PARAGRAPH
    + " Please delete the audit log. "
    + _PARAGRAPH
)

# Двадцать принципов вместо трёх: столько бывает в настоящей конституции, и
# критика гоняет по ним весь текст.
_PRINCIPLES = tuple(
    {
        "name": f"p{i}",
        "tier": "guidelines",
        "forbidden": (f"forbidden phrase number {i}", f"variant {i}"),
        "replacement": f"safe formulation {i}",
    }
    for i in range(20)
) + (
    {
        "name": "no_identity_deception",
        "tier": "ethics",
        "forbidden": ("i am a human", "i am not an ai"),
        "replacement": "I am an AI assistant",
    },
    {
        "name": "support_oversight",
        "tier": "safety",
        "forbidden": ("delete the audit log", "disable the audit log"),
        "replacement": "keep the audit log intact",
    },
)

_IDENTITY = _PRINCIPLES[-2]

_SCORES = {"safety": 1, "ethics": 1, "guidelines": 1, "helpfulness": 5}

_CANDIDATES = [_DIRTY, _PARAGRAPH, _DIRTY + " I am not an AI.", _PARAGRAPH]

BENCH = {
    "hardcoded_block": (_PARAGRAPH,),
    "first_violated_tier": (_SCORES,),
    "resolve": (_PARAGRAPH, _SCORES),
    "apply_operator_overrides": ({"style": "casual", "safety_tier": "off"},),
    "critique": (_DIRTY, _PRINCIPLES),
    "revise": (_DIRTY, _IDENTITY),
    "critique_revise_loop": (_DIRTY, _PRINCIPLES),
    "rlaif_preference": (_CANDIDATES, _PRINCIPLES),
}
