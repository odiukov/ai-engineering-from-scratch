"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_BIG = (1920, 1080)
_SMALL = (800, 600)

_elements = [
    {"eid": f"el{i:04d}", "label": "search_button" if i % 2 else "query_field",
     "x": (i * 37) % 1800, "y": (i * 53) % 1000, "w": 40, "h": 24,
     "sensitive": i % 17 == 0}
    for i in range(2000)
]

_screen = {
    "elements": _elements,
    "dom_text": "Product page. " * 400,
    "allowed_labels": ("search_button", "query_field"),
}

_actions = []
for _ in range(2000):
    if random.random() < 0.7:
        _actions.append({"kind": "click",
                         "x": random.randrange(1920), "y": random.randrange(1080)})
    else:
        _actions.append({"kind": "type", "text": "wireless headphones"})

_long_text = "Search results page with plenty of product descriptions. " * 500


def _always_yes(reason):
    return True


BENCH = {
    "normalize_point": ((960, 540), _BIG),
    "denormalize_point": ((0.5, 0.5), _BIG),
    "rescale_point": ((960, 540), _BIG, _SMALL),
    "scale_elements": (_elements, _BIG, _SMALL),
    "element_at": (_elements, (1799, 999)),
    "contains_injection": (_long_text,),
    "assess_action": ({"kind": "click", "x": 1799, "y": 999}, _screen),
    "run_agent": (_actions, _screen, _always_yes),
}
