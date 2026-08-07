"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_boxes = [f"box{i}" for i in range(60)]
_names = [f"a{i}" for i in range(30)]

_agent = {"name": "me", "order": 1, "beliefs": {}, "models": {}}
for _b in _boxes:
    _agent["beliefs"][_b] = random.random()
for _n in _names:
    _agent["models"][_n] = {
        "beliefs": {b: random.random() for b in _boxes},
        "models": {},
    }

BENCH = {
    "new_agent": ("observer", 2),
    "update_belief": (_agent, ("a0",), "marble", "basket_A"),
    "belief_of": (_agent, ("a0",), "box0"),
    "observe": (_agent, "marble", "basket_A", _names),
    "predict_search": (_agent, "a0", "box0"),
    "sally_anne": (1,),
    "choose_box": (_agent, _boxes, _names),
    "simulate_collection": (4, 4, 1, random.Random(0), 200),
}
