"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_wide_state = tuple(float(x) for x in (9, 8, 7, 6, 5, 4))
_wide = {"state": _wide_state, "trace": (), "visits": 0, "value_sum": 0.0,
         "children": []}
_child = {"state": (24.0, 1.0), "trace": ("good",), "visits": 3,
          "value_sum": 2.4, "children": []}
_root = {"state": (99.0,), "trace": (), "visits": 3, "value_sum": 2.4,
         "children": [_child]}

BENCH = {
    "make_node": (_wide_state, ()),
    "expand": (_wide,),
    "value": (_wide,),
    "beam_search": ({"state": (8.0, 3.0, 1.0, 1.0), "trace": (), "visits": 0,
                     "value_sum": 0.0, "children": []}, 24, 8, 3),
    "uct": (10, _child, 1.4),
    "select_path": (_root, 1.4),
    "backpropagate": ([_root, _child], 1.0),
    "mcts": ({"state": (8.0, 3.0, 1.0, 1.0), "trace": (), "visits": 0,
              "value_sum": 0.0, "children": []}, 60, random.Random(0)),
}
