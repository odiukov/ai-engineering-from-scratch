"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

END = "__end__"

_N = 60  # длина цепочки узлов: столько же чекпоинтов на один прогон


def _step(state):
    return {"n": state.get("n", 0) + 1, "messages": ["tick"]}


def _gate(state):
    if not state.get("human_approval"):
        return {"__pause__": "awaiting human approval"}
    return {"messages": ["approved"]}


# прямая цепочка n0 -> n1 -> ... -> END
_graph = {
    "entry": "n0",
    "nodes": {f"n{i}": _step for i in range(_N)},
    "edges": {
        **{f"n{i}": [(f"n{i + 1}", None)] for i in range(_N - 1)},
        f"n{_N - 1}": [(END, None)],
    },
}

# та же цепочка, но замкнутая в кольцо: resume можно звать сколько угодно раз,
# работа на вызов остаётся одинаковой, а max_steps не даёт зависнуть
_loop_graph = {
    "entry": "gate",
    "nodes": {"gate": _gate, **{f"n{i}": _step for i in range(_N)}},
    "edges": {
        "gate": [("n0", None)],
        **{f"n{i}": [(f"n{i + 1}", None)] for i in range(_N - 1)},
        f"n{_N - 1}": [("n0", None)],
    },
}

_state = {"n": 0, "messages": ["start"], "input": "bench"}
_update = {"n": 1, "messages": ["tick"], "route": "bug"}

# граф пошире, чтобы валидатору и выбору ребра было что перебирать
_wide = {
    "entry": "root",
    "nodes": {"root": _step, **{f"leaf{i}": _step for i in range(200)}},
    "edges": {
        "root": [(f"leaf{i}", (lambda s, i=i: s.get("n") == i)) for i in range(200)]
    },
}

_store = {"s1": [(f"n{i}", dict(_state)) for i in range(_N)]}
_resume_store = {"s1": [("gate", {"n": 0, "messages": ["start"]})]}

_big_state = {f"k{i}": i for i in range(500)}
_saved = {f"k{i}": i for i in range(0, 500, 2)}

BENCH = {
    "merge_update": (_state, _update),
    "next_node": (_wide, "root", {"n": 199}),
    "validate_graph": (_wide,),
    "save_checkpoint": ({}, "s1", "n0", _state),
    "load_checkpoint": (_store, "s1", 0),
    "run_graph": (_graph, dict(_state), {}, "s1"),
    "resume": (_loop_graph, _resume_store, "s1", {"human_approval": True}, _N),
    "missing_from_checkpoint": (_big_state, _saved),
}
