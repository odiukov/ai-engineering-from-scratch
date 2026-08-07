"""Входные данные для замера скорости."""

_END = "__end__"


def _inc(state):
    return {"n": state["n"] + 1}


def _append(old, new):
    return list(old or []) + list(new)


def _router(state):
    return "again" if state["n"] < 200 else "done"


_nodes = {f"n{i}": _inc for i in range(50)}
_edges = {f"n{i}": (f"n{i + 1}" if i < 49 else _END) for i in range(50)}

_loop_nodes = {"a": _inc}
_loop_edges = {"a": (_router, {"again": "a", "done": _END})}
_loop_graph = {"nodes": _loop_nodes, "edges": _loop_edges, "entry": "a", "reducers": {}}

_messages = [{"role": "user", "content": f"m{i}"} for i in range(500)]
_state = {"n": 0, "messages": _messages, "plan": ["a", "b"], "budget": 10}

def _model(messages):
    return {"role": "assistant", "content": "done"}


def _city(city):
    return city


_tools = {"temp": _city}

_run = {
    "checkpoints": [
        {"id": i, "node": "a", "next": "a", "state": {"n": i}} for i in range(200)
    ]
}

BENCH = {
    "add_messages": (_messages, _messages),
    "merge_state": (_state, {"n": 1, "messages": _messages}, {"messages": _append}),
    "compile_graph": (_nodes, _edges, "n0"),
    "route": (_loop_graph, "a", {"n": 5}),
    "run_graph": (_loop_graph, {"n": 0}, 500),
    "resume": (_loop_graph, _run["checkpoints"], 100, None, 500),
    "build_react_graph": (_model, _tools),
}
