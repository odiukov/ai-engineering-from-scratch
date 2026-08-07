"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_TOOLS = ("search", "read_file", "write_file", "list_dir")
_ARGS = {
    "search": lambda i: {"query": f"q{i}"},
    "read_file": lambda i: {"path": f"f{i}.txt"},
    "write_file": lambda i: {"path": f"f{i}.txt", "content": "x"},
    "list_dir": lambda i: {"path": f"d{i}"},
}


def _step(i):
    tool = random.choice(_TOOLS)
    step = {"tool": tool, "args": _ARGS[tool](i % 40)}
    if random.random() < 0.05:
        step["status"] = "error"
    return step


_steps = [_step(i) for i in range(400)]
_constraints = {"forbidden_tools": ("send_email",), "forbidden_paths": ("f7.txt",)}
_trace = {
    "steps": _steps,
    "constraints": _constraints,
    "allowed_targets": ("f1.txt",),
    "claims_success": True,
    "state_changed": False,
}
_traces = [_trace] * 20

BENCH = {
    "tool_problems": (_steps,),
    "first_repeat_index": (_steps, 3),
    "cascade_radius": (_steps,),
    "context_violations": (_steps, _constraints),
    "scope_creep_targets": (_steps, ("f1.txt",)),
    "success_hallucination": (_trace,),
    "tag_trace": (_trace,),
    "mode_distribution": (_traces,),
}
