"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_FILES = [f"pkg/mod_{i}.py" for i in range(40)]
_ACTIONS = ["read", "write", "run"]

_trace = [
    {
        "action": random.choice(_ACTIONS),
        "target": random.choice(_FILES),
        "ok": random.random() > 0.2,
    }
    for _ in range(6000)
]
_trace.append({"action": "stop", "target": "done", "ok": True})

_allowed = _FILES[:20]
_acceptance = [f"pytest tests/test_{i}.py" for i in range(30)]
_scores = {"instructions": 2, "state": 0, "scope": 1, "feedback": 2,
           "verification": 1, "review": 0, "handoff": 2}

BENCH = {
    "missing_surfaces": (["scope", "state", "review"],),
    "weakest_surface": (_scores,),
    "repeated_steps": (_trace,),
    "off_scope_writes": (_trace, _allowed),
    "acceptance_status": (_trace, _acceptance),
    "classify_failures": (_trace, _allowed, _acceptance),
    "surfaces_to_fix": (["loop", "scope_creep", "premature_stop"],),
    "failure_report": (_trace, _allowed, _acceptance),
}
