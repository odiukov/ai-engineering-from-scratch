"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_NAMES = [f"agent-{i}" for i in range(50)]
_AGENTS = {
    name: {
        "name": name,
        "instructions": "routine",
        "handoffs": tuple(n for n in _NAMES if n != name),
    }
    for name in _NAMES
}
# Правила ведут по цепочке вперёд и обрываются на последнем агенте.
_RULES = {
    _NAMES[i]: [("go", _NAMES[i + 1])] for i in range(len(_NAMES) - 1)
}
_HISTORY = [("user", f"message {i}") for i in range(800)]
_TRACE = [random.choice(_NAMES) for _ in range(2000)]

BENCH = {
    "make_agent": ("triage", "Route the user.", tuple(_NAMES[1:])),
    "can_handoff": (_AGENTS["agent-0"], "agent-49"),
    "resolve_target": (_AGENTS, "ghost", "agent-0"),
    "context_transfer": (_HISTORY, "last_n", 100),
    "is_ping_pong": (_TRACE, 4),
    "route": (_AGENTS["agent-0"], "go ahead", _RULES),
    "run_conversation": (_AGENTS, _RULES, "agent-0", ["go"] * 20, 60),
    "handoff_stats": ({"trace": _TRACE},),
}
