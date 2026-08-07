"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_NAMES = [f"agent-{i}" for i in range(20)]

_pool = [
    {
        "from": random.choice(_NAMES),
        "content": "x" * 40,
        "handoff": random.choice(_NAMES + [None]),
    }
    for _ in range(20000)
]

_agent_a = {"name": "a", "system_prompt": "P", "tools": ["t1", "t2"] * 200,
            "policy": lambda pool: {"content": "x"}}
_agent_b = {"name": "b", "system_prompt": "P", "tools": ["t2", "t1"] * 200,
            "policy": lambda pool: {"content": "y"}}

_team = {
    name: {"name": name, "system_prompt": "P", "tools": [],
           "policy": lambda pool: {"content": "x", "handoff": None}}
    for name in _NAMES
}

BENCH = {
    "make_agent": ("researcher", "Gather facts.", ["search"] * 500,
                   lambda pool: {"content": "x"}),
    "agents_are_interchangeable": (_agent_a, _agent_b),
    "post": ([], "researcher", "note", "writer"),
    "project": (_pool, "agent-7"),
    "round_robin_selector": (_pool, _NAMES),
    "run_static": (_team, [], _NAMES, 20),
}
