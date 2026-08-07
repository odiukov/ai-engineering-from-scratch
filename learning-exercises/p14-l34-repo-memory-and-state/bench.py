"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "active_task_id", "touched_files", "risks", "next_action"],
    "properties": {
        "schema_version": {"type": "integer", "enum": [2]},
        "active_task_id": {"type": ["string", "null"], "pattern": r"^T-\d{3,}$"},
        "touched_files": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
    },
}

_state = {
    "schema_version": 2,
    "active_task_id": "T-042",
    "touched_files": [f"pkg/mod_{i}.py" for i in range(2000)],
    "risks": [f"риск {i}" for i in range(200)],
    "next_action": "продолжить правку",
}

_fs = {"agent_state.json": json.dumps(_state)}

_memory = [
    {
        "key": f"fact-{i}",
        "value": f"значение-{i}",
        "first_seen": random.randint(0, 500),
        "last_seen": random.randint(500, 1000),
    }
    for i in range(4000)
]

_old_state = {
    "schema_version": 1,
    "active_task_id": "T-042",
    "touched_files": _state["touched_files"],
    "blockers": _state["risks"],
    "next_action": "продолжить правку",
}

BENCH = {
    "validate": (_state, _SCHEMA),
    "atomic_write": (_fs, "agent_state.json", json.dumps(_state)),
    "remember": (_memory, "fact-3999", "новое", 2000),
    "forget_stale": (_memory, 1200, 400),
    "commit_state": (_fs, "agent_state.json", _state, _SCHEMA),
    "load_state": (_fs, "agent_state.json", _SCHEMA),
    "migrate_state": (_old_state,),
}
