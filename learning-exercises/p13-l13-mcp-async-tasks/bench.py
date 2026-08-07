"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_STATES = ("working", "input_required", "completed", "failed", "cancelled")

_store = {}
for _i in range(2000):
    _state = random.choice(_STATES)
    _store[f"tsk_{_i}"] = {
        "id": f"tsk_{_i}",
        "state": _state,
        "ttl": random.choice((100, 900000)),
        "createdAt": 0,
        "updatedAt": 0,
        "progress": 0.0,
        "result": "report" if _state == "completed" else None,
        "error": None,
    }

_working = {
    "id": "tsk_bench",
    "state": "working",
    "ttl": 900000,
    "createdAt": 0,
    "updatedAt": 0,
    "progress": 0.0,
    "result": None,
    "error": None,
}

BENCH = {
    "choose_task_support": (12.0,),
    "new_task": ("tsk_bench", 900000, 1000),
    "is_terminal": ("working",),
    "is_expired": (_working, 1000),
    "advance": (_working, "input_required", 1000),
    "cancel_task": (_working, 1000),
    "tasks_result": (_store, "tsk_1000", 1000),
    "recover_after_crash": (_store, 1000),
}
