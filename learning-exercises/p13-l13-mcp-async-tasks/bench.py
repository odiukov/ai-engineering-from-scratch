"""Входные данные для замера скорости."""

from datetime import datetime, timedelta, timezone
import random

random.seed(0)

_origin = datetime(2025, 11, 25, 10, 30, tzinfo=timezone.utc)


def _iso(offset_ms):
    return (_origin + timedelta(milliseconds=offset_ms)).isoformat().replace("+00:00", "Z")


_states = ("working", "input_required", "completed", "failed", "cancelled")
_store = {}
for _i in range(2000):
    _state = random.choice(_states)
    _store[f"tsk_{_i}"] = {
        "task": {
            "taskId": f"tsk_{_i}",
            "status": _state,
            "createdAt": _iso(0),
            "lastUpdatedAt": _iso(100),
            "ttl": random.choice((100, 900000)),
            "pollInterval": 1000,
        },
        "result": {"content": [], "isError": False} if _state == "completed" else None,
    }

_working = {
    "task": {
        "taskId": "tsk_bench",
        "status": "working",
        "createdAt": _iso(0),
        "lastUpdatedAt": _iso(0),
        "ttl": 900000,
        "pollInterval": 1000,
    },
    "result": None,
}

BENCH = {
    "choose_task_support": (12.0,),
    "new_task": ("tsk_bench", 900000, _iso(0)),
    "create_task_result": (_working,),
    "is_terminal": ("working",),
    "is_expired": (_working, _iso(1000)),
    "advance": (_working, "input_required", _iso(1000)),
    "cancel_task": (_working, _iso(1000)),
    "tasks_get": (_store, "tsk_1000", _iso(1000)),
    "recover_after_crash": (_store, _iso(1000)),
}
