"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_SKILLS = [f"skill-{i}" for i in range(200)]
_CARD = {
    "name": "bench-agent",
    "version": "0.1.0",
    "skills": tuple(_SKILLS),
    "endpoints": {
        "card": "http://localhost:8765/.well-known/agent.json",
        "tasks": "http://localhost:8765/tasks",
    },
    "auth": {"type": "bearer"},
    "modalities": ("text", "structured"),
    "protocol_version": "a2a-0.3",
}
_ENCODED = json.dumps(_CARD, sort_keys=True, ensure_ascii=False)
_PAYLOAD = {"code": "\n".join(f"line {i}" for i in range(400))}
_TASK = {
    "id": "t-bench",
    "skill": "skill-199",
    "payload": _PAYLOAD,
    "state": "submitted",
    "artifact": None,
}


def _worker(payload):
    return {"type": "structured", "data": {"lines": payload["code"].count("\n") + 1}}


BENCH = {
    "make_agent_card": ("bench-agent", "0.1.0", _SKILLS, "http://localhost:8765"),
    "encode_card": (_CARD,),
    "decode_card": (_ENCODED,),
    "supports_skill": (_CARD, "skill-199"),
    "make_task": ("t-bench", "skill-199", _PAYLOAD),
    "make_artifact": ("structured", {"issues": []}),
    "advance_task": (_TASK, "working"),
    "run_task": (_CARD, _TASK, _worker),
}
