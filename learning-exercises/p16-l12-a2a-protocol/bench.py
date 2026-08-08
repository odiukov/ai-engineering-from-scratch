"""Входные данные для замера скорости."""

import json


_SKILLS = [
    {"id": f"skill-{i}", "name": f"Skill {i}",
     "description": "Benchmark skill", "tags": ["bench"]}
    for i in range(200)
]
_CARD = {
    "name": "bench-agent",
    "description": "Benchmark agent",
    "supportedInterfaces": [{
        "url": "http://localhost:8765",
        "protocolBinding": "HTTP+JSON",
        "protocolVersion": "1.0",
    }],
    "version": "0.1.0",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "skills": _SKILLS,
}
_ENCODED = json.dumps(_CARD, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
_PAYLOAD = {"code": "\n".join(f"line {i}" for i in range(400))}
_MESSAGE = {
    "messageId": "msg-bench",
    "role": "ROLE_USER",
    "parts": [{
        "data": {"skill": "skill-199", "payload": _PAYLOAD},
        "mediaType": "application/json",
    }],
}
_TASK = {
    "id": "task-bench",
    "contextId": "ctx-task-bench",
    "status": {"state": "TASK_STATE_SUBMITTED"},
    "artifacts": [],
    "history": [_MESSAGE],
}


def _worker(payload):
    return {
        "artifactId": "artifact-bench",
        "parts": [{"data": {"lines": payload["code"].count("\n") + 1},
                   "mediaType": "application/json"}],
    }


BENCH = {
    "make_agent_card": (
        "bench-agent", "Benchmark agent", "0.1.0", _SKILLS,
        "http://localhost:8765",
    ),
    "encode_card": (_CARD,),
    "decode_card": (_ENCODED,),
    "supports_skill": (_CARD, "skill-199"),
    "make_message": ("msg-bench", "skill-199", _PAYLOAD),
    "make_task": (_MESSAGE, lambda: "task-bench-2"),
    "make_artifact": ("artifact-bench", "application/json", {"issues": []}),
    "advance_task": (_TASK, "TASK_STATE_WORKING"),
    "run_task": (_CARD, _MESSAGE, _worker, lambda: "task-bench-3"),
}
