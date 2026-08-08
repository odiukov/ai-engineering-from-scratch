"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_TAGS = ["research", "coding", "review", "docs", "search"]
_MODES = ["text/plain", "application/json", "application/pdf"]

_cards = [
    {
        "name": f"agent-{i}",
        "description": f"agent-{i} agent",
        "supportedInterfaces": [{"url": f"https://a{i}.local/a2a/v1",
                                  "protocolBinding": "HTTP+JSON",
                                  "protocolVersion": "1.0"}],
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [random.choice(_MODES)],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": f"skill-{i}-{j}",
                "name": f"Skill {i}-{j}",
                "description": "Benchmark skill",
                "tags": random.sample(_TAGS, 2),
                "inputModes": [random.choice(_MODES)],
                "outputModes": ["text/plain"],
            }
            for j in range(4)
        ],
    }
    for i in range(2000)
]

_skills = _cards[0]["skills"]
_task = {"id": "t", "context_id": "c", "state": "working",
         "artifacts": [{"id": f"a{i}", "name": "r", "parts": ["x"] * 5}
                       for i in range(500)]}
_event = {"kind": "artifactUpdate",
          "artifact": {"id": "a499", "name": "r", "parts": ["y"]},
          "append": True}

_secrets = {f"did:wba:a{i}": f"key-{i}" for i in range(2000)}
_message = {"id": "msg-001", "role": "user", "parts": [{"kind": "text", "text": "x"}]}

BENCH = {
    "agent_card": ("researcher", "https://r.local", _skills,
                   ["text/plain"], ["application/json"]),
    "discover": (_cards, "research", "text/plain"),
    "new_task": ("t-1", "ctx-1"),
    "apply_event": (_task, _event),
    "sign": ("coder-key", _message),
    "verify": (_secrets, "did:wba:a1999", _message, "deadbeef"),
    "audit_run": ("r-1", "researcher", _message, lambda m: ("ok", [])),
}
