"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_catalog = [
    {"name": f"model-{i}", "cost": random.random(),
     "speed": random.random(), "intelligence": random.random()}
    for i in range(200)
]

_prefs = {
    "costPriority": 0.3,
    "speedPriority": 0.2,
    "intelligencePriority": 0.5,
    "hints": [{"name": "model-42"}],
}

_messages = [f"round {i}: " + "x" * 200 for i in range(30)]

_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "sampling/createMessage",
    "params": {
        "messages": [{"role": "user", "content": {"type": "text", "text": "summarize"}}],
        "includeContext": "none",
        "maxTokens": 1024,
        "tools": [{"name": "fetch_url", "description": "d", "inputSchema": {}}],
    },
}

_plan = [[f"round {i}"] for i in range(50)]


def _client(request):
    return {
        "jsonrpc": "2.0",
        "id": request["id"],
        "result": {"role": "assistant",
                   "content": {"type": "text", "text": "ok"},
                   "model": "model-0",
                   "stopReason": "endTurn"},
    }


BENCH = {
    "model_preferences": (0.6, 0.2, 0.2, ["model-42", "model-7"]),
    "create_message_request": (1, _messages, "be terse", _prefs, 2048, "none"),
    "pick_model": (_catalog, _prefs),
    "sampling_result": (1, "x" * 500, "model-0", "endTurn"),
    "spend_sample": ({"loop": 1}, "loop", 1000),
    "run_sampling_loop": (_plan, _client, _prefs, 100),
    "review_request": (_request,),
}
