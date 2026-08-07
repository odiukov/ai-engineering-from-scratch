"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим


def _echo(text="hi"):
    return text


def _rows(n=40):
    return [{"id": f"note-{i}", "title": f"note {i}"} for i in range(n)]


_registry = {
    "echo": {"tool": {"name": "echo", "description": "Echo", "inputSchema": {}}, "handler": _echo},
    "rows": {"tool": {"name": "rows", "description": "Rows", "inputSchema": {}}, "handler": _rows},
}

_server = {
    "name": "bench",
    "version": "1.0.0",
    "subscribe": False,
    "tools": _registry,
    "resources": {
        f"notes://note-{i}": {"name": f"note {i}", "mimeType": "text/plain", "text": "x" * 64}
        for i in range(50)
    },
    "prompts": {
        "review": {
            "description": "Review",
            "messages": [{"role": "user", "content": {"type": "text", "text": "go"}}],
        }
    },
}

_methods = ["ping", "tools/list", "resources/list", "initialize", "tools/delete"]
_lines = []
for _i in range(400):
    _method = random.choice(_methods)
    if random.random() < 0.2:
        _lines.append(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    else:
        _lines.append(json.dumps({"jsonrpc": "2.0", "id": _i, "method": _method}))

BENCH = {
    "annotations": (True, False, True, False),
    "needs_confirmation": ({"name": "x", "annotations": {"readOnlyHint": True}},),
    "initialize_result": (_server,),
    "tool_content": (_rows(200),),
    "call_tool": (_registry, "rows", {"n": 100}),
    "dispatch": (_server, {"jsonrpc": "2.0", "id": 1, "method": "resources/list"}),
    "serve_lines": (_server, _lines),
}
