"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_rng = random.Random(0)

_messages = [
    {"jsonrpc": "2.0", "id": i, "method": "tools/call",
     "params": {"name": "notes_search", "arguments": {"query": "mcp" * 5}}}
    for i in range(300)
]
_stream = "".join(json.dumps(m) + "\n" for m in _messages)

_allowlist = ["http://localhost", "https://claude.ai"] + [
    f"https://*.tenant{i}.example.com" for i in range(30)
]

_events = [{"id": str(i), "data": "x" * 40} for i in range(500)]
_sse_text = "".join(f"id: {e['id']}\ndata: {e['data']}\n\n" for e in _events)


def _handler(message):
    if message is None or "id" not in message:
        return None
    return {"jsonrpc": "2.0", "id": message["id"], "result": {"ok": True}}


_state = {
    "endpoint": "/mcp",
    "allowlist": _allowlist,
    "sessions": {"ab" * 16: {"events": []}},
    "handler": _handler,
}

BENCH = {
    "split_stdio": ("", _stream),
    "new_session_id": (random.Random(1),),
    "origin_allowed": ("https://app.tenant29.example.com", _allowlist),
    "sse_event": ("line\n" * 20, 42, "message"),
    "parse_sse": (_sse_text,),
    "replay_after": (_events, "480"),
    "detect_transport": ({"status": 200, "headers": {"Content-Type": "application/json"}},),
    "handle_http": (_state, "POST", "/mcp",
                    {"Origin": "http://localhost", "Mcp-Session-Id": "ab" * 16},
                    {"jsonrpc": "2.0", "id": 1, "method": "ping"}, _rng),
}
