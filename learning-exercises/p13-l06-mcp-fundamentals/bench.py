"""Входные данные для замера скорости."""

import random

random.seed(0)

_CLIENT_CAPS = {"roots": {"listChanged": True}, "sampling": {}, "elicitation": {}}
_SERVER_CAPS = {
    "tools": {"listChanged": True},
    "resources": {"subscribe": True, "listChanged": True},
    "prompts": {"listChanged": True},
}

_METHODS = [
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/read",
    "resources/subscribe",
    "prompts/list",
    "prompts/get",
    "sampling/createMessage",
]

# Сессия на ~4000 сообщений: столько набегает за час работы агента с
# подключённым MCP-сервером. На ней видна разница между сопоставлением по
# словарю и наивным поиском ответа обходом всего транскрипта.
_TRANSCRIPT = [
    {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"capabilities": _CLIENT_CAPS}},
    {"jsonrpc": "2.0", "id": 0, "result": {"capabilities": _SERVER_CAPS}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
]
for _i in range(1, 2000):
    _TRANSCRIPT.append({"jsonrpc": "2.0", "id": _i, "method": random.choice(_METHODS)})
    _TRANSCRIPT.append({"jsonrpc": "2.0", "id": _i, "result": {"ok": True}})
random.shuffle(_TRANSCRIPT)

BENCH = {
    "classify_message": (_TRANSCRIPT[0],),
    "primitive_of": ("notifications/resources/updated",),
    "owner_of": ("sampling",),
    "negotiated_features": (_CLIENT_CAPS, _SERVER_CAPS),
    "is_permitted": ("resources/subscribe", _CLIENT_CAPS, _SERVER_CAPS),
    "pair_messages": (_TRANSCRIPT,),
    "trace": (_TRANSCRIPT,),
    "transcript_stats": (_TRANSCRIPT,),
}
