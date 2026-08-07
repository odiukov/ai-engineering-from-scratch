"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

PROTOCOL = "2025-11-25"

_names = ["notes", "files", "github", "postgres", "slack", "jira"]
_common = ["search", "list", "read"]


def _session(name, tools):
    return {
        "name": name,
        "protocolVersion": PROTOCOL,
        "capabilities": {"tools": {"listChanged": True}, "resources": {"subscribe": True}},
        "serverInfo": {"name": name, "version": "1.0.0"},
        "tools": [{"name": t, "description": t, "inputSchema": {}} for t in tools],
        "pending": {},
        "stale": False,
        "dirty": [],
        "alive": True,
    }


# у каждого сервера свои инструменты плюс общие имена — коллизий много
_sessions = [
    _session(name, _common + [f"{name}_{i}" for i in range(20)])
    for name in _names
]

_namespace = {}
for _s in _sessions:
    for _t in _s["tools"]:
        _key = _t["name"] if _t["name"] not in _namespace else f"{_s['name']}/{_t['name']}"
        _namespace[_key] = {"server": _s["name"], "tool": _t}

_reader_session = _session("notes", _common)
_reader_session["pending"] = {i: "tools/call" for i in range(500)}

_incoming = []
for _i in range(500):
    _roll = random.random()
    if _roll < 0.6:
        _incoming.append({"jsonrpc": "2.0", "id": _i, "result": {"content": []}})
    elif _roll < 0.8:
        _incoming.append({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    else:
        _incoming.append({"jsonrpc": "2.0", "id": 10000 + _i,
                          "method": "sampling/createMessage", "params": {}})

BENCH = {
    "handshake_messages": (1, "host", "0.1", {"roots": {"listChanged": True}}),
    "new_session": ("notes", {"protocolVersion": PROTOCOL,
                              "capabilities": {"tools": {}},
                              "serverInfo": {"name": "notes", "version": "1.0"}}),
    "supports": (_sessions[0], "resources.subscribe"),
    "merge_tools": (_sessions, "prefix"),
    "route_call": (_namespace, 1, "github/search", {"q": "mcp"}),
    "drain": (_reader_session, _incoming),
    "apply_notification": (_session("notes", _common),
                           {"jsonrpc": "2.0", "method": "notifications/resources/updated",
                            "params": {"uri": "notes://note-1"}}),
}
