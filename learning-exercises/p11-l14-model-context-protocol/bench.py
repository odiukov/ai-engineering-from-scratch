"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_props = {f"p{i}": {"type": "integer"} for i in range(20)}
_required = tuple(f"p{i}" for i in range(10))
_schema = {
    "name": "wide",
    "description": "Tool with twenty parameters",
    "inputSchema": {"type": "object", "properties": _props, "required": list(_required)},
}
_arguments = {f"p{i}": i for i in range(20)}

_server = {
    "name": "bench-server",
    "version": "1.0.0",
    "tools": {
        f"t{i}": {
            "schema": {
                "name": f"t{i}",
                "description": "bench tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                },
            },
            "handler": (lambda a: a * 2),
        }
        for i in range(40)
    },
}

_call = {"jsonrpc": "2.0", "method": "tools/call", "id": 1,
         "params": {"name": "t7", "arguments": {"a": 3}}}
_batch = [
    {"jsonrpc": "2.0", "method": "tools/call", "id": i,
     "params": {"name": f"t{i % 40}", "arguments": {"a": i}}}
    for i in range(200)
]

BENCH = {
    "make_request": ("tools/call", {"name": "t1", "arguments": {"a": 1}}, 1),
    "make_response": (1, {"ok": True}),
    "make_error": (1, -32602, "Invalid params"),
    "tool_schema": ("wide", "Tool with twenty parameters", _props, _required),
    "validate_arguments": (_schema, _arguments),
    "call_tool": (_server, 1, {"name": "t7", "arguments": {"a": 3}}),
    "handle": (_server, _call),
    "handle_batch": (_server, _batch),
}
