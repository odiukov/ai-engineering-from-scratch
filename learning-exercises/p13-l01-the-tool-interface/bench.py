"""Входные данные для замера скорости."""

import random

random.seed(0)

_ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}

_CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city"],
}

# Реестр на 40 инструментов: столько же порядка, сколько у среднего
# MCP-хоста с парой подключённых серверов.
_REGISTRY = [
    {
        "name": f"tool_{i:02d}",
        "description": "Use when the user asks for X. Do not use for Y.",
        "input_schema": _ADD_SCHEMA,
        "executor": (lambda args: args["a"] + args["b"]),
        "consequential": i % 7 == 0,
    }
    for i in range(40)
]
_REGISTRY.append(
    {
        "name": "add",
        "description": "Use when the user asks for a sum. Do not use for products.",
        "input_schema": _ADD_SCHEMA,
        "executor": (lambda args: args["a"] + args["b"]),
        "consequential": False,
    }
)

_ARGS = {"city": random.choice(["Tokyo", "Zurich", "Lagos"]), "units": "celsius"}
_CALL = {"id": "call_bench", "name": "add", "arguments": {"a": 2, "b": 3}}

_ALWAYS_CALLS = lambda messages: {"tool_calls": [dict(_CALL)]}

BENCH = {
    "make_tool": ("add", "Use when ...", _ADD_SCHEMA, len, False),
    "describe_registry": (_REGISTRY,),
    "validate_arguments": (_CITY_SCHEMA, _ARGS),
    "make_tool_call": ("call_bench", "add", {"a": 2, "b": 3}),
    "execute_call": (_REGISTRY, _CALL),
    "needs_confirmation": (_REGISTRY, _CALL),
    "run_loop": (_REGISTRY, "bench", _ALWAYS_CALLS, 5),
}
