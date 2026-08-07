"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "days": {"type": "integer"},
    },
    "required": ["city"],
}

_CITIES = ["tokyo", "london", "paris", "berlin", "madrid"]
_DB = {c: {"temp_c": 10 + i} for i, c in enumerate(_CITIES)}


def _weather(city, units="celsius", days=1):
    return dict(_DB.get(city.lower(), {"error": True}), city=city, days=days)


_REGISTRY = {
    "get_weather": {
        "definition": {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city.",
                "parameters": _SCHEMA,
            },
        },
        "function": _weather,
    }
}

_CALLS = [
    {"name": "get_weather", "arguments": {"city": random.choice(_CITIES), "days": 3}}
    for _ in range(200)
]
_RAW = json.dumps([{"name": c["name"], "arguments": c["arguments"]} for c in _CALLS])
_CASES = [(f"weather in {random.choice(_CITIES)}", "get_weather") for _ in range(200)]


def _decide(message, conversation):
    if any(m["role"] == "tool" for m in conversation):
        return []
    return [{"name": "get_weather", "arguments": {"city": "tokyo"}}]


BENCH = {
    "register_tool": ({}, "get_weather", "Get weather.", _SCHEMA, _weather),
    "validate_arguments": (_SCHEMA, {"city": "tokyo", "units": "celsius", "days": 3}),
    "parse_tool_calls": (_RAW,),
    "execute_tool_call": (_REGISTRY, _CALLS[0]),
    "run_tool_calls": (_REGISTRY, _CALLS, 1000),
    "agent_loop": (_REGISTRY, "weather in tokyo", _decide, 5),
    "tool_selection_accuracy": (_decide, _CASES),
}
