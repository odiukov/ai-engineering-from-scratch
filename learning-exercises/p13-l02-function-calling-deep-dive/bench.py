"""Входные данные для замера скорости."""

import json
import random

random.seed(0)


def _nested(levels):
    node = {"type": "string", "description": "leaf"}
    for i in range(levels - 1):
        node = {
            "type": "object",
            "properties": {f"f{i}": node, "flag": {"type": "boolean"}},
            "required": [f"f{i}"],
            "additionalProperties": False,
        }
    return node


_SCHEMA = _nested(6)

_TOOLS = [
    {
        "name": f"tool_{i:03d}",
        "description": "Use when the user asks for X. Do not use for Y.",
        "input_schema": _SCHEMA,
        "strict": True,
    }
    for i in range(120)
]

_TOOL = _TOOLS[0]

_OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {
                "tool_calls": [
                    {
                        "id": f"call_{i:03d}",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps(
                                {"city": random.choice(["Tokyo", "Zurich"]), "units": "celsius"}
                            ),
                        },
                    }
                    for i in range(30)
                ]
            }
        }
    ]
}

BENCH = {
    "schema_depth": (_SCHEMA,),
    "gemini_schema": (_SCHEMA,),
    "declare": ("gemini", _TOOL),
    "tool_choice_for": ("gemini", "force", "tool_000"),
    "parse_tool_calls": ("openai", _OPENAI_RESPONSE),
    "make_tool_result": ("anthropic", "toolu_1", "tool_000", "ok"),
    "check_limits": ("openai", _TOOLS),
}
