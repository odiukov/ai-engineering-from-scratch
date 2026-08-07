"""Входные данные для замера скорости."""

import random

random.seed(0)

_DESCRIPTION = (
    "Use when the user wants to see all notes or a filtered list by tag. "
    "Do not use for reading one note's body; use notes_get instead."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        f"field_{i:02d}": {"type": "string", "description": "a field"} for i in range(12)
    },
    "required": ["field_00"],
    "additionalProperties": False,
}

# Реестр на 300 инструментов: столько набирается у хоста с десятком
# подключённых MCP-серверов. Часть имён намеренно дублируется, чтобы
# наивная проверка дубликатов за O(n^2) была заметна на фоне словаря.
_REGISTRY = [
    {
        "name": f"notes_{i % 150:03d}",
        "description": _DESCRIPTION,
        "input_schema": _SCHEMA,
    }
    for i in range(300)
]

_FINDINGS = [
    {
        "severity": random.choice(["block", "warn", "nit"]),
        "path": f"tool_{i}",
        "rule": "field_missing_description",
        "message": "m",
    }
    for i in range(5000)
]

BENCH = {
    "is_snake_case": ("notes_list_v2",),
    "lint_name": ("get_weather_in_tokyo",),
    "lint_description": ("notes_list", _DESCRIPTION),
    "lint_schema": ("notes_list", _SCHEMA),
    "lint_tool": (_REGISTRY[0],),
    "lint_registry": (_REGISTRY,),
    "severity_summary": (_FINDINGS,),
    "passes_ci": (_FINDINGS,),
}
