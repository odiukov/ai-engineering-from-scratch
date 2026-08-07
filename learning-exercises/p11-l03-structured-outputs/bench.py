"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_schema = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number", "minimum": 0},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "price"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
}

_data = {
    "items": [
        {
            "name": f"item-{i}",
            "price": round(random.uniform(1, 1000), 2),
            "tags": [f"tag-{random.randrange(20)}" for _ in range(5)],
        }
        for i in range(500)
    ]
}

_raw = "```json\n" + json.dumps(_data) + "\n```"
_partial = json.dumps(_data)[:-1]

_fields = {f"field_{i}": {"type": (str, int, float, bool)[i % 4]} for i in range(200)}

BENCH = {
    "strip_code_fence": (_raw,),
    "parse_llm_json": (_raw,),
    "validate": (_data, _schema),
    "python_type_to_schema": (str,),
    "model_to_schema": (_fields,),
    "next_valid_tokens": (_partial,),
    "extract_with_retry": ("text", _schema, lambda *_: _raw),
}
