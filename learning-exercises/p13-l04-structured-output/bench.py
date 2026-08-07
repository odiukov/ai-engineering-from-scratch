"""Входные данные для замера скорости."""

import json
import random

random.seed(0)

_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "sku": {"type": "string", "pattern": "^[A-Z0-9-]+$"},
        "qty": {"type": "integer", "minimum": 1},
        "unit_usd": {"type": "number", "minimum": 0},
    },
    "required": ["sku", "qty", "unit_usd"],
    "additionalProperties": False,
}

_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string", "minLength": 1, "maxLength": 200},
        "line_items": {"type": "array", "items": _ITEM_SCHEMA},
        "total_usd": {"type": "number", "minimum": 0},
        "currency": {"type": "string", "enum": ["USD", "EUR", "INR"]},
    },
    "required": ["customer", "line_items", "total_usd", "currency"],
    "additionalProperties": False,
}

# Счёт на 800 позиций: столько бывает у оптового заказа, и на нём видно
# разницу между рекурсией и повторным обходом схемы на каждом элементе.
_INVOICE = {
    "customer": "Acme Corp",
    "line_items": [
        {
            "sku": f"SKU-{i:05d}",
            "qty": random.randint(1, 20),
            "unit_usd": round(random.uniform(1, 500), 2),
        }
        for i in range(800)
    ],
    "total_usd": 12345.67,
    "currency": "USD",
}

_RAW = json.dumps(_INVOICE)
_ERRORS = [(f"$.line_items[{i}].qty", "below minimum 1") for i in range(200)]
_FIELDS = {f"field_{i:03d}": random.choice([str, int, float, bool]) for i in range(200)}

_MODEL = lambda feedback: {"content": _RAW}

BENCH = {
    "validate": (_INVOICE_SCHEMA, _INVOICE),
    "strict_mode_problems": (_INVOICE_SCHEMA,),
    "make_strict": (_INVOICE_SCHEMA,),
    "schema_from_fields": (_FIELDS,),
    "parse_output": ({"content": _RAW}, _INVOICE_SCHEMA),
    "retry_prompt": (_ERRORS,),
    "extract_with_retry": (_MODEL, _INVOICE_SCHEMA, 3),
}
