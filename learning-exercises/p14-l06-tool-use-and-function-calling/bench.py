"""Входные данные для замера скорости."""

_N = 300

_wide_schema = {
    "type": "object",
    "properties": {f"f{i}": {"type": "integer", "minimum": 0, "maximum": 10 ** 9}
                   for i in range(_N)},
    "required": [f"f{i}" for i in range(_N)],
}
_wide_args = {f"f{i}": str(i) for i in range(_N)}


def _sink(**kwargs):
    return len(kwargs)


def _add(a, b):
    return a + b


_int_pair = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}

# make_tool сам собирает словарь, поэтому реестр строим им же: bench не должен
# знать раскладку ключей инструмента лучше, чем её знает эталон.
_tool_list = [
    {"name": f"tool{i}",
     "description": f"Tool {i}. Use when topic{i} comes up in the request.",
     "input_schema": _int_pair,
     "executor": _add,
     "timeout_s": 5.0}
    for i in range(_N)
]
_tool_list.append({"name": "sink",
                   "description": "Use when many fields arrive at once.",
                   "input_schema": _wide_schema,
                   "executor": _sink,
                   "timeout_s": 5.0})
_registry = {t["name"]: t for t in _tool_list}

_calls = [{"tool_use_id": f"u{i}", "name": f"tool{i}", "args": {"a": i, "b": 1}}
          for i in range(_N)]

# Обратный порядок завершения: dispatch_many обязана отдавать наблюдения
# в порядке вызовов, а не в порядке готовности, и замер это нагружает.
# completion_order — перестановка ИНДЕКСОВ calls, не tool_use_id.
_reverse_order = list(reversed(range(len(_calls))))

# Много исходов подряд, чтобы breaker_allows пришлось просканировать хвост.
# outcomes — пары (at, ok) в хронологическом порядке.
_outcomes = [(float(i), i % 5 != 0) for i in range(_N)]

BENCH = {
    "coerce_value": ("12345", {"type": "integer"}),
    "validate_args": (_wide_args, _wide_schema),
    "make_tool": ("add", "Add integers a and b when asked.", _int_pair, _add),
    "build_registry": (_tool_list,),
    "tool_catalog": (_registry,),
    "dispatch": (_registry, {"tool_use_id": "u1", "name": "tool1",
                             "args": {"a": 2, "b": 3}}),
    "dispatch_many": (_registry, _calls, _reverse_order),
    "breaker_allows": (_outcomes, float(_N)),
}
