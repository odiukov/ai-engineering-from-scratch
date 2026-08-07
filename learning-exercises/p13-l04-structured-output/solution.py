"""
Структурированный вывод: JSON Schema, strict mode, отказы — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Pydantic, Zod и strict mode у OpenAI делают одно и то же тремя способами:
объявляют схему один раз и заставляют модель ей соответствовать. Здесь мы
пишем этот слой руками. Соответствие настоящему API:

    validate             <-  ValidationError с путями у Pydantic v2
    strict_mode_problems <-  400 от OpenAI при response_format strict
    make_strict          <-  то, что Pydantic дописывает в model_json_schema()
    schema_from_fields   <-  BaseModel -> JSON Schema
    parse_output         <-  разбор поля content либо refusal из ответа
    retry_prompt         <-  текст, который скармливают модели на повторе
    extract_with_retry   <-  цикл generate -> parse -> validate -> retry

Модели и сети нет: шаг «спросить модель» подменён функцией-параметром.

Ошибка валидации везде — пара (путь, сообщение), например
("$.line_items[0].qty", "below minimum 1").
"""

import json
import re

# Соответствие типов JSON Schema питоновским. "number" принимает и int, и float.
JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}

# Питоновские типы -> имена типов JSON Schema. bool раньше int: он его подкласс.
PYTHON_TYPES = [
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
]


def validate(schema, value, path="$"):
    """Проверить значение по JSON Schema. Список пар (путь, сообщение).

    Пустой список означает «сходится».

    validate({"type": "integer", "minimum": 1}, 0)
        ->  [("$", "below minimum 1")]
    validate({"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}, {})
        ->  [("$.a", "missing required field")]

    Поддержано подмножество 2020-12: type, required, properties, items, enum,
    minimum, maximum, minLength, maxLength, pattern, additionalProperties.

    Путь важнее сообщения: именно он уходит в промпт повторной попытки, и по
    нему модель понимает, какое поле чинить в глубоко вложенном объекте.

    Ловушка: в Python bool — подкласс int, поэтому True прошёл бы и как
    integer, и как number. Отсекай его отдельно.
    """
    errors = []
    expected = schema.get("type")

    if expected is not None:
        allowed = JSON_TYPES[expected]
        ok = isinstance(value, allowed) and not (
            expected in ("integer", "number") and isinstance(value, bool)
        )
        if not ok:
            # тип не тот — остальные проверки бессмысленны, они бы упали
            return [(path, f"expected {expected}, got {type(value).__name__}")]

    if expected == "object":
        properties = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in value:
                errors.append((f"{path}.{field}", "missing required field"))
        if schema.get("additionalProperties") is False:
            for extra in value:
                if extra not in properties:
                    errors.append((f"{path}.{extra}", "additional property not allowed"))
        for field, sub in properties.items():
            if field in value:
                errors.extend(validate(sub, value[field], f"{path}.{field}"))
    elif expected == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(schema["items"], item, f"{path}[{i}]"))
    elif expected == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append((path, f"shorter than minLength {schema['minLength']}"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append((path, f"longer than maxLength {schema['maxLength']}"))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append((path, f"does not match pattern {schema['pattern']!r}"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append((path, f"below minimum {schema['minimum']}"))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append((path, f"above maximum {schema['maximum']}"))
    if "enum" in schema and value not in schema["enum"]:
        errors.append((path, f"value {value!r} not in enum {schema['enum']}"))
    return errors


def strict_mode_problems(schema, path="$"):
    """Что мешает схеме пройти strict mode у OpenAI. Список (путь, сообщение).

    Три требования, и все три проверяются рекурсивно:
      * у каждого объекта additionalProperties: false;
      * КАЖДОЕ свойство перечислено в required (необязательных полей в strict
        mode не бывает — необязательность выражают типом ["string", "null"]);
      * никаких $ref.

    strict_mode_problems({"type": "object",
                          "properties": {"a": {"type": "string"}},
                          "required": [], "additionalProperties": False})
        ->  [("$.a", "property not listed in required")]

    Эти три ошибки OpenAI отдаёт как 400 в момент запроса, ещё до модели.
    Anthropic и Gemini их не проверяют, поэтому одна и та же схема
    работает у двоих и валится у третьего.
    """
    problems = []
    if "$ref" in schema:
        problems.append((path, "$ref is not allowed in strict mode"))
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            problems.append((path, "additionalProperties must be false"))
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for field in properties:
            if field not in required:
                problems.append((f"{path}.{field}", "property not listed in required"))
        for field, sub in properties.items():
            problems.extend(strict_mode_problems(sub, f"{path}.{field}"))
    elif schema.get("type") == "array" and "items" in schema:
        problems.extend(strict_mode_problems(schema["items"], f"{path}[]"))
    return problems


def make_strict(schema):
    """Переписать схему так, чтобы она прошла strict mode. Новая схема.

    Каждому объекту дописывается additionalProperties: false и required со
    ВСЕМИ его свойствами. Рекурсивно, включая элементы массивов.

    make_strict({"type": "object", "properties": {"a": {"type": "string"}}})
        ->  {"type": "object", "properties": {"a": {"type": "string"}},
             "required": ["a"], "additionalProperties": False}

    Проверяемое свойство: strict_mode_problems(make_strict(s)) всегда пусто,
    если в схеме не было $ref — его дописыванием ключей не убрать.

    Ловушка: исходную схему править нельзя. Она обычно живёт как константа
    модуля и уходит ещё в двух провайдеров, где strict mode не нужен.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object":
        properties = {
            name: make_strict(sub) for name, sub in out.get("properties", {}).items()
        }
        out["properties"] = properties
        # порядок required = порядок объявления свойств, так диффы читаемее
        out["required"] = list(properties)
        out["additionalProperties"] = False
    elif out.get("type") == "array" and "items" in out:
        out["items"] = make_strict(out["items"])
    return out


def schema_from_fields(fields):
    """Питоновские типы полей -> strict-совместимая JSON Schema.

    fields — словарь {имя поля: питоновский тип}, порядок сохраняется.

    schema_from_fields({"customer": str, "total_usd": float})
        ->  {"type": "object",
             "properties": {"customer": {"type": "string"},
                            "total_usd": {"type": "number"}},
             "required": ["customer", "total_usd"],
             "additionalProperties": False}

    Ровно это делает Pydantic в model_json_schema(): читает аннотации и
    собирает схему. Разница в том, что Pydantic знает вложенные модели, а
    мы ограничиваемся плоскими полями.

    Ловушка: bool нужно проверять ДО int, иначе True станет "integer" —
    bool в Python подкласс int.
    """
    properties = {}
    for name, py_type in fields.items():
        json_name = next((n for t, n in PYTHON_TYPES if py_type is t), None)
        if json_name is None:
            raise ValueError(f"{name}: unsupported python type {py_type!r}")
        properties[name] = {"type": json_name}
    return make_strict({"type": "object", "properties": properties})


def parse_output(message, schema):
    """Ответ модели -> один из четырёх типизированных исходов.

    message — словарь провайдера: {"content": <строка или None>,
                                   "refusal": <строка или None>}.

    Результат всегда одной формы:
        {"kind": "ok" | "refusal" | "parse_error" | "violation",
         "value": <разобранный JSON или None>,
         "errors": [(путь, сообщение), ...],
         "reason": <причина отказа или None>}

    parse_output({"content": '{"a": 1}'}, {"type": "object", ...})
        ->  {"kind": "ok", "value": {"a": 1}, "errors": [], "reason": None}
    parse_output({"refusal": "This is a poem, not an invoice."}, schema)
        ->  {"kind": "refusal", "value": None, "errors": [], "reason": "..."}

    Отказ проверяется ПЕРВЫМ и разбирать content при нём не надо: strict mode
    не умеет заставить модель ответить, и отказ — это штатный исход, а не
    сбой. Путать его с ошибкой валидации значит уйти в повторные попытки
    там, где повторять нечего.
    """
    refusal = message.get("refusal")
    if refusal:
        return {"kind": "refusal", "value": None, "errors": [], "reason": refusal}
    try:
        value = json.loads(message.get("content") or "")
    except ValueError as exc:  # JSONDecodeError — её подкласс
        return {"kind": "parse_error", "value": None, "errors": [("$", str(exc))], "reason": None}
    errors = validate(schema, value)
    if errors:
        return {"kind": "violation", "value": value, "errors": errors, "reason": None}
    return {"kind": "ok", "value": value, "errors": [], "reason": None}


def retry_prompt(errors):
    """Ошибки валидации -> текст, который дописывают в промпт повтора.

    retry_prompt([("$.total_usd", "below minimum 0")])
        ->  "Your previous output did not match the schema. Fix these and "
            "reply with JSON only:\\n- $.total_usd: below minimum 0"

    Ошибок нет — пустая строка: чинить нечего, повторять незачем.

    Смысл: модель должна увидеть ПУТЬ и ПРЕТЕНЗИЮ, а не «invalid JSON».
    Замеры показывают, что типизированное сообщение об ошибке вдвое
    сокращает число повторов на слабых моделях.
    """
    if not errors:
        return ""
    lines = "\n".join(f"- {path}: {message}" for path, message in errors)
    return (
        "Your previous output did not match the schema. "
        "Fix these and reply with JSON only:\n" + lines
    )


def extract_with_retry(call_model, schema, max_attempts=3):
    """Цикл извлечения: спросить, разобрать, проверить, при неудаче повторить.

    call_model(feedback) — подмена модели. На первой попытке feedback=None,
    дальше — строка из retry_prompt. Возвращает словарь сообщения провайдера.

    Результат — то же, что у parse_output, плюс ключ "attempts".

    Три правила, и все три несут смысл:
      * успех останавливает цикл сразу;
      * ОТКАЗ тоже останавливает цикл сразу и attempts остаётся 1 — модель
        не «ошиблась», она сказала, что задача не решается; повторять её —
        значит платить за три одинаковых отказа;
      * ошибка разбора или схемы даёт повтор, но не больше max_attempts.

    Ловушка: считать попытки надо реально сделанные, а не max_attempts.
    """
    feedback = None
    result = None
    for attempt in range(1, max_attempts + 1):
        result = parse_output(call_model(feedback), schema)
        result["attempts"] = attempt
        if result["kind"] in ("ok", "refusal"):
            return result
        feedback = retry_prompt(result["errors"])
    return result
