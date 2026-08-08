"""
Structured outputs: JSON Schema, валидация, повторные попытки — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import json
import re

# Отображение питоновских типов в имена типов JSON Schema.
TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Что может стоять на месте значения. Используется в next_valid_tokens.
VALUE_TOKENS = ['"', "0-9", "true", "false", "null", "[", "{"]


def strip_code_fence(raw):
    """Снять markdown-обёртку ```json ... ``` вокруг ответа модели.

    strip_code_fence('```json\\n{"a": 1}\\n```')  ->  '{"a": 1}'
    strip_code_fence('```\\n[1, 2]\\n```')        ->  '[1, 2]'
    strip_code_fence('{"a": 1}')                  ->  '{"a": 1}'

    Самый частый сбой формата: модель просят вернуть JSON, а она возвращает
    его внутри блока кода. Языковой тег после открывающих кавычек
    (json, JSON, javascript) — опционален.

    Если фенсов нет, вернуть исходный текст без обрамляющих пробелов.
    """
    text = raw.strip()
    # ленивый .*? и DOTALL: берём содержимое ПЕРВОГО блока, а не всё до
    # последних кавычек в файле
    match = re.match(r"^```[A-Za-z]*\s*\n?(.*?)\n?```$", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_llm_json(raw):
    """Разобрать ответ модели в объект. Возвращает (data, error).

    error — None при успехе, иначе строка с описанием проблемы.

    parse_llm_json('{"a": 1}')                    ->  ({"a": 1}, None)
    parse_llm_json('Here is the JSON: {"a": 1}')  ->  ({"a": 1}, None)
    parse_llm_json('not json at all')             ->  (None, "...")

    Две обработки перед json.loads: снять фенсы (strip_code_fence) и, если
    и так не разобралось, вырезать кусок от первой открывающей скобки до
    последней закрывающей — так отсекается преамбула вида "Here's the JSON:"
    и болтовня после.

    Соответствует тому, что делает Instructor перед валидацией.
    """
    text = strip_code_fence(raw)
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError):
        pass

    # вторая попытка: только то, что похоже на JSON-литерал
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if starts and ends and min(starts) < max(ends):
        try:
            return json.loads(text[min(starts) : max(ends) + 1]), None
        except (json.JSONDecodeError, TypeError) as exc:
            return None, str(exc)
    return None, "no JSON object or array found"


def validate(data, schema, path="$"):
    """Проверить данные по JSON Schema. Возвращает список ошибок (пустой — всё ок).

    validate({"a": 1}, {"type": "object", "properties": {"a": {"type": "integer"}}})
        ->  []
    validate({"price": -5}, {"type": "object",
                             "properties": {"price": {"type": "number", "minimum": 0}}})
        ->  одна ошибка про minimum

    Поддерживается: type (object/array/string/number/integer/boolean),
    required, properties, additionalProperties=False, enum, minimum/maximum,
    minItems/maxItems. Вложенность — рекурсией, path растёт как "$.a[0].b".

    Ловушки:
      * bool в Python — подкласс int, поэтому True пролезет как integer,
        если не отсечь его явно;
      * лишнее поле само по себе НЕ ошибка: жалуемся только когда в схеме
        стоит "additionalProperties": False.

    Соответствует BaseModel.model_validate() из Pydantic.
    """
    errors = []
    kind = schema.get("type")

    if kind == "object":
        if not isinstance(data, dict):
            return [f"{path}: expected object, got {type(data).__name__}"]
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                errors.append(f"{path}.{key}: required field missing")
        for key, value in data.items():
            if key in properties:
                errors.extend(validate(value, properties[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: additional property not allowed")

    elif kind == "array":
        if not isinstance(data, list):
            return [f"{path}: expected array, got {type(data).__name__}"]
        if len(data) < schema.get("minItems", 0):
            errors.append(f"{path}: has {len(data)} items, minItems is {schema['minItems']}")
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(f"{path}: has {len(data)} items, maxItems is {schema['maxItems']}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(data):
                errors.extend(validate(item, item_schema, f"{path}[{i}]"))

    elif kind == "string":
        if not isinstance(data, str):
            return [f"{path}: expected string, got {type(data).__name__}"]
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: '{data}' not in enum {schema['enum']}")

    elif kind in ("number", "integer"):
        # bool отсекаем первым: isinstance(True, int) истинно
        ok = not isinstance(data, bool) and isinstance(
            data, int if kind == "integer" else (int, float)
        )
        if not ok:
            return [f"{path}: expected {kind}, got {type(data).__name__}"]
        if "minimum" in schema and data < schema["minimum"]:
            errors.append(f"{path}: {data} is less than minimum {schema['minimum']}")
        if "maximum" in schema and data > schema["maximum"]:
            errors.append(f"{path}: {data} is greater than maximum {schema['maximum']}")
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: {data!r} not in enum {schema['enum']}")

    elif kind == "boolean":
        if not isinstance(data, bool):
            return [f"{path}: expected boolean, got {type(data).__name__}"]
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: {data!r} not in enum {schema['enum']}")

    return errors


def python_type_to_schema(py_type):
    """Питоновский тип -> кусок JSON Schema.

    python_type_to_schema(str)    ->  {"type": "string"}
    python_type_to_schema(float)  ->  {"type": "number"}
    python_type_to_schema(set)    ->  ValueError

    Обрати внимание: int это "integer", а float это "number". В JSON Schema
    это разные типы, и 1.5 не пройдёт проверку на integer.
    """
    if py_type not in TYPE_NAMES:
        raise ValueError(f"unsupported type: {py_type!r}")
    return {"type": TYPE_NAMES[py_type]}


def model_to_schema(fields):
    """Описание модели -> JSON Schema объекта.

    fields — dict {имя поля: спецификация}. Спецификация: обязательный ключ
    "type" (питоновский тип) и необязательные "required" (по умолчанию True),
    "enum", "minimum", "maximum", "items" (питоновский тип элементов массива).

    model_to_schema({"price": {"type": float, "minimum": 0}})
        ->  {"type": "object",
             "properties": {"price": {"type": "number", "minimum": 0}},
             "required": ["price"]}

    Порядок в required — порядок объявления полей, чтобы схема не прыгала
    от запуска к запуску.

    Соответствует Product.model_json_schema() из Pydantic.
    """
    properties, required = {}, []
    for name, spec in fields.items():
        prop = python_type_to_schema(spec["type"])
        if "items" in spec:
            prop["items"] = python_type_to_schema(spec["items"])
        for key in ("enum", "minimum", "maximum"):
            if key in spec:
                prop[key] = spec[key]
        properties[name] = prop
        if spec.get("required", True):
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def next_valid_tokens(partial_json):
    """Какие категории токенов допустимы после уже сгенерированного куска JSON.

    next_valid_tokens("")                  ->  ["{"]
    next_valid_tokens('{')                 ->  ['"', "}"]
    next_valid_tokens('{"price":')         ->  VALUE_TOKENS
    next_valid_tokens('{"price": 348}')    ->  ["<EOS>"]

    Это грубая модель constrained decoding: решение принимается по последнему
    значимому символу, без разбора вложенности. Настоящий движок (Outlines,
    XGrammar) компилирует схему в автомат и маскирует словарь на каждом шаге —
    здесь мы показываем идею, а не реализуем её целиком.

    Порядок проверок важен: строка, оканчивающаяся на '":', оканчивается на
    двоеточие, а не на кавычку. Если сначала спросить про кавычку, ветка
    про ':' окажется недостижимой.
    """
    text = partial_json.strip()
    if not text:
        return ["{"]
    try:
        json.loads(text)
        return ["<EOS>"]
    except (json.JSONDecodeError, TypeError):
        pass

    last = text[-1]
    if last == "{":
        return ['"', "}"]
    if last == "[":
        return VALUE_TOKENS + ["]"]
    if last in ":,":
        # после запятой объект закрыть нельзя: висячая запятая невалидна
        return VALUE_TOKENS
    if last == '"':
        # закрылась строка (ключ или значение) — дальше либо ':' , либо ',' , либо '}'
        return [":", ",", "}"]
    if last.isdigit():
        return ["0-9", ".", ",", "}", "]"]
    if last in "}]":
        return [",", "}", "]"]
    return ["a-z", '"']


def extract_with_retry(text, schema, call_model, max_retries=3):
    """Цикл «спросить модель — разобрать — проверить — переспросить с ошибками».

    call_model(text, attempt, errors) -> строка ответа модели. attempt
    считается с нуля, errors — список ошибок предыдущей попытки (на первой
    попытке пустой).

    Возвращает dict: {"data": объект или None, "attempts": сколько раз
    звали модель, "errors": ошибки последней попытки (пустой список при успехе)}.

    extract_with_retry(t, schema, lambda *_: '{"a": 1}')["attempts"]  ->  1

    Смысл передачи errors обратно в модель: это самая дешёвая коррекция из
    существующих. Именно так работает Instructor с max_retries.
    """
    errors = []
    for attempt in range(max_retries):
        data, parse_error = parse_llm_json(call_model(text, attempt, errors))
        if parse_error is not None:
            errors = [parse_error]
            continue
        errors = validate(data, schema)
        if not errors:
            return {"data": data, "attempts": attempt + 1, "errors": []}
    return {"data": None, "attempts": max_retries, "errors": errors}
