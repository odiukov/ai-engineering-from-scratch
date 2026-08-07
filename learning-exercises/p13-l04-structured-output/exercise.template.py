"""
Структурированный вывод: JSON Schema, strict mode, отказы

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l04-structured-output
Разбор:  /check-code p13-l04-structured-output
"""

import json
import re

JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
