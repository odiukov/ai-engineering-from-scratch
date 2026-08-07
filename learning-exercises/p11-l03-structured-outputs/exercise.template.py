"""
Structured outputs: JSON Schema, валидация, повторные попытки

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l03-structured-outputs
Разбор:  /check-code p11-l03-structured-outputs
"""

import json
import re

TYPE_NAMES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def python_type_to_schema(py_type):
    """Питоновский тип -> кусок JSON Schema.

    python_type_to_schema(str)    ->  {"type": "string"}
    python_type_to_schema(float)  ->  {"type": "number"}
    python_type_to_schema(set)    ->  ValueError

    Обрати внимание: int это "integer", а float это "number". В JSON Schema
    это разные типы, и 1.5 не пройдёт проверку на integer.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
