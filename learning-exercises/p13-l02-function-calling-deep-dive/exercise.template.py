"""
Function calling у трёх провайдеров: один инструмент, три формы

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l02-function-calling-deep-dive
Разбор:  /check-code p13-l02-function-calling-deep-dive
"""

import json

PROVIDERS = ("openai", "anthropic", "gemini")
TOOL_LIMITS = {"openai": 128, "anthropic": 64, "gemini": 64}
DEPTH_LIMITS = {"openai": 5, "anthropic": 10, "gemini": 10}
ANTHROPIC_MODES = {"auto": "auto", "none": "none", "required": "any"}
GEMINI_MODES = {"auto": "AUTO", "none": "NONE", "required": "ANY", "force": "ANY"}


def schema_depth(schema):
    """Глубина вложенности JSON Schema. Скаляр — 1, объект — 1 + глубина полей.

    schema_depth({"type": "string"})                              ->  1
    schema_depth({"type": "object", "properties": {}})            ->  1
    schema_depth({"type": "object",
                  "properties": {"a": {"type": "string"}}})       ->  2
    schema_depth({"type": "array",
                  "items": {"type": "object",
                            "properties": {"a": {"type": "string"}}}})  ->  3

    OpenAI не принимает схему глубже пяти уровней. Считать надо рекурсивно и
    по properties, и по items: массив объектов съедает два уровня, а не один.
    """
    raise NotImplementedError


def gemini_schema(node):
    """Схема JSON Schema 2020-12 -> диалект Gemini. Новая структура, не правка.

    gemini_schema({"type": "object",
                   "properties": {"a": {"type": "integer"}},
                   "additionalProperties": False})
        ->  {"type": "OBJECT", "properties": {"a": {"type": "INTEGER"}}}
    gemini_schema({"type": "array", "items": {"type": "string"}})
        ->  {"type": "ARRAY", "items": {"type": "STRING"}}

    Два отличия диалекта: имена типов заглавными и ключа additionalProperties
    не существует — Gemini говорит на подмножестве OpenAPI 3.0.

    Ловушка: переписывать узлы на месте нельзя. Один и тот же канонический
    инструмент уходит и в OpenAI, и в Anthropic; испортишь схему здесь —
    сломаются оба, причём молча.
    """
    raise NotImplementedError


def declare(provider, tool):
    """Канонический инструмент -> объявление в форме провайдера.

    declare("openai", TOOL)
        ->  {"type": "function",
             "function": {"name": ..., "description": ...,
                          "parameters": <схема>, "strict": True}}
    declare("anthropic", TOOL)
        ->  {"name": ..., "description": ..., "input_schema": <схема>}
    declare("gemini", TOOL)
        ->  {"functionDeclarations": [{"name": ..., "description": ...,
                                       "parameters": <схема КАПСОМ>}]}

    Одна и та же тройка «имя, описание, схема» в трёх конвертах. Схема лежит
    в parameters у OpenAI и Gemini, в input_schema у Anthropic.

    Ловушка: ключа strict в форме Anthropic быть НЕ должно, даже если он есть
    в каноническом инструменте. У Anthropic такого флага нет, схема и так
    контракт, а лишнее поле в теле запроса — это 400 от API.
    """
    raise NotImplementedError


def tool_choice_for(provider, mode, tool_name=None):
    """Канонический режим выбора инструмента -> форма провайдера.

    Режимы: "auto" (модель решает), "required" (обязан вызвать хоть один),
    "none" (запрещено вызывать), "force" (обязан вызвать именно tool_name).

    tool_choice_for("openai", "auto")         ->  "auto"
    tool_choice_for("anthropic", "required")  ->  {"type": "any"}
    tool_choice_for("gemini", "none")
        ->  {"function_calling_config": {"mode": "NONE"}}
    tool_choice_for("openai", "force", "add")
        ->  {"type": "function", "function": {"name": "add"}}

    Обрати внимание: у OpenAI три режима из четырёх — просто строки, а у
    Anthropic и Gemini всегда словарь. Одинаковый смысл, разные типы.

    Неизвестный провайдер или режим — ValueError, а не молчаливый "auto":
    иначе опечатка в режиме превратится в разрешение вызывать что угодно.
    Режим "force" без tool_name — тоже ValueError.
    """
    raise NotImplementedError


def parse_tool_calls(provider, response):
    """Ответ провайдера -> канонический список вызовов.

    Каждый вызов: {"id": ..., "name": ..., "arguments": {...}}.
    Ответ без вызовов (модель просто написала текст) даёт пустой список.

    parse_tool_calls("openai", {"choices": [{"message": {"tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "add", "arguments": '{"a": 1}'}}]}}]})
        ->  [{"id": "call_1", "name": "add", "arguments": {"a": 1}}]

    Ловушка: OpenAI отдаёт arguments СТРОКОЙ с JSON внутри, а Anthropic и
    Gemini — уже разобранным объектом. Забудешь json.loads — получишь строку
    там, где ждал словарь, и упадёшь через два слоя от места ошибки.

    Вторая ловушка: у Anthropic вызовы лежат вперемешку с текстовыми блоками
    в одном массиве content. Блоки с type != "tool_use" нужно пропускать.
    """
    raise NotImplementedError


def make_tool_result(provider, call_id, name, content):
    """Результат инструмента -> сообщение в форме провайдера.

    make_tool_result("openai", "call_1", "add", "5")
        ->  {"role": "tool", "tool_call_id": "call_1", "content": "5"}
    make_tool_result("anthropic", "toolu_1", "add", "5")
        ->  {"role": "user",
             "content": [{"type": "tool_result",
                          "tool_use_id": "toolu_1", "content": "5"}]}
    make_tool_result("gemini", "fc-1", "add", "5")
        ->  {"functionResponse": {"id": "fc-1", "name": "add",
                                  "response": {"result": "5"}}}

    Три разных имени для одного и того же id: tool_call_id, tool_use_id, id.
    Имя инструмента нужно только Gemini — остальные находят вызов по id.

    Ловушка: у Anthropic результат приходит от роли user, а не tool. Роли
    tool в Messages API вообще не существует.
    """
    raise NotImplementedError


def check_limits(provider, tools):
    """Проверить набор инструментов на лимиты провайдера. Список претензий.

    Пустой список означает «запрос пройдёт».

    check_limits("openai", [tool] * 200)  ->  ["too many tools: 200 > 128"]
    check_limits("openai", [deep])        ->  ["deep: schema depth 6 > 5"]
    check_limits("anthropic", [deep])     ->  []   (у Anthropic лимит 10)

    Оба лимита провайдер проверяет у себя и отвечает 400 ещё до того, как
    модель увидит запрос. Проверять их на своей стороне дешевле: 400 из
    середины продакшена не говорит, какой именно инструмент виноват.

    Порядок претензий: сначала к набору целиком, потом по инструментам в
    порядке объявления.
    """
    raise NotImplementedError
