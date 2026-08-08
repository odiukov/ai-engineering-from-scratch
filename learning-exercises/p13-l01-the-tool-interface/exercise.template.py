"""
Интерфейс инструмента: цикл из четырёх шагов

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l01-the-tool-interface
Разбор:  /check-code p13-l01-the-tool-interface
"""

import copy
import json

MAX_TURNS = 5
JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def make_tool(name, description, input_schema, executor, consequential=False):
    """Запись реестра: имя, описание, JSON Schema аргументов и исполнитель.

    make_tool("add", "Use when ...", {"type": "object", "properties": {}},
              lambda args: args["a"] + args["b"])
        ->  {"name": "add", "description": "Use when ...",
             "input_schema": {...}, "executor": <функция>,
             "consequential": False}

    consequential=True помечает инструмент, который тратит деньги или меняет
    состояние (send_email, delete_file). Флаг живёт только у хоста: модели он
    не показывается, по нему хост решает, спрашивать ли пользователя.

    Ловушка: схему нужно СКОПИРОВАТЬ. Если положить в реестр ссылку на словарь
    вызывающего кода, его позднейшая правка тихо переопределит контракт уже
    опубликованного инструмента.
    """
    raise NotImplementedError


def describe_registry(registry):
    """Шаг describe: что из реестра уходит модели. Порядок сохраняется.

    describe_registry([make_tool("add", "Use when ...", SCHEMA, fn)])
        ->  [{"name": "add", "description": "Use when ...",
              "input_schema": SCHEMA}]

    Наружу идут ровно три поля. executor — питоновская функция, её нечем
    сериализовать. consequential — приватное решение хоста о безопасности,
    модели про него знать незачем: покажешь флаг — модель начнёт его
    обсуждать вместо того, чтобы вызывать инструмент.
    """
    raise NotImplementedError


def validate_arguments(input_schema, arguments):
    """Проверить arguments по схеме. Вернуть список претензий, пустой — годно.

    validate_arguments(WEATHER, {"city": "Tokyo"})     ->  []
    validate_arguments(WEATHER, {})                    ->  ["missing required property: city"]
    validate_arguments(WEATHER, {"city": 5})           ->  ["city: expected string, got int"]

    Поддерживается подмножество JSON Schema 2020-12: type, required, enum и
    additionalProperties. Проверка идёт ДО запуска: аргументы приходят
    от модели, и на слабых моделях выдуманное поле или не тот тип — рядовое
    событие, а не экзотика.

    Ловушка: в Python bool — подкласс int, поэтому isinstance(True, int) даёт
    True. Без отдельной проверки {"qty": True} проедет как integer.
    """
    raise NotImplementedError


def make_tool_call(call_id, name, arguments):
    """Шаг decide: то, что модель эмитит вместо текста.

    make_tool_call("call_1", "add", {"a": 2, "b": 3})
        ->  {"id": "call_1", "name": "add", "arguments": {"a": 2, "b": 3}}

    Три поля стабильны у всех провайдеров, меняются только имена обёрток:
    OpenAI кладёт это в tool_calls[], Anthropic — в блок tool_use, Gemini —
    в parts[].functionCall.

    id не украшение: при параллельных вызовах результаты возвращаются не в том
    порядке, в каком их запросили, и склеить результат с вызовом можно только
    по id. Вызов без id — это ошибка, а не «ну хоть что-то».
    """
    raise NotImplementedError


def execute_call(registry, call):
    """Шаг execute: найти инструмент, проверить аргументы, запустить.

    Вернуть сообщение роли tool — то, что уйдёт обратно в модель:

        {"role": "tool", "tool_call_id": <id вызова>, "name": <имя>,
         "content": <строка>, "is_error": <bool>}

    execute_call(REG, {"id": "c1", "name": "add", "arguments": {"a": 2, "b": 3}})
        ->  {..., "content": "5", "is_error": False}
    execute_call(REG, {"id": "c2", "name": "nope", "arguments": {}})
        ->  {..., "content": "Unknown tool: nope. Available: add", "is_error": True}

    Три вида неудачи — нет такого инструмента, аргументы не по схеме, упал сам
    исполнитель — дают ОДИНАКОВУЮ форму: обычное сообщение с is_error=True и
    ТЕМ ЖЕ tool_call_id. Модель должна увидеть ошибку и попробовать иначе;
    исключение, вылетевшее из хоста, отнимает у неё эту возможность.

    content — всегда строка: строковый результат уходит как есть, всё
    остальное сериализуется в JSON.
    """
    raise NotImplementedError


def needs_confirmation(registry, call):
    """Нужно ли спросить пользователя, прежде чем выполнять вызов.

    needs_confirmation(REG, {"id": "c1", "name": "add", "arguments": {}})
        ->  False   (чистый инструмент, побочных эффектов нет)
    needs_confirmation(REG, {"id": "c2", "name": "send_email", "arguments": {}})
        ->  True    (consequential)
    needs_confirmation(REG, {"id": "c3", "name": "nope", "arguments": {}})
        ->  True    (инструмента нет в реестре)

    Неизвестный инструмент даёт True, а не False. Гейт по умолчанию закрыт:
    если хост не знает, что это за вызов, он тем более не знает, что тот
    натворит. Ошибиться в сторону лишнего вопроса дёшево, в обратную — нет.
    """
    raise NotImplementedError


def run_loop(registry, user_message, decide, max_turns=MAX_TURNS):
    """Цикл из четырёх шагов целиком. Вернуть стенограмму и причину остановки.

    decide(messages) — подмена модели. Возвращает либо {"content": "текст"}
    (финальный ответ), либо {"tool_calls": [<вызов>, ...]}.

    Результат:
        {"messages": [...], "stop_reason": "final" | "max_turns", "turns": int}

    run_loop(REG, "2+3?", lambda msgs: {"content": "5"})
        ->  {"messages": [<user>, <assistant>], "stop_reason": "final", "turns": 1}

    Каждый ход: decide -> для каждого вызова execute -> результаты в историю
    (шаг observe) -> следующий ход. Выход из цикла — либо модель ответила
    текстом, либо исчерпан лимит ходов.

    Ловушка: без лимита цикл, из которого модель не умеет выйти, крутится,
    пока не кончатся деньги. Пост-мортемы «агент сжёг $400 за ночь» — ровно
    этот забытый предохранитель.
    """
    raise NotImplementedError
