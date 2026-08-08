"""
Интерфейс инструмента: цикл из четырёх шагов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок собирает руками то, что провайдерский SDK прячет за одним вызовом
`client.messages.create(tools=[...])`. Соответствие настоящему API:

    make_tool           <-  запись в реестре инструментов хоста
    describe_registry   <-  массив tools в теле запроса к модели (шаг describe)
    validate_arguments  <-  проверка arguments по inputSchema до запуска
    make_tool_call      <-  блок tool_use / элемент tool_calls[] (шаг decide)
    execute_call        <-  шаг execute + сообщение роли tool с тем же id
    needs_confirmation  <-  гейт подтверждения перед consequential-вызовом
    run_loop            <-  агентный цикл хоста с ограничением по числу ходов

Сети нет и модели нет: шаг decide подменён функцией, которую передают
параметром. Ровно так его подменяют в тестах настоящего агента.
"""

import copy
import json

# Предохранитель: сколько ходов цикла хост готов отработать, прежде чем
# признать, что модель зациклилась. Claude Code ставит 20, OpenAI Assistants
# 10, Cursor 25. Для урока хватит пяти.
MAX_TURNS = 5

# Соответствие типов JSON Schema питоновским. "number" принимает и int, и float.
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
    return {
        "name": name,
        "description": description,
        # deepcopy, а не dict(): properties — вложенный словарь, поверхностная
        # копия оставила бы его общим с вызывающим кодом
        "input_schema": copy.deepcopy(input_schema),
        "executor": executor,
        "consequential": consequential,
    }


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
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }
        for tool in registry
    ]


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
    if not isinstance(arguments, dict):
        return [f"expected object, got {type(arguments).__name__}"]

    properties = input_schema.get("properties", {})
    problems = []
    # сначала обязательные поля: их отсутствие важнее опечаток в остальных
    for name in input_schema.get("required", []):
        if name not in arguments:
            problems.append(f"missing required property: {name}")

    for name, value in arguments.items():
        spec = properties.get(name)
        if spec is None:
            # JSON Schema разрешает дополнительные свойства по умолчанию.
            # Закрытой схема становится только с явным false.
            if input_schema.get("additionalProperties", True) is False:
                problems.append(f"unknown property: {name}")
            continue
        expected = spec.get("type")
        if expected is not None:
            allowed = JSON_TYPES[expected]
            # bool отсекаем руками: True прошёл бы и как integer, и как number
            ok = isinstance(value, allowed) and not (
                expected in ("integer", "number") and isinstance(value, bool)
            )
            if not ok:
                problems.append(
                    f"{name}: expected {expected}, got {type(value).__name__}"
                )
                # тип не тот — проверять enum поверх этого бессмысленно
                continue
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{name}: {value!r} not in enum {spec['enum']}")
    return problems


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
    if not call_id:
        raise ValueError("tool call must carry a non-empty id")
    return {"id": call_id, "name": name, "arguments": dict(arguments)}


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
    by_name = {tool["name"]: tool for tool in registry}
    tool = by_name.get(call["name"])

    def message(content, is_error):
        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "name": call["name"],
            "content": content,
            "is_error": is_error,
        }

    if tool is None:
        available = ", ".join(sorted(by_name))
        return message(f"Unknown tool: {call['name']}. Available: {available}", True)

    problems = validate_arguments(tool["input_schema"], call["arguments"])
    if problems:
        # текст ошибки читает модель, а не человек: он должен объяснять,
        # что именно чинить в следующей попытке
        return message("Invalid input: " + "; ".join(problems), True)

    try:
        result = tool["executor"](call["arguments"])
    except Exception as exc:  # noqa: BLE001 - падение инструмента не роняет хост
        return message(f"{type(exc).__name__}: {exc}", True)

    content = result if isinstance(result, str) else json.dumps(result, sort_keys=True)
    return message(content, False)


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
    for tool in registry:
        if tool["name"] == call["name"]:
            return bool(tool["consequential"])
    return True


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
    messages = [{"role": "user", "content": user_message}]
    for turn in range(1, max_turns + 1):
        decision = decide(messages)
        if "content" in decision:
            messages.append({"role": "assistant", "content": decision["content"]})
            return {"messages": messages, "stop_reason": "final", "turns": turn}
        calls = decision["tool_calls"]
        messages.append({"role": "assistant", "tool_calls": calls})
        for call in calls:
            messages.append(execute_call(registry, call))
    return {"messages": messages, "stop_reason": "max_turns", "turns": max_turns}
