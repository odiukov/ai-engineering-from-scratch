"""
Function calling у трёх провайдеров: один инструмент, три формы — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Цикл вызова инструмента у OpenAI, Anthropic и Google одинаковый, а формы
сообщений разные. Библиотеки-обёртки (AbstractToolset в Pydantic AI,
UniversalToolNode в LangGraph, BaseTool в LlamaIndex) прячут разницу за одним
классом; здесь мы пишем этот слой руками. Соответствие настоящему API:

    schema_depth      <-  то, по чему OpenAI режет запрос ещё до модели
    gemini_schema     <-  перевод JSON Schema 2020-12 в диалект OpenAPI 3.0
    declare           <-  tools=[...] в теле запроса, три разные формы
    tool_choice_for   <-  tool_choice / tool_config.function_calling_config
    parse_tool_calls  <-  разбор tool_calls[] / content[] / parts[]
    make_tool_result  <-  сообщение роли tool / блок tool_result / functionResponse
    check_limits      <-  то, на чём запрос падает с 400 ещё до модели

Сети нет: мы переводим словари в словари. HTTP тут ничего не добавляет.

Канонический инструмент во всех функциях — словарь
    {"name": str, "description": str, "input_schema": dict, "strict": bool}
где "strict" необязателен и по умолчанию True.
"""

import json

PROVIDERS = ("openai", "anthropic", "gemini")

# Сколько инструментов провайдер принимает в одном запросе.
TOOL_LIMITS = {"openai": 128, "anthropic": 64, "gemini": 64}

# Предельная глубина вложенности схемы. У Anthropic формально не ограничена,
# на практике за десятью уровнями модель перестаёт попадать в схему.
DEPTH_LIMITS = {"openai": 5, "anthropic": 10, "gemini": 10}

# Как называется каждый канонический режим tool_choice у Anthropic и Gemini.
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
    children = [schema_depth(spec) for spec in schema.get("properties", {}).values()]
    if "items" in schema:
        children.append(schema_depth(schema["items"]))
    # default=0: пустой объект — это всё ещё один уровень, а не ноль
    return 1 + max(children, default=0)


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
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(value, str):
                out[key] = value.upper()
                continue
            out[key] = gemini_schema(value)
        return out
    if isinstance(node, list):
        return [gemini_schema(item) for item in node]
    # числа, строки и булевы значения внутри схемы (minimum, enum, ...)
    return node


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
    if provider == "openai":
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
                "strict": tool.get("strict", True),
            },
        }
    if provider == "anthropic":
        return {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }
    if provider == "gemini":
        return {
            "functionDeclarations": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": gemini_schema(tool["input_schema"]),
                }
            ]
        }
    raise ValueError(f"unknown provider: {provider}")


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
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    if mode not in ("auto", "required", "none", "force"):
        raise ValueError(f"unknown tool_choice mode: {mode}")
    if mode == "force" and not tool_name:
        raise ValueError("mode 'force' requires tool_name")

    if provider == "openai":
        if mode == "force":
            return {"type": "function", "function": {"name": tool_name}}
        return mode
    if provider == "anthropic":
        if mode == "force":
            return {"type": "tool", "name": tool_name}
        return {"type": ANTHROPIC_MODES[mode]}
    config = {"mode": GEMINI_MODES[mode]}
    if mode == "force":
        # у Gemini нет отдельного режима "именно этот инструмент":
        # ANY + белый список из одного имени
        config["allowed_function_names"] = [tool_name]
    return {"function_calling_config": config}


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
    if provider == "openai":
        message = response["choices"][0]["message"]
        return [
            {
                "id": item["id"],
                "name": item["function"]["name"],
                "arguments": json.loads(item["function"]["arguments"]),
            }
            for item in message.get("tool_calls") or []
        ]
    if provider == "anthropic":
        return [
            {"id": block["id"], "name": block["name"], "arguments": block["input"]}
            for block in response.get("content") or []
            if block.get("type") == "tool_use"
        ]
    if provider == "gemini":
        parts = response["candidates"][0]["content"].get("parts") or []
        return [
            {
                "id": part["functionCall"].get("id", ""),
                "name": part["functionCall"]["name"],
                "arguments": part["functionCall"]["args"],
            }
            for part in parts
            if "functionCall" in part
        ]
    raise ValueError(f"unknown provider: {provider}")


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
    if provider == "openai":
        return {"role": "tool", "tool_call_id": call_id, "content": content}
    if provider == "anthropic":
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content}
            ],
        }
    if provider == "gemini":
        # response обязан быть объектом: голую строку Gemini не принимает
        return {
            "functionResponse": {
                "id": call_id,
                "name": name,
                "response": {"result": content},
            }
        }
    raise ValueError(f"unknown provider: {provider}")


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
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    problems = []
    limit = TOOL_LIMITS[provider]
    if len(tools) > limit:
        problems.append(f"too many tools: {len(tools)} > {limit}")
    max_depth = DEPTH_LIMITS[provider]
    for tool in tools:
        depth = schema_depth(tool["input_schema"])
        if depth > max_depth:
            problems.append(f"{tool['name']}: schema depth {depth} > {max_depth}")
    return problems
