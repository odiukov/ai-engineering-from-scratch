"""
Tool use и function calling — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь руками собирается то, что провайдеры дают одной строкой: реестр
инструментов, валидатор подмножества JSON Schema, приведение типов,
параллельный диспетчер с корреляционными id и circuit breaker. В Anthropic
API это `tools=[...]` с `input_schema` и блоки `tool_use`/`tool_result`; в
OpenAI Agents SDK — декоратор `@function_tool`, который сам строит схему из
аннотаций. Ни сети, ни LLM: инструмент — обычная функция.
"""

# Описание инструмента модель читает, чтобы выбрать инструмент. Пустое или
# однословное описание — главная причина «выбран не тот инструмент», поэтому
# реестр отказывается такое принимать.
MIN_DESCRIPTION_WORDS = 5


def coerce_value(value, prop):
    """Привести значение к типу из схемы. Вернуть (value, error).

    error is None означает успех.

    coerce_value("5", {"type": "integer"})   ->  (5, None)
    coerce_value(2, {"type": "number"})      ->  (2.0, None)
    coerce_value("x", {"type": "integer"})
        ->  ("x", "cannot coerce string 'x' to integer")
    coerce_value(5, {"type": "string"})
        ->  (5, "expected string, got int")

    Приводим только однозначное: строку "5" к integer можно, число 5 к
    string — нельзя, иначе коэрсия начнёт прятать реальные баги.

    Ловушка: в Python bool — подкласс int, поэтому True без отдельной
    проверки просочится как integer 1. Схема, которая просит integer, а
    получает True, обязана ругаться.

    Схема без "type" пропускается как есть: валидируем то, что описано.
    """
    t = prop.get("type")
    if t == "integer":
        # isinstance(True, int) == True, поэтому bool отсекаем явно
        if isinstance(value, int) and not isinstance(value, bool):
            return (value, None)
        if isinstance(value, str):
            try:
                return (int(value), None)
            except ValueError:
                return (value, f"cannot coerce string {value!r} to integer")
        return (value, f"expected integer, got {type(value).__name__}")
    if t == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (float(value), None)
        if isinstance(value, str):
            try:
                return (float(value), None)
            except ValueError:
                return (value, f"cannot coerce string {value!r} to number")
        return (value, f"expected number, got {type(value).__name__}")
    if t == "boolean":
        # строку "true" сознательно не принимаем: она неоднозначна
        if isinstance(value, bool):
            return (value, None)
        return (value, f"expected boolean, got {type(value).__name__}")
    if t == "string":
        if isinstance(value, str):
            return (value, None)
        return (value, f"expected string, got {type(value).__name__}")
    if t == "array":
        if isinstance(value, list):
            return (value, None)
        return (value, f"expected array, got {type(value).__name__}")
    if t == "object":
        if isinstance(value, dict):
            return (value, None)
        return (value, f"expected object, got {type(value).__name__}")
    return (value, None)


def validate_args(args, schema):
    """Проверить аргументы вызова по схеме. Вернуть (validated, errors).

    schema — подмножество JSON Schema: properties, required, enum,
    minimum, maximum.

    s = {"properties": {"a": {"type": "integer"}}, "required": ["a"]}
    validate_args({"a": "5"}, s)  ->  ({"a": 5}, [])
    validate_args({}, s)          ->  ({}, ["missing required: a"])
    validate_args({"a": 1, "b": 2}, s)
        ->  ({"a": 1}, ["unknown field: b"])

    Порядок ошибок фиксирован: сначала пропущенные обязательные поля в
    порядке required, потом остальные в порядке args.

    Ловушка: возвращать надо ВСЕ ошибки, а не первую. Модель получает их
    одной наблюдаемой строкой и чинит вызов за одну попытку, а не за пять.

    Поле, на котором нашлась ошибка, в validated не попадает — иначе
    исполнитель получит невалидное значение.
    """
    props = schema.get("properties", {})
    required = schema.get("required", [])
    validated = {}
    errors = []

    for name in required:
        if name not in args:
            errors.append(f"missing required: {name}")

    for name, value in args.items():
        prop = props.get(name)
        if prop is None:
            # лишний аргумент — это не мелочь: модель придумала поле
            errors.append(f"unknown field: {name}")
            continue
        coerced, err = coerce_value(value, prop)
        if err:
            errors.append(f"{name}: {err}")
            continue
        if "enum" in prop and coerced not in prop["enum"]:
            errors.append(f"{name}: {coerced!r} not in {prop['enum']}")
            continue
        if prop.get("type") in ("integer", "number"):
            if "minimum" in prop and coerced < prop["minimum"]:
                errors.append(f"{name}: {coerced} < minimum {prop['minimum']}")
                continue
            if "maximum" in prop and coerced > prop["maximum"]:
                errors.append(f"{name}: {coerced} > maximum {prop['maximum']}")
                continue
        validated[name] = coerced

    return (validated, errors)


def make_tool(name, description, input_schema, executor, timeout_s=5.0):
    """Собрать описание инструмента. Вернуть словарь с этими же ключами.

    make_tool("add", "Add two integers a and b together.",
              {"properties": {}}, sum)["name"]  ->  'add'

    ValueError, если:
      * name пустое;
      * в description меньше MIN_DESCRIPTION_WORDS слов;
      * input_schema не словарь или в нём нет "properties";
      * timeout_s не положительный.

    Проверка длины описания — не придирка. Модель выбирает инструмент по
    описанию, и «Add.» вместо «Add two integers; use for any integer
    addition» даёт ровно тот класс ошибок, который в BFCL считается
    wrong-tool-picked.

    Таймаут и границы доступа тут только объявляются: настоящая песочница —
    отдельный урок, но объявить их надо на этапе регистрации, иначе позже
    их никто не добавит.
    """
    if not name or not name.strip():
        raise ValueError("tool name must not be empty")
    if len(description.split()) < MIN_DESCRIPTION_WORDS:
        raise ValueError(
            f"description must have at least {MIN_DESCRIPTION_WORDS} words: {description!r}"
        )
    if not isinstance(input_schema, dict) or "properties" not in input_schema:
        raise ValueError("input_schema must be a dict with a 'properties' key")
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "executor": executor,
        "timeout_s": timeout_s,
    }


def build_registry(tools):
    """Индекс инструментов по имени. Вернуть словарь name -> tool.

    build_registry([])  ->  {}

    ValueError на дублирующемся имени: два инструмента с одним именем
    означают, что один из них молча недостижим, а модель об этом не
    узнает.
    """
    registry = {}
    for tool in tools:
        if tool["name"] in registry:
            raise ValueError(f"duplicate tool name: {tool['name']!r}")
        registry[tool["name"]] = tool
    return registry


def tool_catalog(registry):
    """Каталог для промпта: только name, description, input_schema.

    Список отсортирован по имени.

    tool_catalog({})  ->  []

    Исполнитель и таймаут в каталог не попадают: это внутренние детали
    рантайма, модели они не нужны, а в промпте они стоят токенов.

    Сортировка нужна для стабильности промпта: переставь инструменты — и
    prompt caching перестанет попадать в кэш на каждом запросе.
    """
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }
        for tool in sorted(registry.values(), key=lambda t: t["name"])
    ]


def dispatch(registry, call):
    """Выполнить один tool call. Вернуть словарь-наблюдение.

    call — словарь с ключами tool_use_id, name, args.
    Ответ — словарь с ключами tool_use_id, ok, content (content всегда str).

    Ошибки НЕ бросаются наружу, а возвращаются как наблюдение:
      неизвестный инструмент  ->  "error: unknown tool 'subtract'"
      провал валидации        ->  "validation error: unknown field: c"
      падение исполнителя     ->  "execution error: ZeroDivisionError: ..."

    Порядок шагов важен: сначала валидация, только потом вызов
    исполнителя. Вызов с лишним аргументом обязан быть отвергнут ДО того,
    как инструмент что-то запишет или отправит.

    tool_use_id возвращается тем же — по нему провайдер сшивает tool_use и
    tool_result. Потеряешь его, и в параллельном turn результаты уедут не
    к тем вызовам.
    """
    tool = registry.get(call["name"])
    if tool is None:
        return {
            "tool_use_id": call["tool_use_id"],
            "ok": False,
            "content": f"error: unknown tool {call['name']!r}",
        }
    validated, errors = validate_args(call["args"], tool["input_schema"])
    if errors:
        return {
            "tool_use_id": call["tool_use_id"],
            "ok": False,
            "content": "validation error: " + "; ".join(errors),
        }
    try:
        content = tool["executor"](**validated)
    except Exception as exc:
        # исключение исполнителя — тоже наблюдение: агент должен получить
        # шанс починить вызов, а не уронить весь цикл
        return {
            "tool_use_id": call["tool_use_id"],
            "ok": False,
            "content": f"execution error: {type(exc).__name__}: {exc}",
        }
    return {"tool_use_id": call["tool_use_id"], "ok": True, "content": str(content)}


def dispatch_many(registry, calls, completion_order=None):
    """Параллельный turn: несколько вызовов сразу. Вернуть список наблюдений.

    completion_order — перестановка индексов calls, задающая порядок, в
    котором инструменты фактически завершились. По умолчанию — как пришли.

    Результаты возвращаются в порядке calls независимо от
    completion_order: сшивка идёт по tool_use_id, а не по времени ответа.

    ValueError, если completion_order не перестановка индексов или если
    среди вызовов есть повторяющийся tool_use_id.

    Здесь и живёт главный баг параллельного tool use: перепутанные
    корреляционные id отправляют результат одного инструмента как ответ
    другого, и модель уверенно продолжает по чужим данным.
    """
    order = list(range(len(calls))) if completion_order is None else list(completion_order)
    if sorted(order) != list(range(len(calls))):
        raise ValueError("completion_order must be a permutation of call indices")
    by_id = {}
    for index in order:
        call = calls[index]
        if call["tool_use_id"] in by_id:
            raise ValueError(f"duplicate tool_use_id: {call['tool_use_id']!r}")
        by_id[call["tool_use_id"]] = dispatch(registry, call)
    # сборка по исходному порядку вызовов, а не по порядку завершения
    return [by_id[call["tool_use_id"]] for call in calls]


def breaker_allows(outcomes, now, threshold=3, cooldown_s=60.0):
    """Circuit breaker: можно ли ещё звать этот инструмент.

    outcomes — список пар (at, ok) в хронологическом порядке: время
    попытки и её исход. now — текущее время, тоже параметром.

    breaker_allows([], 100.0)                              ->  True
    breaker_allows([(1.0, False), (2.0, False)], 3.0)      ->  True
    breaker_allows([(1.0, False), (2.0, False), (3.0, False)], 4.0)   ->  False
    breaker_allows([(1.0, False), (2.0, False), (3.0, False)], 70.0)  ->  True

    Считаются ПОДРЯД идущие провалы: один успех обнуляет счётчик, иначе
    инструмент, который падает раз в сто вызовов, однажды закроется
    навсегда.

    Время приходит аргументом, а не из time.time(): иначе тест на
    остывание пришлось бы писать через sleep.
    """
    consecutive = 0
    last_failure_at = None
    for at, ok in outcomes:
        if ok:
            consecutive = 0
        else:
            consecutive += 1
            last_failure_at = at
    if consecutive < threshold:
        return True
    return now - last_failure_at >= cooldown_s
