"""
Дизайн схемы инструмента: линтер реестра — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Замеры Composio и Databricks дают 10–20 процентных пунктов точности выбора
инструмента только на переименовании и переписывании описаний. Здесь мы
собираем линтер, который проверяет правила урока и годится для CI.
Соответствие настоящей практике:

    is_snake_case      <-  правило именования, которое любят все токенизаторы
    lint_name          <-  имя стабильно, без аргументов и без времён
    lint_description   <-  паттерн «Use when X. Do not use for Y.» и длина
    lint_schema        <-  типизированные поля, enum на закрытых множествах
    lint_tool          <-  один инструмент целиком
    lint_registry      <-  весь реестр плюс проверки уровня набора
    severity_summary   <-  строчка итога для лога сборки
    passes_ci          <-  зелёная или красная сборка

Претензия (finding) — словарь одной формы:
    {"severity": "block" | "warn" | "nit", "path": ..., "rule": ..., "message": ...}
Тесты держатся за "rule" и "severity": текст message — для человека.
"""

import re

SEVERITIES = ("block", "warn", "nit")

SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

# Аргумент, зашитый в имя: get_weather_in_tokyo вместо get_weather(city).
ARGUMENT_IN_NAME = re.compile(r"_(in|for|at|by|of)_[a-z0-9]+$")

# Времена в имени: инструмент называет действие, а не когда оно случилось.
TENSE_MARKERS = ("_was_", "_will_", "_been_", "_yesterday", "_tomorrow", "_later")

# Непрямая инъекция в описании. Описание уходит в контекст модели дословно,
# поэтому враждебный сервер прячет команды именно здесь.
INJECTION_PATTERNS = (
    r"<system>",
    r"ignore (previous|all|prior) (instructions|prompts)",
    r"bit\.ly|tinyurl",
    r"you must now",
    r"~/\.ssh",
)

MIN_DESCRIPTION = 40
MAX_DESCRIPTION = 1024

# Больше трёх значений в аргументе action — инструмент пора разрезать.
MAX_ACTION_VALUES = 3

# Конструктор претензии. Готов, писать его не нужно — просто вызывай.
finding = lambda severity, path, rule, message: {
    "severity": severity,
    "path": path,
    "rule": rule,
    "message": message,
}


def is_snake_case(name):
    """Годится ли имя как snake_case: строчные, цифры, одиночные подчёркивания.

    is_snake_case("get_weather")   ->  True
    is_snake_case("notes_list_v2") ->  True
    is_snake_case("getWeather")    ->  False
    is_snake_case("_private")      ->  False
    is_snake_case("get__weather")  ->  False

    Почему это вообще правило: camelCase у части токенизаторов рвётся по
    границе слов на несколько токенов, и имя в промпте выглядит для модели
    иначе, чем в схеме.
    """
    return bool(SNAKE_CASE.match(name))


def lint_name(name):
    """Проверить имя инструмента. Список претензий.

    Правила:
      name_not_snake_case   (block)  — camelCase, дефисы, пустое имя
      name_embeds_argument  (warn)   — get_weather_in_tokyo вместо параметра
      name_has_tense_marker (warn)   — get_weather_tomorrow

    lint_name("get_weather")           ->  []
    lint_name("getWeather")            ->  [<block name_not_snake_case>]
    lint_name("get_weather_in_tokyo")  ->  [<warn name_embeds_argument>]

    Имя — часть контракта. Переименование ломает всех, кто уже вызывает
    инструмент, поэтому дешевле не ошибиться сразу.
    """
    problems = []
    if not is_snake_case(name):
        problems.append(finding("block", name, "name_not_snake_case", "name must be snake_case"))
    if ARGUMENT_IN_NAME.search(name):
        problems.append(
            finding("warn", name, "name_embeds_argument", "argument baked into the name")
        )
    if any(marker in name for marker in TENSE_MARKERS):
        problems.append(
            finding("warn", name, "name_has_tense_marker", "name carries a tense marker")
        )
    return problems


def lint_description(name, description):
    """Проверить описание инструмента. Список претензий.

    Правила:
      description_too_short      (block)  — короче 40 символов
      description_too_long       (block)  — длиннее 1024 символов
      description_missing_use_when   (warn)  — нет «Use when ...»
      description_missing_do_not_use (warn)  — нет «Do not use for ...»
      description_injection      (block)  — следы непрямой инъекции

    lint_description("get_weather",
        "Use when the user asks about current conditions. "
        "Do not use for forecasts.")  ->  []

    Фраза «Do not use for» — самая полезная строчка описания: она
    отграничивает инструмент от соседей по реестру, а именно на соседях
    модель и промахивается.

    Ловушка: описание уходит в контекст модели ДОСЛОВНО. Инъекция в нём —
    это block, а не косметика: враждебный MCP-сервер прячет команды здесь.
    """
    problems = []
    if len(description) < MIN_DESCRIPTION:
        problems.append(
            finding(
                "block",
                name,
                "description_too_short",
                f"description is {len(description)} chars, need {MIN_DESCRIPTION}",
            )
        )
    if len(description) > MAX_DESCRIPTION:
        problems.append(
            finding(
                "block",
                name,
                "description_too_long",
                f"description is {len(description)} chars, max {MAX_DESCRIPTION}",
            )
        )
    lowered = description.lower()
    if "use when" not in lowered:
        problems.append(
            finding("warn", name, "description_missing_use_when", "no 'Use when' sentence")
        )
    if "do not use" not in lowered:
        problems.append(
            finding(
                "warn",
                name,
                "description_missing_do_not_use",
                "no 'Do not use for' disambiguation",
            )
        )
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            problems.append(
                finding(
                    "block",
                    name,
                    "description_injection",
                    f"possible tool poisoning: {pattern!r}",
                )
            )
    return problems


def lint_schema(name, schema):
    """Проверить схему аргументов. Список претензий.

    Правила:
      schema_root_not_object   (block)  — корень схемы обязан быть object
      schema_missing_required  (warn)   — нет списка required
      field_untyped            (block)  — у поля нет type
      field_missing_description (nit)   — у поля нет description
      monolithic_action        (warn)   — action: str без enum или с enum > 3

    lint_schema("add", {"type": "object",
                        "properties": {"a": {"type": "integer",
                                             "description": "left operand"}},
                        "required": ["a"]})  ->  []

    Про monolithic_action: do_everything(action, target, options) выглядит
    сухо и DRY, а модель по нему промахивается на 15–30% чаще. Больше трёх
    значений action — режь на атомарные инструменты.

    Путь у претензий к полям — "имя_инструмента.имя_поля": по нему видно,
    какое именно поле чинить.
    """
    if schema.get("type") != "object":
        return [finding("block", name, "schema_root_not_object", "schema root must be object")]

    problems = []
    if "required" not in schema:
        problems.append(
            finding("warn", name, "schema_missing_required", "schema has no 'required' list")
        )
    for field, spec in schema.get("properties", {}).items():
        path = f"{name}.{field}"
        if "type" not in spec:
            problems.append(finding("block", path, "field_untyped", "field has no type"))
        if "description" not in spec:
            problems.append(
                finding("nit", path, "field_missing_description", "field has no description")
            )
        if field == "action" and spec.get("type") == "string":
            values = spec.get("enum", [])
            if len(values) > MAX_ACTION_VALUES or not values:
                problems.append(
                    finding(
                        "warn",
                        name,
                        "monolithic_action",
                        f"'action' string with {len(values)} enum values; split the tool",
                    )
                )
    return problems


def lint_tool(tool):
    """Проверить один инструмент целиком: имя, описание, схема.

    tool — словарь {"name", "description", "input_schema"}.
    Отсутствующие ключи не роняют линтер, а дают свои претензии: реестр
    приезжает из чужого сервера и может быть каким угодно.

    lint_tool(GOOD_TOOL)  ->  []

    Порядок претензий: сначала про имя, потом про описание, потом про схему.
    Так вывод линтера читается сверху вниз в том же порядке, в каком
    инструмент объявлен.
    """
    name = tool.get("name", "")
    return (
        lint_name(name)
        + lint_description(name, tool.get("description", ""))
        + lint_schema(name, tool.get("input_schema", {}))
    )


def lint_registry(registry):
    """Проверить весь реестр. Список претензий.

    Дополнительно к правилам одного инструмента — правило уровня набора:
      duplicate_name (block) — два инструмента с одним именем

    Дубликат имени не ловится проверкой одного инструмента в отрыве от
    остальных, а ломает всё: клиент вызовет по имени тот, что нашёлся
    первым, и это будет зависеть от порядка загрузки серверов.

    Претензия про дубликат выдаётся ОДИН раз на имя, а не по разу на копию.
    """
    problems = []
    seen = set()
    for tool in registry:
        name = tool.get("name", "")
        if name in seen:
            continue
        if sum(1 for t in registry if t.get("name", "") == name) > 1:
            problems.append(finding("block", name, "duplicate_name", "duplicate tool name"))
        seen.add(name)
    for tool in registry:
        problems.extend(lint_tool(tool))
    return problems


def severity_summary(findings):
    """Сколько претензий какой тяжести. Все три ключа всегда на месте.

    severity_summary([])  ->  {"block": 0, "warn": 0, "nit": 0}

    Нули важны: строчка лога сборки не должна менять форму от того, повезло
    сегодня или нет, иначе её не распарсить.
    """
    counts = {severity: 0 for severity in SEVERITIES}
    for item in findings:
        counts[item["severity"]] += 1
    return counts


def passes_ci(findings):
    """Пропускать ли сборку: блокирующих претензий нет.

    passes_ci([])                    ->  True
    passes_ci([<nit>, <warn>])       ->  True
    passes_ci([<block>])             ->  False

    warn и nit сборку не роняют намеренно. Линтер, который валит билд на
    каждой мелочи, отключают целиком — и вместе с мелочами перестают ловить
    инъекции в описаниях.
    """
    return all(item["severity"] != "block" for item in findings)
