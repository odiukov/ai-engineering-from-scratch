"""
Дизайн схемы инструмента: линтер реестра

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l05-tool-schema-design
Разбор:  /check-code p13-l05-tool-schema-design
"""

import re

SEVERITIES = ("block", "warn", "nit")
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
ARGUMENT_IN_NAME = re.compile(r"_(in|for|at|by|of)_[a-z0-9]+$")
TENSE_MARKERS = ("_was_", "_will_", "_been_", "_yesterday", "_tomorrow", "_later")
INJECTION_PATTERNS = (
    r"<system>",
    r"ignore (previous|all|prior) (instructions|prompts)",
    r"bit\.ly|tinyurl",
    r"you must now",
    r"~/\.ssh",
)
MIN_DESCRIPTION = 40
MAX_DESCRIPTION = 1024
MAX_ACTION_VALUES = 3
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def lint_registry(registry):
    """Проверить весь реестр. Список претензий.

    Дополнительно к правилам одного инструмента — правило уровня набора:
      duplicate_name (block) — два инструмента с одним именем

    Дубликат имени не ловится проверкой одного инструмента в отрыве от
    остальных, а ломает всё: клиент вызовет по имени тот, что нашёлся
    первым, и это будет зависеть от порядка загрузки серверов.

    Претензия про дубликат выдаётся ОДИН раз на имя, а не по разу на копию.
    """
    raise NotImplementedError


def severity_summary(findings):
    """Сколько претензий какой тяжести. Все три ключа всегда на месте.

    severity_summary([])  ->  {"block": 0, "warn": 0, "nit": 0}

    Нули важны: строчка лога сборки не должна менять форму от того, повезло
    сегодня или нет, иначе её не распарсить.
    """
    raise NotImplementedError


def passes_ci(findings):
    """Пропускать ли сборку: блокирующих претензий нет.

    passes_ci([])                    ->  True
    passes_ci([<nit>, <warn>])       ->  True
    passes_ci([<block>])             ->  False

    warn и nit сборку не роняют намеренно. Линтер, который валит билд на
    каждой мелочи, отключают целиком — и вместе с мелочами перестают ловить
    инъекции в описаниях.
    """
    raise NotImplementedError
