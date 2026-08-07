"""
Слой маршрутизации LLM

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l21-llm-routing-layer
Разбор:  /check-code p13-l21-llm-routing-layer
"""

import hashlib
import json
import re

MODELS = {
    "openai/gpt-4o": {"input": 5.0, "output": 15.0, "quality": 0.92, "latency_ms": 900},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60, "quality": 0.74, "latency_ms": 350},
    "anthropic/claude-sonnet": {"input": 3.0, "output": 15.0, "quality": 0.94, "latency_ms": 1100},
    "anthropic/claude-haiku": {"input": 0.80, "output": 4.0, "quality": 0.78, "latency_ms": 300},
    "google/gemini-pro": {"input": 1.25, "output": 5.0, "quality": 0.86, "latency_ms": 700},
}
ROUTES = {
    "smart": ("openai/gpt-4o", "anthropic/claude-sonnet", "google/gemini-pro"),
    "fast": ("openai/gpt-4o-mini", "anthropic/claude-haiku"),
}
PII_PATTERNS = (
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b\d{16}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
)
REDACTED = "[REDACTED]"
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def redact_pii(text):
    """Вырезать персональные данные из текста. Вернуть (текст, кортеж меток).

    redact_pii("call me at 123-45-6789")
        ->  ("call me at [REDACTED]", ("ssn",))
    redact_pii("mail a@b.com or card 1234567890123456")
        ->  ("mail [REDACTED] or card [REDACTED]", ("credit_card", "email"))
    redact_pii("explain MCP")  ->  ("explain MCP", ())

    Метки идут в порядке PII_PATTERNS, а не в порядке появления в тексте:
    отчёт guardrail-а должен быть одинаков для одинакового набора находок.

    Пустой кортеж меток означает «ничего не нашли». Возвращать True/False
    мало: аудиту нужно знать, ЧТО именно вырезали.

    Это первый шаг маршрутизатора, до выбора провайдера. Отредактировать
    после отправки уже нечего.
    """
    raise NotImplementedError


def estimate_cost(model, input_tokens, output_tokens):
    """Стоимость одного вызова в долларах.

    estimate_cost("openai/gpt-4o", 1000, 1000)  ->  0.02
    estimate_cost("openai/gpt-4o", 0, 0)        ->  0.0

    Ловушка: цены в MODELS указаны за МИЛЛИОН токенов, как в прайсах
    провайдеров. Забыл поделить на 1e6 — и дашборд покажет миллион долларов
    за день, а кто-то успеет поверить.

    Неизвестная модель — ValueError: тихий ноль занизил бы отчёт по расходам
    ровно на ту модель, которую забыли внести в прайс-лист.
    """
    raise NotImplementedError


def cheapest_model(candidates, min_quality=0.0, max_latency_ms=None, tokens=(1000, 1000)):
    """Самая дешёвая модель из candidates, проходящая по качеству и латентности.

    Вернуть имя или None, если ни одна не проходит.

    cheapest_model(MODELS)                          ->  "openai/gpt-4o-mini"
    cheapest_model(MODELS, min_quality=0.80)        ->  "google/gemini-pro"
    cheapest_model(MODELS, min_quality=0.99)        ->  None

    Порядок проверок принципиален: сначала отсекаем по качеству и
    латентности, и только потом сравниваем цену. Наоборот получится
    «выбрали самую дешёвую и понадеялись, что сойдёт» — ровно та ошибка,
    из-за которой триаж уезжает на модель, которая его не тянет.

    Дешевизна считается на модельном объёме tokens, а не по цене за токен:
    у моделей разное соотношение input/output, и «дешевле по входу» не
    значит «дешевле по вызову».

    При равной цене выигрывает меньшее имя — ответ обязан не зависеть от
    порядка перебора candidates.
    """
    raise NotImplementedError


def cache_key(alias, messages):
    """Ключ кэша по алиасу и сообщениям. Одинаковый смысл — одинаковый ключ.

    cache_key("fast", [{"role": "user", "content": "Explain  MCP"}])
        == cache_key("fast", [{"role": "user", "content": "explain mcp"}])
    cache_key("fast", msgs) != cache_key("smart", msgs)

    Нормализация: схлопываем любые пробелы в один и опускаем регистр. В
    настоящем семантическом кэше вместо этого берут эмбеддинг промпта и
    ищут ближайший — идея та же, ключ грубее.

    Алиас входит в ключ: тот же вопрос к smart и к fast — разные ответы, и
    отдавать ответ дешёвой модели вместо умной нельзя.

    Ловушка: json.dumps без sort_keys вернёт разные строки для словарей с
    разным порядком ключей, и кэш будет промахиваться на ровном месте.
    """
    raise NotImplementedError


def resolve_chain(alias, routes=None):
    """Развернуть алиас в цепочку моделей по приоритету.

    resolve_chain("fast")  ->  ("openai/gpt-4o-mini", "anthropic/claude-haiku")
    resolve_chain("openai/gpt-4o")  ->  ("openai/gpt-4o",)
    resolve_chain("genius")  ->  ValueError

    Конкретное имя модели тоже принимается: клиент имеет право попросить
    ровно её, минуя алиас. Тогда цепочка из одного звена и фолбэка нет.

    Цепочка проверяется целиком: модель, которой нет в прайс-листе, — это
    ValueError сразу, а не сюрприз на третьем фолбэке в три часа ночи.
    """
    raise NotImplementedError


def route(alias, messages, provider, routes=None, cache=None):
    """Провести запрос через шлюз: редакция, кэш, фолбэк, учёт стоимости.

    provider — функция (model, messages) -> ответ вида
        {"status": 200, "usage": {"input_tokens": 10, "output_tokens": 20}, ...}
    Ответ со статусом из RETRY_STATUSES означает «попробуй следующего».

    Возвращается запись о вызове — всегда с одним и тем же набором ключей:
        alias, model, attempts, status, input_tokens, output_tokens,
        cost_usd, redacted, cached, cache_key, response, error

    route("smart", msgs, provider_с_упавшим_gpt4o)
        ->  attempts ["openai/gpt-4o", "anthropic/claude-sonnet"],
            model "anthropic/claude-sonnet", error None

    Три правила, каждое из которых стоило кому-то денег:

      * запрос не теряется. Пока в цепочке есть звенья, шлюз идёт дальше;
        error заполняется только когда кончились все.
      * 4xx (кроме 429) фолбэк НЕ вызывает. Кривой запрос будет кривым и у
        следующего провайдера — цепочка просто умножит счёт на три.
      * в провайдера уходит уже отредактированный текст. Кэш тоже считается
        по нему: иначе ключ зависел бы от того, что мы обещали не хранить.
    """
    raise NotImplementedError


def charge(ledger, team, cost_usd, cap_usd):
    """Списать расход команде, если она укладывается в лимит. True/False.

    ledger — словарь команда -> уже потрачено; функция его правит.

    charge({}, "search", 0.5, 1.0)              ->  True,  ledger {"search": 0.5}
    charge({"search": 0.9}, "search", 0.5, 1.0) ->  False, ledger не изменился

    Отказ обязан не оставлять следов: ни увеличенной суммы, ни новой записи
    о команде, которая ни разу не проехала. Иначе после отказа лимит съедет,
    и команда потеряет доступ навсегда.

    Сравнение с допуском 1e-12: суммы складываются из долей цента, и точное
    равенство на float ловится не всегда.
    """
    raise NotImplementedError


def spend_report(invocations):
    """Свести записи о вызовах в отчёт по моделям.

    spend_report([inv_gpt4o, inv_gpt4o, inv_haiku])
        ->  {"openai/gpt-4o": {"calls": 2, "cached": 0, "input_tokens": ...,
                               "output_tokens": ..., "cost_usd": ...},
             "anthropic/claude-haiku": {...}}

    Вызовы, где ни один провайдер не ответил (model is None), в отчёт не
    попадают: платить не за что, а строка "None" в дашборде только мешает.

    Попадания в кэш считаются отдельным счётчиком cached, но в calls входят:
    нагрузка на шлюз была, стоимости не было. Ровно эта разница и есть
    экономия, ради которой кэш ставили.
    """
    raise NotImplementedError
