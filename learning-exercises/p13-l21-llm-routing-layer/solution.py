"""
Слой маршрутизации LLM — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

LiteLLM, OpenRouter и Portkey дают всё это конфигом на десять строк. Здесь мы
пишем сам шлюз, чтобы стало видно, из чего он состоит. Соответствие
настоящему продукту:

    redact_pii      <-  guardrail на входе (Portkey PII redaction)
    estimate_cost   <-  cost tracking по прайс-листу шлюза
    cheapest_model  <-  cost-aware routing strategy
    cache_key       <-  ключ семантического кэша (у нас — нормализованный текст,
                        в проде — эмбеддинг промпта)
    resolve_chain   <-  model alias -> fallback chain из конфига
    route           <-  сам /v1/chat/completions прокси с ретраями
    charge          <-  per-key budget, он же per-team spend cap
    spend_report    <-  агрегация дашборда

Сети нет: провайдер приходит параметром — функция (model, messages) -> ответ
со статусом. Это позволяет проверить фолбэк, не устраивая настоящую аварию.
Времени тоже нет: латентность модели лежит в её описании, а не меряется
секундомером.
"""

import hashlib
import json
import re

# Прайс-лист и характеристики моделей. Цены — за МИЛЛИОН токенов, как в
# прайсах провайдеров. quality — доля пройденных eval-ов, latency_ms —
# медиана из наблюдений шлюза. Цифры вымышленные.
MODELS = {
    "openai/gpt-4o": {"input": 5.0, "output": 15.0, "quality": 0.92, "latency_ms": 900},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60, "quality": 0.74, "latency_ms": 350},
    "anthropic/claude-sonnet": {"input": 3.0, "output": 15.0, "quality": 0.94, "latency_ms": 1100},
    "anthropic/claude-haiku": {"input": 0.80, "output": 4.0, "quality": 0.78, "latency_ms": 300},
    "google/gemini-pro": {"input": 1.25, "output": 5.0, "quality": 0.86, "latency_ms": 700},
}

# Алиас в коде -> цепочка провайдеров в порядке приоритета.
ROUTES = {
    "smart": ("openai/gpt-4o", "anthropic/claude-sonnet", "google/gemini-pro"),
    "fast": ("openai/gpt-4o-mini", "anthropic/claude-haiku"),
}

# Порядок важен: SSN проверяется раньше карты, иначе шестнадцатизначный
# фрагмент карты мог бы перехватить часть другого совпадения.
PII_PATTERNS = (
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b\d{16}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
)

REDACTED = "[REDACTED]"

# Статусы, на которых имеет смысл идти к следующему провайдеру. 429 сюда
# входит: у соседнего провайдера свой лимит.
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
    tags = []
    for tag, pattern in PII_PATTERNS:
        text, found = pattern.subn(REDACTED, text)
        if found:
            tags.append(tag)
    return text, tuple(tags)


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
    spec = MODELS.get(model)
    if spec is None:
        raise ValueError(f"no price for model: {model}")
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be >= 0")
    return (input_tokens * spec["input"] + output_tokens * spec["output"]) / 1_000_000


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
    best_key = None
    best_name = None
    for name in candidates:
        spec = MODELS.get(name)
        if spec is None:
            raise ValueError(f"unknown model: {name}")
        if spec["quality"] < min_quality:
            continue
        if max_latency_ms is not None and spec["latency_ms"] > max_latency_ms:
            continue
        key = (estimate_cost(name, tokens[0], tokens[1]), name)
        if best_key is None or key < best_key:
            best_key, best_name = key, name
    return best_name


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
    normalized = [
        {"role": m["role"], "content": " ".join(m["content"].split()).lower()}
        for m in messages
    ]
    blob = json.dumps(
        {"alias": alias, "messages": normalized},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    table = ROUTES if routes is None else routes
    if alias in table:
        chain = tuple(table[alias])
    elif alias in MODELS:
        chain = (alias,)
    else:
        raise ValueError(f"unknown alias or model: {alias}")
    unknown = [m for m in chain if m not in MODELS]
    if unknown:
        raise ValueError(f"chain for {alias} names unpriced models: {', '.join(unknown)}")
    return chain


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
    invocation = {
        "alias": alias,
        "model": None,
        "attempts": [],
        "status": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "redacted": (),
        "cached": False,
        "cache_key": None,
        "response": None,
        "error": None,
    }
    clean = []
    tags = []
    for message in messages:
        text, found = redact_pii(message["content"])
        tags.extend(t for t in found if t not in tags)
        clean.append({"role": message["role"], "content": text})
    invocation["redacted"] = tuple(tags)

    key = cache_key(alias, clean)
    invocation["cache_key"] = key
    if cache is not None and key in cache:
        hit = cache[key]
        invocation["cached"] = True
        invocation["model"] = hit["model"]
        invocation["response"] = hit["response"]
        invocation["status"] = 200
        # attempts остаётся пустым: провайдера не трогали, платить не за что
        return invocation

    for model in resolve_chain(alias, routes):
        invocation["attempts"].append(model)
        response = provider(model, clean)
        status = response.get("status", 200)
        invocation["status"] = status
        if status == 200:
            usage = response.get("usage", {})
            invocation["model"] = model
            invocation["input_tokens"] = usage.get("input_tokens", 0)
            invocation["output_tokens"] = usage.get("output_tokens", 0)
            invocation["cost_usd"] = estimate_cost(
                model, invocation["input_tokens"], invocation["output_tokens"]
            )
            invocation["response"] = response
            if cache is not None:
                cache[key] = {"model": model, "response": response}
            return invocation
        if status not in RETRY_STATUSES:
            invocation["error"] = f"{model}: non-retryable status {status}"
            return invocation
    invocation["error"] = "all providers failed"
    return invocation


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
    if cost_usd < 0:
        raise ValueError("cost_usd must be >= 0")
    spent = ledger.get(team, 0.0)
    if spent + cost_usd > cap_usd + 1e-12:
        return False
    ledger[team] = spent + cost_usd
    return True


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
    report = {}
    for invocation in invocations:
        model = invocation.get("model")
        if model is None:
            continue
        row = report.setdefault(
            model,
            {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        row["calls"] += 1
        if invocation.get("cached"):
            row["cached"] += 1
        row["input_tokens"] += invocation.get("input_tokens", 0)
        row["output_tokens"] += invocation.get("output_tokens", 0)
        row["cost_usd"] += invocation.get("cost_usd", 0.0)
    return report
