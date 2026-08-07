"""
Продакшен-приложение на LLM — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Урок собирает вместе то, что в бою обычно прячется за FastAPI-мидлварями и
SDK провайдера: подсчёт цены запроса, экспоненциальный backoff с jitter,
цепочку запасных моделей, детерминированное A/B по хешу пользователя и
перцентили латентности. Здесь всё это пишется руками, без сети и без сна:
случайность приходит параметром rng, время — параметром, а «вызов модели» —
обычной функцией, которую передают снаружи.
"""

import hashlib
import math

# Прайс-лист: модель -> (цена за 1M входных токенов, за 1M выходных), USD.
# Совпадает с MODEL_PRICING из code/main.py урока.
MODEL_PRICING = {
    "claude-sonnet-5": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

# Цепочка деградации: сначала самая сильная модель, дальше — что дешевле и
# доступнее. Порядок важен, поэтому это кортеж, а не множество.
FALLBACK_CHAIN = ("claude-sonnet-5", "gpt-4o", "gpt-4o-mini")

# Последний рубеж, когда упала вся цепочка. Пользователь получает текст,
# а не стектрейс.
DEGRADED_TEXT = "Service temporarily unavailable. Please try again in a moment."


class ProviderError(Exception):
    """Провайдер не ответил: 429, 500 или таймаут.

    Свой класс, а не RuntimeError, специально: NotImplementedError — тоже
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """


def estimate_tokens(text):
    """Грубая оценка числа токенов в тексте: слов x 4/3, но не меньше единицы.

    estimate_tokens("hello world")        ->  2
    estimate_tokens("a b c d e f")        ->  8
    estimate_tokens("")                   ->  1

    Настоящий токенизатор считает по BPE-словарю и в стандартную библиотеку не
    входит. Для прикидки бюджета правило «3 слова ~ 4 токена» ошибается на
    10-20% и этого хватает, чтобы поймать запрос на 50 000 токенов до того,
    как он придёт в API.
    """
    # max(1, ...) нужен не для красоты: пустая строка иначе даст 0 токенов,
    # и деление на количество токенов в метриках рухнет.
    return max(1, len(text.split()) * 4 // 3)


def request_cost(model, input_tokens, output_tokens, pricing):
    """Цена одного запроса в долларах по прайс-листу pricing.

    request_cost("gpt-4o", 1500, 400, MODEL_PRICING)       ->  0.00775
    request_cost("gpt-4o-mini", 1500, 400, MODEL_PRICING)  ->  0.000465

    pricing — словарь {модель: (цена за 1M входных, цена за 1M выходных)}.

    Ловушка: неизвестную модель нельзя молча оценивать по цене какой-нибудь
    соседней. Так рождаются счета-сюрпризы. Неизвестная модель — ValueError.
    """
    if model not in pricing:
        known = ", ".join(sorted(pricing))
        raise ValueError(f"нет цены для модели {model!r} (известны: {known})")
    in_price, out_price = pricing[model]
    cost = input_tokens / 1_000_000 * in_price + output_tokens / 1_000_000 * out_price
    # округление до 8 знаков: дешёвые модели дают цену порядка 1e-7,
    # округление до центов превратило бы весь трафик в ноль
    return round(cost, 8)


def backoff_delay(attempt, base=1.0, cap=10.0, rng=None):
    """Пауза перед попыткой номер attempt: экспонента с потолком плюс jitter.

    backoff_delay(0)  ->  0.0   (первая попытка идёт сразу)
    backoff_delay(1)  ->  1.0
    backoff_delay(2)  ->  2.0
    backoff_delay(3)  ->  4.0
    backoff_delay(9)  ->  10.0  (упёрлось в cap)

    rng — объект random.Random. Если его передали, к паузе добавляется jitter
    из диапазона [0, задержка/2). Без rng функция строго детерминирована.

    Зачем jitter: без него тысяча клиентов, отвалившихся в одну секунду,
    ретраится тоже в одну секунду и добивает провайдера повторно. С jitter
    их ретраи размазываются по интервалу.

    Ловушка: jitter обязан браться из ПЕРЕДАННОГО rng, а не из глобального
    random. Иначе один и тот же seed даёт разные прогоны и тест не повторить.
    """
    if attempt <= 0:
        return 0.0
    delay = min(base * 2 ** (attempt - 1), cap)
    if rng is None:
        return delay
    return delay + rng.uniform(0.0, delay / 2)


def retry_with_backoff(call, max_retries=3, base=1.0, cap=10.0, rng=None):
    """Повторяет call(attempt), пока тот бросает ProviderError.

    Возвращает кортеж (результат, число попыток, суммарная пауза в секундах).
    Когда попытки кончились — бросает ProviderError.

    retry_with_backoff(lambda a: "ok")            ->  ("ok", 1, 0.0)
    retry_with_backoff(падает_один_раз)           ->  ("ok", 2, 1.0)

    call принимает номер попытки (0, 1, 2, ...) — это позволяет тесту
    сымитировать «первые две попытки 500, третья успех».

    Никакого time.sleep здесь нет: паузу мы СЧИТАЕМ и возвращаем. В бою на
    её месте стоял бы await asyncio.sleep(delay), но тест, который реально
    спит четыре секунды, никто не станет запускать.
    """
    total_delay = 0.0
    last = None
    for attempt in range(max_retries + 1):
        total_delay += backoff_delay(attempt, base, cap, rng)
        try:
            return call(attempt), attempt + 1, total_delay
        except ProviderError as exc:
            last = exc
    raise ProviderError(f"исчерпаны {max_retries + 1} попыток: {last}")


def call_with_fallback(models, call, max_retries=3, base=1.0, cap=10.0, rng=None):
    """Перебирает цепочку моделей, каждую — с ретраями. Никогда не бросает.

    Возвращает словарь с ключами:
      model         — модель, которая ответила, либо None
      text          — ответ либо DEGRADED_TEXT
      degraded      — True, если упала вся цепочка
      models_tried  — список моделей в порядке перебора

    call принимает (model, attempt) и либо возвращает текст, либо бросает
    ProviderError.

    Это graceful degradation из урока: падение вторичной системы не имеет
    права уронить основной поток. Пользователь всегда получает хоть что-то,
    пусть и от модели подешевле.
    """
    tried = []
    for model in models:
        tried.append(model)
        try:
            # замыкание по model через аргумент по умолчанию: без этого все
            # лямбды в цикле увидели бы последнее значение переменной
            text, _, _ = retry_with_backoff(
                lambda attempt, m=model: call(m, attempt), max_retries, base, cap, rng
            )
        except ProviderError:
            continue
        return {"model": model, "text": text, "degraded": False, "models_tried": tried}
    return {"model": None, "text": DEGRADED_TEXT, "degraded": True, "models_tried": tried}


def ab_bucket(user_id, experiment, traffic_pct):
    """Ветка A/B-эксперимента для пользователя: "variant" или "control".

    ab_bucket("user_001", "chat_v2", 10)     ->  "control"
    ab_bucket("user_001", "other_exp", 10)   ->  "variant"
    ab_bucket("bob", "chat_v2", 20)          ->  "variant"

    Бакет считается как md5("<user_id>:<experiment>") mod 100.

    Почему хеш, а не random: пользователь обязан видеть одну и ту же ветку на
    всех своих запросах, иначе метрики эксперимента бессмысленны, а интерфейс
    мигает. Имя эксперимента входит в хеш, чтобы человек, попавший в вариант
    одного теста, не оказывался в варианте всех остальных.
    """
    digest = hashlib.md5(f"{user_id}:{experiment}".encode()).hexdigest()
    return "variant" if int(digest, 16) % 100 < traffic_pct else "control"


def percentiles(values, ps):
    """Перцентили по методу ближайшего ранга. Возвращает {p: значение}.

    percentiles([1, 2, 3, 4], (50, 100))     ->  {50: 2, 100: 4}
    percentiles(list(range(1, 101)), (99,))  ->  {99: 99}

    Пустой список — ValueError: среднее по нулю запросов не бывает.
    p вне интервала (0, 100] — тоже ValueError.

    Зачем в проде именно перцентили, а не среднее: одна восьмисекундная
    хвостовая задержка растворяется в среднем, но именно она гонит
    пользователей прочь. P99 её видит, среднее — нет.
    """
    if not values:
        raise ValueError("percentiles: пустой список значений")
    # сортируем копию: журнал запросов принадлежит вызывающему коду,
    # молча его переупорядочивать нельзя
    ordered = sorted(values)
    out = {}
    for p in ps:
        if not 0 < p <= 100:
            raise ValueError(f"перцентиль вне (0, 100]: {p}")
        rank = math.ceil(p / 100 * len(ordered))
        out[p] = ordered[rank - 1]
    return out


def summarize_requests(logs, pricing):
    """Сводка по журналу запросов — то, что уходит на дашборд.

    logs — список словарей с ключами model, input_tokens, output_tokens,
    latency_ms, cache_hit (bool), error (строка или None).

    Возвращает словарь: requests, total_cost_usd, avg_cost_usd,
    cache_hit_rate_pct, error_rate_pct, p50_latency_ms, p99_latency_ms,
    cost_by_model.

    Попадание в кэш стоит ноль и в cost_by_model уходит нулём — строку из
    журнала при этом не выбрасываем, иначе hit rate посчитать будет не по чему.

    Пустой журнал — ValueError. «Ноль запросов, всё хорошо» — худший вид
    зелёного дашборда.
    """
    if not logs:
        raise ValueError("summarize_requests: пустой журнал")

    cost_by_model = {}
    total_cost = 0.0
    hits = 0
    errors = 0
    for entry in logs:
        if entry["cache_hit"]:
            hits += 1
            cost = 0.0
        else:
            cost = request_cost(
                entry["model"], entry["input_tokens"], entry["output_tokens"], pricing
            )
        if entry.get("error"):
            errors += 1
        total_cost += cost
        cost_by_model[entry["model"]] = cost_by_model.get(entry["model"], 0.0) + cost

    lat = percentiles([e["latency_ms"] for e in logs], (50, 99))
    n = len(logs)
    return {
        "requests": n,
        "total_cost_usd": round(total_cost, 8),
        "avg_cost_usd": round(total_cost / n, 8),
        "cache_hit_rate_pct": round(hits / n * 100, 2),
        "error_rate_pct": round(errors / n * 100, 2),
        "p50_latency_ms": lat[50],
        "p99_latency_ms": lat[99],
        "cost_by_model": {k: round(v, 8) for k, v in cost_by_model.items()},
    }
