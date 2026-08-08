"""
Кэширование, rate limiting и оптимизация стоимости — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib
import json
import re

# Цены в долларах за миллион токенов. Числа иллюстративные и устаревают —
# в проде их берут со страницы тарифов провайдера, а не хардкодят.
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
}

# Ключевые слова маршрутизатора. Порядок проверки: сначала "простое",
# потом "сложное" — как в уроке.
SIMPLE_KEYWORDS = (
    "what time", "return policy", "hello", "thanks", "hours", "address",
    "phone", "price",
)
COMPLEX_KEYWORDS = (
    "analyze", "compare", "explain why", "write code", "debug", "architect",
    "design", "trade-off", "evaluate",
)

# Токен запроса: слово с дефисами и апострофами внутри.
WORD_RE = re.compile(r"[a-z0-9'-]+")

ROUTING_TABLE = {
    "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
    "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
    "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    """Стоимость одного вызова в долларах. Вернуть разбивку по статьям.

    calculate_cost("gpt-4o", 1000, 500)
        ->  {'input_cost': 0.0025, 'cached_input_cost': 0.0,
             'output_cost': 0.005, 'total_cost': 0.0075, ...}
    calculate_cost("gpt-4o", 1000, 500, cached_input_tokens=800)
        ->  total_cost = 0.0005 + 0.001 + 0.005 = 0.0065

    cached_input_tokens — часть входа, попавшая в prompt cache провайдера.
    Она тарифицируется по ставке cached_input, остальное — по input.

    Ловушка: cached_input_tokens входит В input_tokens, а не добавляется к
    ним. Если сложить их, счёт вырастет ровно на кэш — то есть на то, что
    ты пытался сэкономить. Кэшированных токенов не может быть больше, чем
    входных: такое сочетание — ошибка вызывающего, а не ноль в отчёте.

    Неизвестная модель — тоже ошибка. Молча вернуть 0.0 значит спрятать
    незаметно растущую статью расходов.
    """
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")
    if cached_input_tokens > input_tokens:
        raise ValueError(
            f"cached_input_tokens ({cached_input_tokens}) > input_tokens ({input_tokens})"
        )

    pricing = MODEL_PRICING[model]
    fresh = input_tokens - cached_input_tokens
    input_cost = fresh / 1_000_000 * pricing["input"]
    cached_cost = cached_input_tokens / 1_000_000 * pricing["cached_input"]
    output_cost = output_tokens / 1_000_000 * pricing["output"]
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": input_cost,
        "cached_input_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + cached_cost + output_cost,
    }


def cache_key(model, messages, temperature):
    """Ключ кэша: sha256 от нормализованных модели, сообщений и температуры.

    m1 = [{"role": "user", "content": "What is the return policy?"}]
    m2 = [{"role": "USER", "content": "  what is   the RETURN POLICY?  "}]
    cache_key("gpt-4o", m1, 0.0) == cache_key("gpt-4o", m2, 0.0)   ->  True
    cache_key("gpt-4o", m1, 0.0) == cache_key("gpt-4o-mini", m1, 0.0) -> False

    Нормализация: регистр вниз, повторные пробелы и переводы строк схлопнуты
    в один, края обрезаны, температура округлена до 4 знаков.

    Модель и температура ОБЯЗАНЫ входить в ключ. Иначе ответ дешёвой модели
    отдастся вместо дорогой, а ответ на temperature=0 — вместо творческого.

    Отличие от prompt caching провайдера: тот совпадает по префиксу байт в
    байт, здесь совпадение нормализованное. Это уровень приложения, и
    решение "считать разный регистр одинаковым" принимаешь ты.
    """
    payload = {
        "model": model.strip().lower(),
        "temperature": round(float(temperature), 4),
        "messages": [
            {
                "role": str(m.get("role", "")).strip().lower(),
                "content": " ".join(str(m.get("content", "")).split()).lower(),
            }
            for m in messages
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cache_lookup(cache, model, messages, temperature=0.0, now=0.0):
    """Поискать ответ в кэше. Вернуть ответ или None, обновив счётчики.

    cache = {"entries": {}, "hits": 0, "misses": 0, "ttl": 60.0, "max_size": 8}
    cache_lookup(cache, "gpt-4o", msgs)   ->  None,  cache["misses"] == 1

    Кэш — обычный словарь вида
        {"entries": {ключ: {"response", "created", "hits"}},
         "hits": int, "misses": int, "ttl": float, "max_size": int}

    Три правила:
      * temperature > 0 — ответ недетерминированный, кэш пропускается совсем,
        и это промах, а не попадание;
      * запись старше ttl считается протухшей: промах и удаление записи;
      * попадание увеличивает и cache["hits"], и счётчик самой записи.

    now передаётся снаружи, а не берётся из time.time(): иначе тест на ttl
    приходится писать через sleep, и он становится медленным и хрупким.
    """
    if temperature > 0:
        cache["misses"] += 1
        return None

    key = cache_key(model, messages, temperature)
    entry = cache["entries"].get(key)
    if entry is not None:
        if now - entry["created"] < cache["ttl"]:
            cache["hits"] += 1
            entry["hits"] += 1
            return entry["response"]
        del cache["entries"][key]  # протухла — держать её незачем

    cache["misses"] += 1
    return None


def cache_store(cache, model, messages, temperature, response, now=0.0):
    """Положить ответ в кэш. Вернуть тот же cache.

    Ничего не кладётся при temperature > 0: сохранённый ответ выдавался бы
    вместо нового сэмпла, и вся недетерминированность, за которую заплачено,
    пропадала бы.

    При вставке нового ключа в заполненный кэш вытесняется самая старая по
    created. Перезапись существующего ключа не увеличивает кэш и ничего другого
    вытеснять не должна. Ничьи разруливаются по ключу, чтобы вытеснение не зависело
    от порядка обхода словаря.
    """
    if temperature > 0:
        return cache

    key = cache_key(model, messages, temperature)
    if key not in cache["entries"] and len(cache["entries"]) >= cache["max_size"]:
        oldest = min(cache["entries"], key=lambda k: (cache["entries"][k]["created"], k))
        del cache["entries"][oldest]

    cache["entries"][key] = {"response": response, "created": now, "hits": 0}
    return cache


def route_model(query, tier="pro"):
    """Выбрать самую дешёвую модель, которая справится с запросом.

    route_model("Hello")                            ->  complexity 'simple'
    route_model("Analyze the trade-offs of Kafka")  ->  complexity 'complex'
    route_model("Summarize this quarterly report for the board")
                                                    ->  complexity 'medium'

    Вернуть {"query", "complexity", "model", "tier"}.

    Правило из урока: запрос из пяти слов и меньше ИЛИ содержащий простое
    ключевое слово — simple; иначе сложное ключевое слово — complex;
    иначе medium. Проверка на simple идёт первой.

    Ловушка: искать однословное ключевое слово простым `kw in query` нельзя.
    "no" находится внутри "monoliths" и "know", "hi" — внутри "this". Запрос
    "Analyze the trade-offs between microservices and monoliths" при таком
    поиске уезжает в самую дешёвую модель. Однословные ключи сверяй с
    ТОКЕНАМИ запроса (по началу слова, чтобы "trade-offs" совпало с
    "trade-off"), а по всей строке ищи только многословные фразы.

    Честное ограничение, которое остаётся и после этого: запрос
    "Explain why the price of GPUs..." попадёт в simple из-за слова "price",
    хотя это разбор. Настоящий роутер строят на эмбеддингах или маленькой
    обученной модели — но и такой уже экономит 40-70% счёта.
    """
    q = query.lower()
    tokens = WORD_RE.findall(q)

    def has(keywords):
        for kw in keywords:
            if " " in kw:
                if kw in q:
                    return True
            elif any(t.startswith(kw) for t in tokens):
                return True
        return False

    if len(q.split()) <= 5 or has(SIMPLE_KEYWORDS):
        complexity = "simple"
    elif has(COMPLEX_KEYWORDS):
        complexity = "complex"
    else:
        complexity = "medium"

    row = ROUTING_TABLE[complexity]
    return {
        "query": query,
        "complexity": complexity,
        "model": row.get(tier, row["free"]),
        "tier": tier,
    }


def token_bucket_take(bucket, tokens_needed, now):
    """Списать токены из ведра. Вернуть {"allowed", "tokens_available", "retry_after"}.

    bucket = {"tokens": 100.0, "capacity": 100.0, "refill_rate": 10.0, "last_refill": 0.0}
    token_bucket_take(bucket, 60, now=0.0)   ->  allowed True,  осталось 40
    token_bucket_take(bucket, 60, now=0.0)   ->  allowed False, retry_after 2.0
    token_bucket_take(bucket, 60, now=2.0)   ->  allowed True   (ведро долилось)

    Ведро доливается непрерывно: elapsed * refill_rate, но не выше capacity.
    Отсюда и смысл алгоритма — всплеск на весь объём ведра разрешён, а вот
    средняя скорость всё равно ограничена refill_rate.

    При отказе НИЧЕГО не списывается, а retry_after показывает, через
    сколько секунд накопится недостающее. Пустой ответ "нельзя" без времени
    ожидания заставляет клиента долбиться в цикле.

    Ведро меняется на месте: это состояние пользователя, оно живёт между
    запросами.
    """
    elapsed = max(0.0, now - bucket["last_refill"])
    bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + elapsed * bucket["refill_rate"])
    bucket["last_refill"] = now

    if bucket["tokens"] < tokens_needed:
        deficit = tokens_needed - bucket["tokens"]
        return {
            "allowed": False,
            "tokens_available": bucket["tokens"],
            "retry_after": deficit / bucket["refill_rate"] if bucket["refill_rate"] else float("inf"),
        }

    bucket["tokens"] -= tokens_needed
    return {"allowed": True, "tokens_available": bucket["tokens"], "retry_after": 0.0}


def serve_query(cache, query, tier="pro", now=0.0):
    """Полный путь запроса: кэш -> маршрутизация -> вызов -> цена. Вернуть лог.

    Ключи лога: query, model, complexity, response, cache_status,
    input_tokens, output_tokens, cost, saved_cost.

    Промах: запрос уходит в модель, cost — реальная цена, saved_cost = 0.0.
    Попадание: ответ берётся из кэша, cost = 0.0, а saved_cost равен той
    цене, которую вызов стоил бы. Ответ при этом ТОТ ЖЕ САМЫЙ — в этом весь
    смысл кэша: меняется цена, а не ответ.

    Токены считаются по формуле-заглушке из урока:
        input  = слов * 4 + 500   (системный промпт и контекст)
        output = 150 + слов * 2
    Настоящий счётчик — это токенайзер модели; здесь его нет, и подменять
    его чем-то умным смысла тоже нет.
    """
    messages = [{"role": "user", "content": query}]
    route = route_model(query, tier)
    model = route["model"]

    words = len(query.split())
    input_tokens = words * 4 + 500
    output_tokens = 150 + words * 2
    full_cost = calculate_cost(model, input_tokens, output_tokens)["total_cost"]

    cached = cache_lookup(cache, model, messages, 0.0, now)
    if cached is not None:
        return {
            "query": query,
            "model": model,
            "complexity": route["complexity"],
            "response": cached,
            "cache_status": "hit",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "saved_cost": full_cost,
        }

    response = f"[{model}] {query}"  # заглушка вместо вызова API
    cache_store(cache, model, messages, 0.0, response, now)
    return {
        "query": query,
        "model": model,
        "complexity": route["complexity"],
        "response": response,
        "cache_status": "miss",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": full_cost,
        "saved_cost": 0.0,
    }


def summarize_usage(logs):
    """Свести логи вызовов в отчёт: сколько потрачено и сколько сэкономлено.

    Вернуть {"calls", "total_cost", "saved_cost", "cache_hits", "hit_rate",
             "avg_cost_per_call", "cost_by_model"}.

    summarize_usage([])  ->  всё нули, без деления на ноль.

    Полезное тождество: total_cost + saved_cost — это то, во что обошёлся бы
    тот же трафик без кэша. Оно и есть ответ на вопрос "а кэш вообще
    окупился".

    cost_by_model разбивает расходы по моделям: {модель: {"calls", "cost"}}.
    Без этой разбивки не видно, что 8% запросов, ушедших в Opus, съедают
    половину счёта.
    """
    calls = len(logs)
    if calls == 0:
        return {
            "calls": 0,
            "total_cost": 0.0,
            "saved_cost": 0.0,
            "cache_hits": 0,
            "hit_rate": 0.0,
            "avg_cost_per_call": 0.0,
            "cost_by_model": {},
        }

    total_cost = sum(log["cost"] for log in logs)
    saved_cost = sum(log["saved_cost"] for log in logs)
    hits = sum(1 for log in logs if log["cache_status"] == "hit")

    by_model = {}
    for log in logs:
        row = by_model.setdefault(log["model"], {"calls": 0, "cost": 0.0})
        row["calls"] += 1
        row["cost"] += log["cost"]

    return {
        "calls": calls,
        "total_cost": total_cost,
        "saved_cost": saved_cost,
        "cache_hits": hits,
        "hit_rate": hits / calls,
        "avg_cost_per_call": total_cost / calls,
        "cost_by_model": by_model,
    }
