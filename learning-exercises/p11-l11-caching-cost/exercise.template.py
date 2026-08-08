"""
Кэширование, rate limiting и оптимизация стоимости

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l11-caching-cost
Разбор:  /check-code p11-l11-caching-cost
"""

import hashlib
import json
import re

MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
}
SIMPLE_KEYWORDS = (
    "what time", "return policy", "hello", "thanks", "hours", "address",
    "phone", "price",
)
COMPLEX_KEYWORDS = (
    "analyze", "compare", "explain why", "write code", "debug", "architect",
    "design", "trade-off", "evaluate",
)
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
