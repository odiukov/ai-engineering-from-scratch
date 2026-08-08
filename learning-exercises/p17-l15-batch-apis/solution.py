"""
Batch API: очередь, скидка и SLA завершения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Чему это соответствует в настоящих API:

    sync_cost, cached_cost  <-  обычный вызов /v1/messages с cache_control
    batch_cost              <-  OpenAI /v1/batches, Anthropic Message Batches,
                                Vertex Batch Prediction: скидка на batch и
                                отдельная политика совмещения с cache
    submit                  <-  загрузка JSONL и создание batch job
    drain_window            <-  планировщик провайдера: задания считаются в
                                окно недозагрузки GPU, а не сразу
    sla_report              <-  ответ на вопрос «а если не успеет к утру»
    triage                  <-  решение interactive / semi / batch по SLA
    lane_decision           <-  то, ради чего урок: сколько денег лежит на
                                столе, пока workload считается «real-time»

Ни сети, ни провайдера: batch API — это очередь со скидкой, и очередь
моделируется честно. Часы приходят параметрами (submitted_h, границы окна),
внутри нет ни time.time(), ни sleep.

Цены — снимок рейт-карты Sonnet-класса на 2026-04, они дрейфуют.
"""

import math

# $/M токенов.
PRICE_INPUT = 3.00
PRICE_CACHED_READ = 0.30
PRICE_OUTPUT = 15.00

# Премия за запись префикса в кэш (TTL 5 минут).
WRITE_PREMIUM = 1.25

# Скидка batch: множитель к счёту, одинаковый у всех провайдеров.
BATCH_DISCOUNT = 0.50

# Политика задаётся явно: Anthropic складывает batch и cache, а у Vertex
# Gemini цена cached prefix имеет приоритет и повторно на 50% не режется.
BATCH_CACHE_POLICY = {
    "anthropic": "stack",
    "vertex-gemini": "cache_precedence",
}

# Границы полос по бюджету задержки, в секундах.
INTERACTIVE_MAX_S = 60
SEMI_MAX_S = 3600

# Обещание провайдера по batch, в часах.
BATCH_SLA_H = 24


class BatchError(Exception):
    """Очередь или тариф спрошены о невозможном.

    Свой класс, а не ValueError и не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """


def sync_cost(n, prefix_tokens, unique_tokens, output_tokens):
    """Счёт за n синхронных вызовов без всякого кэша — базовая линия.

    sync_cost(1, 4000, 2000, 200)   ->  0.021
    sync_cost(50_000, 4000, 2000, 200)  ->  1050.0

    Разбор: (4000+2000)/1e6 * 3.00 = 0.018 за вход, 200/1e6 * 15.00 = 0.003
    за выход. Общий системный промпт здесь оплачивается заново каждый раз —
    в этом и смысл базовой линии.

    Ловушка: n < 0 — BatchError. Ноль допустим, это пустой прогон.
    """
    if n < 0:
        raise BatchError(f"n must be non-negative, got {n}")
    per_call = (prefix_tokens + unique_tokens) / 1e6 * PRICE_INPUT
    per_call += output_tokens / 1e6 * PRICE_OUTPUT
    return n * per_call


def cached_cost(n, prefix_tokens, unique_tokens, output_tokens):
    """То же, но общий префикс кэшируется: первый пишет, остальные читают.

    cached_cost(50_000, 4000, 2000, 200)  ->  примерно 510.01

    Первый вызов платит за префикс с премией 1.25x вместо базовой цены,
    остальные n-1 — цену чтения, в десять раз дешевле входа. Уникальная
    часть промпта и выход не кэшируются никогда.

    Ловушка: премия платится ВМЕСТО базовой цены, а не вдобавок к ней. Если
    сложить обе, первый вызов подорожает вдвое и счёт разойдётся с рейт-картой.
    """
    if n < 0:
        raise BatchError(f"n must be non-negative, got {n}")
    if n == 0:
        return 0.0
    tail = (unique_tokens / 1e6 * PRICE_INPUT) + (output_tokens / 1e6 * PRICE_OUTPUT)
    write = prefix_tokens / 1e6 * PRICE_INPUT * WRITE_PREMIUM
    reads = (n - 1) * prefix_tokens / 1e6 * PRICE_CACHED_READ
    return write + reads + n * tail


def batch_cost(n, prefix_tokens, unique_tokens, output_tokens, cached,
               provider="anthropic"):
    """Счёт по batch с явной политикой провайдера для prompt cache.

    batch_cost(50_000, 4000, 2000, 200, False)  ->  525.0
    batch_cost(50_000, 4000, 2000, 200, True)   ->  примерно 255.0

    У Anthropic скидки складываются. У Vertex Gemini cache price takes
    precedence: cached prefix оплачивается по cache-тарифу без дополнительной
    batch-скидки, а уникальный вход и выход всё ещё получают -50%.

    Неизвестный provider — BatchError: молча выбрать финансовую политику
    нельзя.
    """
    if provider not in BATCH_CACHE_POLICY:
        raise BatchError(f"unknown provider policy: {provider!r}")
    if not cached:
        return sync_cost(n, prefix_tokens, unique_tokens, output_tokens) * BATCH_DISCOUNT
    cached_total = cached_cost(n, prefix_tokens, unique_tokens, output_tokens)
    if BATCH_CACHE_POLICY[provider] == "stack":
        return cached_total * BATCH_DISCOUNT
    cached_prefix = cached_cost(n, prefix_tokens, 0, 0)
    uncached_tail = sync_cost(n, 0, unique_tokens, output_tokens)
    return cached_prefix + uncached_tail * BATCH_DISCOUNT


def submit(queue, job_id, n_requests, submitted_h):
    """Положить задание в очередь. Возвращает НОВЫЙ список, вход не портит.

    submit([], "night-1", 20_000, 1.0)  ->  [{'job_id': 'night-1', ...}]

    Ловушка: очередь-аргумент менять нельзя. Тест «вход не изменился» есть,
    и он не про чистоту стиля: очередь дня обычно строят один раз и потом
    гоняют по ней несколько сценариев планирования.

    BatchError на дубль job_id, на n_requests <= 0 и на отрицательное время.
    """
    if any(job["job_id"] == job_id for job in queue):
        raise BatchError(f"duplicate job_id {job_id!r}")
    if n_requests <= 0:
        raise BatchError(f"job {job_id!r}: n_requests must be positive")
    if submitted_h < 0:
        raise BatchError(f"job {job_id!r}: submitted_h must be non-negative")
    return list(queue) + [{
        "job_id": job_id,
        "n_requests": n_requests,
        "submitted_h": float(submitted_h),
    }]


def drain_window(queue, window_start_h, window_end_h, throughput_per_h):
    """Прогнать очередь через ежедневное окно недозагрузки.

    Окно повторяется каждые сутки: [window_start_h, window_end_h) часов от
    начала суток. Задания берутся по (submitted_h, job_id), одно за другим —
    провайдер не начнёт следующее, пока не закончил текущее.

    Возвращает список {"job_id", "finished_h", "wait_h"} в порядке запуска.

    Окно 0-6, пропускная 10000 запросов в час:
      задание на 20000 запросов, подано в 1.0  ->  finished_h 3.0,  wait_h 2.0
      задание на 10000 запросов, подано в 10.0 ->  finished_h 25.0, wait_h 15.0
      задание на 100000 запросов, подано в 1.0 ->  finished_h 29.0, wait_h 28.0

    Третий случай — про то, почему SLA это 24 часа, а не «2-6 часов как
    обычно»: работа не влезла в одно окно и доехала только следующей ночью.

    Ловушка: окно закрылось посреди задания — работу надо ПРОДОЛЖИТЬ в
    следующем окне, а не начать заново и не досчитать за границей окна.
    """
    if not 0 <= window_start_h < window_end_h <= 24:
        raise BatchError(f"bad window [{window_start_h}, {window_end_h})")
    if throughput_per_h <= 0:
        raise BatchError(f"throughput must be positive, got {throughput_per_h}")

    completions = []
    clock = 0.0
    for job in sorted(queue, key=lambda j: (j["submitted_h"], j["job_id"])):
        t = max(clock, job["submitted_h"])
        remaining_h = job["n_requests"] / throughput_per_h
        # EPS страхует от вечного цикла на хвосте из-за двоичных дробей
        while remaining_h > 1e-12:
            day = math.floor(t / 24.0) * 24.0
            opens, closes = day + window_start_h, day + window_end_h
            if t < opens:
                t = opens
            elif t >= closes:
                t = opens + 24.0
                continue
            step = min(closes - t, remaining_h)
            t += step
            remaining_h -= step
            if remaining_h > 1e-12:
                t = opens + 24.0  # окно закрылось, ждём следующего
        completions.append({
            "job_id": job["job_id"],
            "finished_h": t,
            "wait_h": t - job["submitted_h"],
        })
        clock = t
    return completions


def sla_report(completions, sla_h=BATCH_SLA_H):
    """Уложились ли задания в обещание провайдера.

    Возвращает dict:
      total, met, missed,
      met_fraction  — доля уложившихся, 0.0 на пустом входе,
      max_wait_h    — худшее ожидание, 0.0 на пустом входе,
      worst_job     — id худшего задания или None.

    sla_report([])["met_fraction"]  ->  0.0

    Ловушка на границе: ожидание ровно 24.0 часа — это УЛОЖИЛИСЬ. SLA
    формулируется как «в течение 24 часов», сравнение нестрогое.
    """
    total = len(completions)
    if total == 0:
        return {"total": 0, "met": 0, "missed": 0, "met_fraction": 0.0,
                "max_wait_h": 0.0, "worst_job": None}
    met = sum(1 for c in completions if c["wait_h"] <= sla_h)
    worst = max(completions, key=lambda c: (c["wait_h"], c["job_id"]))
    return {
        "total": total,
        "met": met,
        "missed": total - met,
        "met_fraction": met / total,
        "max_wait_h": worst["wait_h"],
        "worst_job": worst["job_id"],
    }


def triage(latency_budget_s):
    """Полоса по бюджету задержки: 'interactive' | 'semi' | 'batch'.

    triage(5)       ->  'interactive'   (пользователь смотрит на спиннер)
    triage(600)     ->  'semi'          (вернётся через несколько минут)
    triage(86_400)  ->  'batch'         (нужно к утру)

    Границы нестрогие: ровно 60 с — ещё interactive, ровно 3600 с — ещё semi.

    Ловушка не в коде, а в голове: «продакшен» — это не спецификация
    задержки. Спецификация — это SLA. Половина синхронных пайплайнов
    считаются интерактивными только потому, что их так назвали.

    BatchError на бюджет <= 0.
    """
    if latency_budget_s <= 0:
        raise BatchError(f"latency budget must be positive, got {latency_budget_s}")
    if latency_budget_s <= INTERACTIVE_MAX_S:
        return "interactive"
    if latency_budget_s <= SEMI_MAX_S:
        return "semi"
    return "batch"


def lane_decision(n, prefix_tokens, unique_tokens, output_tokens, latency_budget_s,
                  provider="anthropic"):
    """Выбрать полосу и посчитать, сколько это стоит и сколько потеряно.

    Возвращает dict:
      lane            — из triage,
      cost            — счёт в выбранной полосе,
      baseline_cost   — синхронно и без кэша,
      best_cost       — batch + кэш по политике provider, недостижимый минимум,
      saving_usd/pct  — экономия против baseline,
      forgone_usd     — сколько оставлено на столе из-за требования к задержке.

    Кэш доступен во всех полосах, скидка batch — только в 'batch'. У Vertex
    Gemini cache-тариф на общий префикс имеет приоритет над batch-скидкой.

    lane_decision(50_000, 4000, 200, 100, 5)["lane"]       ->  'interactive'
    lane_decision(50_000, 4000, 200, 100, 86_400)["lane"]  ->  'batch'

    Главное свойство: forgone_usd > 0 у interactive и semi и ровно 0 у batch.
    Batch выгоден ровно тогда, когда задержка допустима — не «почти всегда»,
    а именно тогда.
    """
    lane = triage(latency_budget_s)
    baseline = sync_cost(n, prefix_tokens, unique_tokens, output_tokens)
    best = batch_cost(n, prefix_tokens, unique_tokens, output_tokens, True, provider)
    if lane == "batch":
        cost = best
    else:
        cost = cached_cost(n, prefix_tokens, unique_tokens, output_tokens)
    return {
        "lane": lane,
        "cost": cost,
        "baseline_cost": baseline,
        "best_cost": best,
        "saving_usd": baseline - cost,
        "saving_pct": (baseline - cost) / baseline * 100 if baseline else 0.0,
        "forgone_usd": cost - best,
    }
