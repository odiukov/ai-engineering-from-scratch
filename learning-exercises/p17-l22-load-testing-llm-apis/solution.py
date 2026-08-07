"""
Нагрузочное тестирование LLM API — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

LLMPerf, GenAI-Perf, k6 и LLM-Locust делают это за тебя — и делают
по-разному, отчего на одном и том же сервере показывают разные числа.
Здесь мы собираем нагрузочный стенд руками. Соответствие настоящему
инструменту:

    percentile        <-  P50/P95/P99 в отчёте любого харнесса
    prompt_lengths    <-  --mean-input-tokens / --stddev-input-tokens в LLMPerf
    make_workload     <-  сам генератор трафика: длины И разные префиксы
    arrival_schedule  <-  сценарии k6: steady, ramp, spike, soak
    run_load          <-  очередь на стороне сервера: concurrency + queue limit
    summarize         <-  сводка прогона с перцентилями и долей отказов
    apparent_itl      <-  GIL trap: что Locust показывает вместо правды
    ci_gate           <-  thresholds в k6, ломающие сборку на PR

Времени нет: расписание прибытия — список меток, обслуживание считается
арифметикой. Ни одного sleep. Настоящий прогон на 500 запросов занял бы
десять минут, этот — миллисекунды, и потому его можно держать в тестах.
"""

import math

# Модель сервера. Числа из урока (docs/en.md, code/main.py): попадание в
# prefix cache даёт TTFT около 80 мс, промах — около 800 мс, каждый
# следующий токен — около 15 мс.
TTFT_CACHE_HIT_MS = 80.0
TTFT_CACHE_MISS_MS = 800.0
TPOT_MS = 15.0

# Четыре сценария нагрузки из урока.
PATTERNS = ("steady", "ramp", "spike", "soak")


def percentile(samples, q):
    """Перцентиль по методу «ближайший ранг». q — доля в [0, 1].

    percentile([1, 2, 3, 4], 0.5)   ->  2
    percentile([1, 2, 3, 4], 1.0)   ->  4
    percentile([1, 2, 3, 4], 0.0)   ->  1

    Ранг = ceil(q * n), нижняя граница 1. Никакой интерполяции: P99 обязан
    быть НАБЛЮДЁННЫМ значением, а не средним между двумя соседними. Иначе в
    отчёте появляется латентность, которой ни один запрос не показал.

    Ловушка: на ста запросах P99 — это ровно один худший запрос. На
    пятидесяти он не существует вовсе, и харнесс всё равно что-то напечатает.
    Отсюда правило урока про 100-1000 запросов за прогон.

    Пустой список — ValueError: перцентиль ничего не определён, а вернуть
    0.0 значит нарисовать в отчёте идеальную латентность на пустом прогоне.
    """
    if not samples:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(samples)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def prompt_lengths(n, mean_tokens, stddev_tokens, rng, minimum=1):
    """n длин промптов из нормального распределения. Вернуть кортеж целых.

    prompt_lengths(3, 500, 0, None)  ->  (500, 500, 500)
    prompt_lengths(1000, 500, 150, random.Random(0))
        ->  длины около 500 с разбросом около 150

    stddev_tokens == 0 — вырожденный случай: rng не нужен вовсе, все длины
    одинаковы. Это и есть «loop with one prompt», против которого написан
    весь урок.

    Обрезка снизу по minimum: нормальное распределение спокойно выдаёт
    отрицательную длину при mean=500, stddev=300, а запрос из минус двухсот
    токенов отправить некуда.

    В LLMPerf это ровно --mean-input-tokens и --stddev-input-tokens.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if stddev_tokens == 0:
        return tuple(max(minimum, int(mean_tokens)) for _ in range(n))
    if rng is None:
        raise ValueError("rng is required when stddev_tokens != 0")
    return tuple(max(minimum, int(rng.gauss(mean_tokens, stddev_tokens))) for _ in range(n))


def make_workload(n, mean_tokens, stddev_tokens, distinct_prefixes, rng, output_tokens=100):
    """Собрать поток запросов: длины из распределения, префиксы по кругу.

    Вернуть кортеж словарей {"prompt_tokens", "output_tokens", "prefix"}.

    make_workload(4, 2000, 0, 1, None)
        ->  четыре одинаковых запроса с prefix "prefix-0" — тот самый
            uniform-прогон, который врёт
    make_workload(500, 500, 150, 80, random.Random(0))
        ->  реалистичный поток: разные длины, 80 разных префиксов

    Префиксы раздаются по кругу (i % distinct_prefixes), а не случайно:
    доля попаданий в кэш должна зависеть от ОДНОГО параметра, иначе тест на
    prompt-uniformity trap превращается в лотерею.

    distinct_prefixes = 1 — это «loop with one prompt». Сервер видит один и
    тот же префикс, prefix cache попадает почти всегда, пропускная
    способность выглядит великолепно, и до прода это не доживает.
    """
    if distinct_prefixes < 1:
        raise ValueError(f"distinct_prefixes must be >= 1, got {distinct_prefixes}")
    lengths = prompt_lengths(n, mean_tokens, stddev_tokens, rng)
    return tuple(
        {
            "prompt_tokens": length,
            "output_tokens": output_tokens,
            "prefix": f"prefix-{i % distinct_prefixes}",
        }
        for i, length in enumerate(lengths)
    )


def arrival_schedule(pattern, duration_s, base_rps, peak_rps=None):
    """Метки прибытия запросов для одного из четырёх сценариев урока.

    Вернуть кортеж моментов времени в секундах, по возрастанию.

    arrival_schedule("steady", 4.0, 2.0)          ->  (0.0, 0.5, 1.0, ... , 3.5)
    arrival_schedule("ramp", 10.0, 0.0, 10.0)     ->  интервалы сжимаются
    arrival_schedule("spike", 10.0, 10.0, 100.0)  ->  плотная середина
    arrival_schedule("soak", 3600.0, 1.0)         ->  ровный час

    Как считается: время делится на секундные отрезки, для каждого берётся
    мгновенный темп, и внутри отрезка запросы ставятся равномерно.

      steady/soak — темп base_rps всё время. Разница только в длительности:
                    soak идёт часами и ловит утечки памяти, дрейф пулов
                    соединений и переполнение телеметрии;
      ramp        — темп линейно растёт от base_rps до peak_rps: ищем точку
                    излома по ёмкости;
      spike       — base_rps, кроме средней пятой части времени, где
                    peak_rps: проверяем, успеет ли автоскейлер.

    Никакой случайности: пуассоновский поток был бы реалистичнее, но тогда
    тест «на этом темпе очередь не растёт» стал бы вероятностным.

    Неизвестный pattern — ValueError. Опечатка в имени сценария не должна
    молча превращаться в steady, иначе спайк-тест «пройдёт» не запустившись.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern}, expected one of {PATTERNS}")
    if pattern in ("ramp", "spike") and peak_rps is None:
        raise ValueError(f"pattern {pattern} needs peak_rps")
    times = []
    for second in range(int(math.ceil(duration_s))):
        if pattern in ("steady", "soak"):
            rate = base_rps
        elif pattern == "ramp":
            # доля пройденного пути по середине секунды: линейный рост
            progress = (second + 0.5) / duration_s
            rate = base_rps + (peak_rps - base_rps) * progress
        else:  # spike
            in_spike = 0.4 * duration_s <= second < 0.6 * duration_s
            rate = peak_rps if in_spike else base_rps
        count = int(round(rate))
        for k in range(count):
            moment = second + k / count
            if moment < duration_s:
                times.append(moment)
    return tuple(times)


def run_load(workload, arrivals, concurrency, queue_limit, cached_prefixes=None):
    """Прогнать нагрузку через модель очереди. Вернуть кортеж записей.

    Модель: concurrency одинаковых слотов обслуживания и общая очередь
    длиной queue_limit. Запрос, пришедший при полной очереди, получает
    отказ (в проде это 429 или 503) — он и есть та самая доля отказов,
    которую нагрузочный тест обязан считать.

    Запись: {"arrival", "start", "ttft_ms", "total_ms", "wait_ms",
             "cache_hit", "rejected"}.

    run_load(лёгкий_поток, редкие_прибытия, 4, 10)  ->  wait_ms нули, отказов нет
    run_load(тот_же_поток, частые_прибытия, 1, 0)   ->  очередь переполняется

    Время обслуживания: TTFT (по попаданию в prefix cache) плюс
    output_tokens * TPOT_MS. Первый запрос с данным префиксом всегда
    промахивается и сам же кладёт префикс в кэш — отсюда и берётся эффект
    uniform-прогона: один промах, дальше сплошные попадания.

    Слот выбирается самый рано освобождающийся. В очереди стоят принятые
    запросы, ещё не начавшие обслуживание к моменту прибытия нового.
    Запрос, который может начаться немедленно, не отвергается никогда —
    даже при queue_limit = 0.

    Отвергнутый запрос не трогает ни слоты, ни prefix cache: сервер его не
    видел. Записать ему нулевую латентность и посчитать в перцентили —
    классический способ получить красивый отчёт о падающем сервисе.

    len(workload) != len(arrivals) — ValueError: несовпадение обычно значит,
    что расписание сгенерировано на другой темп, и отчёт будет про другой тест.
    """
    if len(workload) != len(arrivals):
        raise ValueError(
            f"workload and arrivals must match: {len(workload)} vs {len(arrivals)}"
        )
    cache = set() if cached_prefixes is None else set(cached_prefixes)
    free_at = [0.0] * max(1, concurrency)
    accepted_starts = []  # моменты НАЧАЛА обслуживания уже принятых запросов
    records = []
    for request, arrival in zip(workload, arrivals):
        slot = min(range(len(free_at)), key=lambda i: free_at[i])
        start = max(arrival, free_at[slot])
        # сколько принятых запросов к этому моменту ещё стоят в очереди
        queued = sum(1 for prior in accepted_starts if prior > arrival)
        if start > arrival and queued >= queue_limit:
            records.append({
                "arrival": arrival, "start": None, "ttft_ms": None,
                "total_ms": None, "wait_ms": None, "cache_hit": None,
                "rejected": True,
            })
            continue
        hit = request["prefix"] in cache
        cache.add(request["prefix"])
        ttft = TTFT_CACHE_HIT_MS if hit else TTFT_CACHE_MISS_MS
        service_ms = ttft + request["output_tokens"] * TPOT_MS
        free_at[slot] = start + service_ms / 1000.0
        accepted_starts.append(start)
        records.append({
            "arrival": arrival,
            "start": start,
            "ttft_ms": ttft + (start - arrival) * 1000.0,
            "total_ms": service_ms + (start - arrival) * 1000.0,
            "wait_ms": (start - arrival) * 1000.0,
            "cache_hit": hit,
            "rejected": False,
        })
    return tuple(records)


def summarize(records):
    """Свести прогон в отчёт с перцентилями и долей отказов.

    Вернуть {"n", "ok", "rejected", "reject_rate", "cache_hit_rate",
             "ttft_p50", "ttft_p95", "ttft_p99", "wait_p99"}.

    summarize(записи_лёгкого_прогона)  ->  reject_rate 0.0, wait_p99 0.0

    Перцентили считаются ТОЛЬКО по успешным запросам, и это ловушка,
    которую все проходят один раз: отвергнутый запрос возвращается мгновенно
    и, попади он в выборку, улучшил бы P99. Чем хуже сервису, тем красивее
    был бы отчёт. Поэтому доля отказов идёт отдельным числом рядом — читать
    перцентили без неё нельзя.

    Прогон без единого успешного запроса — ValueError: перцентиль пустой
    выборки не определён, а «P99 = 0» в такой ситуации хуже ошибки.
    """
    ok = [r for r in records if not r["rejected"]]
    rejected = len(records) - len(ok)
    if not ok:
        raise ValueError("no successful requests: percentiles are undefined")
    ttft = [r["ttft_ms"] for r in ok]
    waits = [r["wait_ms"] for r in ok]
    return {
        "n": len(records),
        "ok": len(ok),
        "rejected": rejected,
        "reject_rate": rejected / len(records),
        "cache_hit_rate": sum(1 for r in ok if r["cache_hit"]) / len(ok),
        "ttft_p50": percentile(ttft, 0.50),
        "ttft_p95": percentile(ttft, 0.95),
        "ttft_p99": percentile(ttft, 0.99),
        "wait_p99": percentile(waits, 0.99),
    }


def apparent_itl(true_itl_ms, tokenize_ms, concurrency, workers=1):
    """Какой inter-token latency ПОКАЖЕТ клиент, считающий токены сам.

    true_itl_ms — сколько на самом деле проходит между токенами на сервере.
    tokenize_ms — сколько клиент тратит на токенизацию одного токена.
    workers     — сколько процессов клиент отвёл под токенизацию.

    apparent_itl(10.0, 0.5, 1, workers=1)    ->  10.0  (клиент успевает)
    apparent_itl(10.0, 0.5, 50, workers=1)   ->  25.0  (клиент не успевает)
    apparent_itl(10.0, 0.5, 50, workers=8)   ->  10.0  (LLM-Locust: раскидали
                                                        токенизацию по процессам)

    Модель GIL trap из урока. Клиент обрабатывает concurrency потоков, и
    вся токенизация в стоковом Locust идёт под одним GIL. Клиент способен
    выдать workers / tokenize_ms токенов в миллисекунду; сервер требует
    concurrency / true_itl_ms. Если клиент медленнее, токены копятся в
    очереди, и измеренный интервал становится КЛИЕНТСКИМ:

        apparent = max(true_itl_ms, concurrency * tokenize_ms / workers)

    Отчёт при этом выглядит абсолютно правдоподобно — «сервер деградирует
    под нагрузкой», — и команда идёт оптимизировать сервер, который ни в
    чём не виноват. Проверяется это тем, что при росте workers «деградация»
    исчезает, чего с настоящей серверной деградацией не бывает.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if true_itl_ms <= 0:
        raise ValueError(f"true_itl_ms must be positive, got {true_itl_ms}")
    return max(true_itl_ms, concurrency * tokenize_ms / workers)


def ci_gate(summary, thresholds):
    """Гейт для CI: пропустить сборку или сломать. Вернуть (passed, breaches).

    thresholds — {имя метрики из summary: максимум}.

    ci_gate(отчёт, {"ttft_p95": 800.0, "reject_rate": 0.05})
        ->  (True, ()) если обе в пределах
        ->  (False, ("ttft_p95",)) если P95 уехал

    Пробой — строгое превышение. Ровно на пороге сборка проходит: «не более
    800 мс» обязано пропускать 800 мс.

    Порядок пробоев — порядок thresholds: сообщение об упавшей сборке
    должно выглядеть одинаково при одинаковом наборе проблем.

    Метрики, которой нет в summary, — KeyError. Гейт по несуществующей
    метрике всегда «проходит», и это худший вид зелёной сборки.
    """
    breaches = []
    for name, limit in thresholds.items():
        if name not in summary:
            raise KeyError(f"summary has no metric named: {name}")
        if summary[name] > limit:
            breaches.append(name)
    return (not breaches), tuple(breaches)
