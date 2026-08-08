"""
Метрики инференса: TTFT, TPOT, ITL, goodput, P99 — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Запрос в этом уроке — словарь с тремя измеренными величинами:
#   ttft_ms       — время до первого токена
#   tpot_ms       — среднее время на выходной токен после первого
#   output_tokens — сколько токенов реально отдали
# Ровно это отдаёт любой benchmark-клиент, и из этого считается всё остальное.
REQUEST_KEYS = ("ttft_ms", "tpot_ms", "output_tokens")

# Потребительский SLO для 70B-чата из урока.
CONSUMER_SLO = {"ttft_ms": 800.0, "tpot_ms": 25.0, "e2e_ms": 3000.0}

# Два инструмента считают ITL по-разному, и это не баг, а разные определения.
ITL_TOOLS = ("genai-perf", "llmperf")


class TraceTooShortError(Exception):
    """Трассы короче двух токенов не хватает, чтобы посчитать ITL по GenAI-Perf.

    Свой класс, а не ZeroDivisionError и не голый RuntimeError: заготовка
    бросает NotImplementedError, который сам является RuntimeError, и тест
    на RuntimeError позеленел бы, ничего не проверив.
    """


def ttft_ms(queue_ms, network_ms, prefill_ms):
    """TTFT — время до первого токена: очередь + сеть + prefill.

    ttft_ms(40, 12, 110)  ->  162.0   (референс Llama-3.1-8B на TRT-LLM)
    ttft_ms(0, 0, 800)    ->  800.0   (32k промпт, один голый prefill)

    Все три слагаемых лечатся по-разному: очередь — планировщиком, сеть —
    географией, prefill — длиной промпта и железом. Поэтому их и не
    складывают в одно «медленно».
    """
    return queue_ms + network_ms + prefill_ms


def e2e_ms(ttft, tpot_ms, output_tokens, network_response_ms=0.0):
    """E2E: TTFT + TPOT * max(output_tokens - 1, 0) + сеть на ответ.

    e2e_ms(162.0, 7.33, 127)  ->  1085.58    (126 интервалов после первого токена)
    e2e_ms(800.0, 7.0, 10)    ->  863.0      (короткий ответ: правит TTFT)

    Считай, какое слагаемое главное: на длинных ответах (>500 токенов) E2E
    определяется TPOT, на коротких с длинным промптом — TTFT. Поэтому E2E
    всегда публикуют вместе с длиной ответа. Первый токен уже включён в
    TTFT, поэтому TPOT платим только за оставшиеся интервалы.
    """
    if output_tokens < 0:
        raise ValueError("output_tokens must not be negative")
    return ttft + tpot_ms * max(output_tokens - 1, 0) + network_response_ms


def itl_ms(ttft, decode_ms, output_tokens, tool):
    """ITL по версии конкретного инструмента. Один прогон — два разных числа.

    decode_ms — суммарное время выдачи токенов ПОСЛЕ первого.

    itl_ms(500, 700, 100, "genai-perf")  ->  7.07...   (700 / 99)
    itl_ms(500, 700, 100, "llmperf")     ->  12.0      ((500 + 700) / 100)

    GenAI-Perf считает интервалы между токенами и начинает со второго
    токена — интервалов на один меньше, чем токенов. LLMPerf делит всё
    время запроса на все токены, то есть тащит TTFT внутрь ITL.

    Ловушка: для однотокенного ответа интервалов нет вообще, и делить не на
    что — это TraceTooShortError, а не ноль и не ZeroDivisionError.
    Незнакомое имя инструмента — ValueError: молча выбирать формулу нельзя.
    """
    if tool not in ITL_TOOLS:
        raise ValueError(f"unknown tool: {tool}")
    if tool == "llmperf":
        if output_tokens < 1:
            raise TraceTooShortError("llmperf needs at least one output token")
        return (ttft + decode_ms) / output_tokens
    if output_tokens < 2:
        raise TraceTooShortError("genai-perf needs at least two output tokens")
    return decode_ms / (output_tokens - 1)


def percentile(values, p):
    """Перцентиль по методу nearest-rank. p задаётся в процентах.

    percentile([1, 2, 3, 4, 5], 50)   ->  3
    percentile(list(range(1, 101)), 99)  ->  99
    percentile([1, 2, 3, 4, 5], 100)  ->  5

    Метод: отсортировать, взять элемент с номером ceil(p/100 * n), нумерация
    с единицы. Никакой интерполяции — перцентиль всегда реально
    наблюдавшееся значение, и это ровно то, что показывают benchmark-тулы.

    Пустая выборка — ValueError: у ничего нет ни P50, ни P99.
    """
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0 <= p <= 100:
        raise ValueError("p must be a percentage between 0 and 100")
    ordered = sorted(values)
    rank = math.ceil(p / 100.0 * len(ordered))
    # rank == 0 бывает только при p == 0; берём минимум
    index = max(0, rank - 1)
    return ordered[index]


def latency_summary(values):
    """Тройка перцентилей плюс среднее: {"p50", "p90", "p99", "mean"}.

    latency_summary([1, 2, 3, 4, 5])["p50"]   ->  3
    latency_summary([1, 2, 3, 4, 5])["mean"]  ->  3.0

    Урок требует публиковать именно тройку. Среднее оставлено рядом
    специально: на нём хорошо видно, что оно НЕ восстанавливает хвост —
    две выборки с одинаковым средним легко расходятся по P99 в разы.
    """
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "mean": sum(values) / len(values),
    }


def throughput_tokens_per_s(requests, elapsed_s):
    """Пропускная способность парка: все выходные токены делить на время.

    throughput_tokens_per_s([{"ttft_ms": 100, "tpot_ms": 7, "output_tokens": 150}], 1.0)
        ->  150.0

    Метрика агрегатная: она ничего не знает про то, дождался ли конкретный
    пользователь ответа. Ровно поэтому «15 000 tok/s» само по себе не
    доказывает, что продукт работает.
    """
    if elapsed_s <= 0:
        raise ValueError("elapsed_s must be positive")
    return sum(r["output_tokens"] for r in requests) / elapsed_s


def goodput(requests, slo):
    """Доля запросов, уложившихся во ВСЕ ограничения SLO одновременно.

    slo — словарь с ключами ttft_ms, tpot_ms, e2e_ms (все три обязательны).

    goodput([{"ttft_ms": 100, "tpot_ms": 7, "output_tokens": 100}], CONSUMER_SLO)
        ->  1.0
    goodput([{"ttft_ms": 900, "tpot_ms": 7, "output_tokens": 100}], CONSUMER_SLO)
        ->  0.0   (TTFT не прошёл — и неважно, что остальное идеально)

    Это И, а не ИЛИ: запрос, выполнивший два условия из трёх, — плохой.
    E2E считается через e2e_ms, а не берётся из запроса.
    """
    if not requests:
        raise ValueError("goodput of an empty run is undefined")
    good = 0
    for r in requests:
        end_to_end = e2e_ms(r["ttft_ms"], r["tpot_ms"], r["output_tokens"])
        if (r["ttft_ms"] <= slo["ttft_ms"]
                and r["tpot_ms"] <= slo["tpot_ms"]
                and end_to_end <= slo["e2e_ms"]):
            good += 1
    return good / len(requests)


def slo_breakdown(requests, slo):
    """Кто именно завалил SLO: {"ttft", "tpot", "e2e", "any"} — счётчики нарушений.

    Один запрос может попасть сразу в несколько счётчиков, поэтому сумма
    первых трёх обычно больше, чем "any".

    slo_breakdown([{"ttft_ms": 900, "tpot_ms": 40, "output_tokens": 10}], CONSUMER_SLO)
        ->  {"ttft": 1, "tpot": 1, "e2e": 0, "any": 1}

    Зачем: goodput говорит «плохо», а разбивка говорит, что чинить —
    планировщик очереди, chunked prefill или длину ответов.
    Связь с goodput: any / len(requests) == 1 - goodput.
    """
    counts = {"ttft": 0, "tpot": 0, "e2e": 0, "any": 0}
    for r in requests:
        end_to_end = e2e_ms(r["ttft_ms"], r["tpot_ms"], r["output_tokens"])
        bad_ttft = r["ttft_ms"] > slo["ttft_ms"]
        bad_tpot = r["tpot_ms"] > slo["tpot_ms"]
        bad_e2e = end_to_end > slo["e2e_ms"]
        counts["ttft"] += bad_ttft
        counts["tpot"] += bad_tpot
        counts["e2e"] += bad_e2e
        counts["any"] += bad_ttft or bad_tpot or bad_e2e
    return counts
