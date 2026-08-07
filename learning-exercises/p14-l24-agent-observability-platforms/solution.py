"""
Observability агентов: Langfuse, Phoenix, Opik — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Собираем руками то, что эти платформы делают за тебя:
  * приём спанов и группировка по сессии — `ingest_spans` (то, что у всех
    трёх называется tracing);
  * длительность сессии по стенным часам — `session_latency_ms`;
  * перцентиль — `latency_percentile` (без него дашборд усредняет всё в
    ноль);
  * guardrail с редактированием PII в стиле Opik — `redact_pii`;
  * LLM-as-a-judge по рубрике — `judge_session` (заглушка судьи
    детерминированная, никакой сети);
  * разбор причин падений и сводка дашборда — `categorize_failures`,
    `summarize`;
  * поиск проблемного трейса, за которым идут в session replay —
    `worst_session`.

Время всегда приходит внутри спанов (start_ns/end_ns), time.time() внутри
нет: иначе метрики нельзя воспроизвести.
"""

import math
import re

# Проверки, которые понимает judge_session. Рубрика — это подмножество этого
# списка с весами.
RUBRIC_CHECKS = ("has_final_answer", "no_tool_errors", "within_step_budget", "no_pii_in_output")

# Бюджет шагов по умолчанию: больше вызовов инструментов за сессию — повод
# посмотреть, не закрутился ли агент.
MAX_STEPS = 5

# Guardrail-паттерны. Держим на уровне модуля: компилировать регулярку на
# каждый спан — заметная часть времени приёма.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
DIGIT_RUN_RE = re.compile(r"\d[\d \-]*\d")
CARD_MIN_DIGITS = 12


def ingest_spans(spans):
    """Разложить поток спанов по сессиям; внутри сессии — по времени старта.

    Спан: {"session_id", "name", "start_ns", "end_ns", "status", "output"}.

    ingest_spans([]) -> {}
    ingest_spans([{"session_id": "s1", "name": "a", "start_ns": 5, "end_ns": 9},
                  {"session_id": "s1", "name": "b", "start_ns": 0, "end_ns": 3}])
        ->  {"s1": [спан "b", спан "a"]}      (b раньше, хотя приехал вторым)

    Ловушка, из которой растут все остальные метрики: спаны приходят из сети
    в произвольном порядке. Если не отсортировать здесь, дашборд будет
    показывать разные числа на одних и тех же данных. Сортируй по
    (start_ns, name) — только start_ns недостаточно, у одновременных спанов
    порядок останется случайным.

    Спан с end_ns раньше start_ns — ValueError: такой спан испортит любую
    длительность, и лучше отбить его на приёме.
    """
    grouped = {}
    for span in spans:
        if span["end_ns"] < span["start_ns"]:
            raise ValueError(f"span {span['name']} ends before it starts")
        grouped.setdefault(span["session_id"], []).append(span)
    # Ключи тоже раскладываем по порядку: сводка не должна зависеть от того,
    # какая сессия попала в поток первой.
    return {sid: sorted(grouped[sid], key=lambda s: (s["start_ns"], s["name"])) for sid in sorted(grouped)}


def session_latency_ms(spans):
    """Длительность сессии по стенным часам: от первого старта до последнего конца.

    session_latency_ms([{"start_ns": 0, "end_ns": 3_000_000}])  ->  3.0
    session_latency_ms([{"start_ns": 0, "end_ns": 5_000_000},
                        {"start_ns": 1_000_000, "end_ns": 2_000_000}])  ->  5.0

    Ловушка: складывать длительности спанов НЕЛЬЗЯ. Вложенный tool-спан
    целиком лежит внутри агентского, и сумма посчитает это время дважды; при
    параллельных вызовах инструментов расхождение доходит до кратного.
    Пользователь ждал стенное время, его и показывай.

    Пустая сессия — ValueError: латентности у неё нет, а 0.0 потом утянет
    вниз среднее по дашборду.
    """
    if not spans:
        raise ValueError("session has no spans")
    start = min(s["start_ns"] for s in spans)
    end = max(s["end_ns"] for s in spans)
    return (end - start) / 1e6


def latency_percentile(values, q):
    """Перцентиль методом ближайшего ранга (nearest-rank).

    latency_percentile([1, 2, 3, 4, 5], 50)   ->  3
    latency_percentile([1, 2, 3, 4, 5], 100)  ->  5
    latency_percentile([1], 95)               ->  1
    latency_percentile([], 95)                ->  ValueError
    latency_percentile([1, 2], 0)             ->  ValueError

    Ближайший ранг: индекс = ceil(q/100 * n) - 1. Никакой интерполяции —
    возвращается реально наблюдавшееся значение, а не среднее между двумя.
    Для латентности это важно: p95 обязан быть числом, которое кто-то
    действительно ждал.

    q = 0 не определён (нулевого ранга не существует), q > 100 тоже —
    ValueError. Порядок значений на входе роли не играет.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 < q <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {q}")
    ordered = sorted(values)
    index = math.ceil(q / 100 * len(ordered)) - 1
    # Клип на всякий случай: при q близком к нулю ceil даёт 0, индекс -1.
    return ordered[max(0, min(index, len(ordered) - 1))]


def redact_pii(text):
    """Guardrail: заменить email на [email], длинные номера на [card].

    redact_pii("write to a.b@x.io")            ->  'write to [email]'
    redact_pii("card 4111 1111 1111 1111 ok")  ->  'card [card] ok'
    redact_pii("step 3 of 10")                 ->  'step 3 of 10'

    «Длинный номер» — последовательность цифр, пробелов и дефисов, в которой
    не меньше CARD_MIN_DIGITS цифр: карту пишут и слитно, и группами по
    четыре, и через дефис.

    Ловушки. Первая: короткие числа трогать нельзя, иначе из трейса пропадут
    номера шагов и коды ошибок. Вторая: функция обязана быть идемпотентной —
    guardrail в проде применяют и на приёме, и перед показом, и второй
    проход не должен ничего портить.

    Зачем: содержимое промптов читает вся дежурная смена. Это и есть
    guardrail-режим Opik, только руками.
    """
    text = EMAIL_RE.sub("[email]", text)

    def mask(match):
        chunk = match.group()
        digits = sum(ch.isdigit() for ch in chunk)
        return "[card]" if digits >= CARD_MIN_DIGITS else chunk

    return DIGIT_RUN_RE.sub(mask, text)


def judge_session(spans, rubric, max_steps=MAX_STEPS):
    """LLM-as-a-judge по рубрике: доля выполненного веса плюс список провалов.

    rubric — {имя проверки из RUBRIC_CHECKS: вес}.

    Возвращает {"score": 0..1, "reasons": [провалившиеся проверки], "passed": bool}.

    Проверки:
      has_final_answer  — среди спанов есть name == "final_answer";
      no_tool_errors    — ни у одного спана status != "ok";
      within_step_budget— спанов с именем на "tool_call" не больше max_steps;
      no_pii_in_output  — redact_pii ничего не изменил ни в одном output.

    judge_session(spans, {"no_tool_errors": 1.0})  ->  score 1.0 или 0.0
    judge_session(spans, {})                       ->  ValueError

    Ловушки. Первая: score — доля от СУММЫ весов, а не сумма весов. Иначе
    рубрика с весами 2 и 2 даст 4 балла из максимума 1, и порог сравнивать
    не с чем; удвоение всех весов не должно менять оценку. Вторая: reasons
    возвращай отсортированными — иначе одинаковые сессии дадут разные
    строки в дашборде. Третья: имя проверки не из RUBRIC_CHECKS — ValueError,
    молча ноль весом ноль опаснее опечатки.

    Судья заглушечный и детерминированный: настоящий LLM-judge без внешних
    инструментов легко выдаёт разные оценки на одном входе, и тогда
    регрессию от шума не отличить (тот самый CRITIC-довод из урока).
    """
    if not spans:
        raise ValueError("session has no spans")
    if not rubric:
        raise ValueError("rubric is empty")
    unknown = sorted(set(rubric) - set(RUBRIC_CHECKS))
    if unknown:
        raise ValueError(f"unknown rubric checks: {unknown}")
    total_weight = sum(rubric.values())
    if total_weight <= 0:
        raise ValueError("rubric weights must sum above zero")

    outputs = [span.get("output") or "" for span in spans]
    outcomes = {
        "has_final_answer": any(s["name"] == "final_answer" for s in spans),
        "no_tool_errors": all(s.get("status", "ok") == "ok" for s in spans),
        "within_step_budget": sum(s["name"].startswith("tool_call") for s in spans) <= max_steps,
        "no_pii_in_output": all(redact_pii(text) == text for text in outputs),
    }
    passed_weight = sum(weight for check, weight in rubric.items() if outcomes[check])
    reasons = sorted(check for check in rubric if not outcomes[check])
    return {"score": passed_weight / total_weight, "reasons": reasons, "passed": not reasons}


def categorize_failures(sessions, rubric, max_steps=MAX_STEPS):
    """Сколько сессий провалило каждую проверку рубрики.

    sessions — то, что вернул ingest_spans.

    categorize_failures({}, {"no_tool_errors": 1.0})  ->  {}
    две сессии, обе с ошибкой инструмента
        ->  {"no_tool_errors": 2}

    Ловушка: сессия с двумя провалами добавляет по единице КАЖДОЙ причине, но
    в каждую — ровно один раз, сколько бы спанов в ней ни падало. Иначе одна
    длинная сессия с двадцатью ошибками перевесит в топе двадцать разных
    сессий, и дежурный пойдёт чинить не то.

    Результат не зависит от порядка сессий: ключи обходятся отсортированными.
    """
    counts = {}
    for session_id in sorted(sessions):
        verdict = judge_session(sessions[session_id], rubric, max_steps)
        for reason in verdict["reasons"]:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def summarize(sessions, rubric, slow_ms, max_steps=MAX_STEPS):
    """Сводка дашборда: доля падений, топ причин, оценки и латентность.

    Возвращает dict с ключами:
      sessions, failure_rate, top_reasons, score_mean,
      latency_mean_ms, latency_p95_ms, latency_max_ms, slow_sessions, rows.

    top_reasons — список пар (причина, счётчик), по убыванию счётчика, при
    равенстве — по алфавиту.
    slow_sessions — id сессий с латентностью строго больше slow_ms.

    summarize({}, {"no_tool_errors": 1.0}, 100)  ->  ValueError

    Ловушка, ради которой в сводке есть и p95, и max, и slow_sessions:
    среднее прячет одиночный длинный трейс. Девяносто девять сессий по 10 ms
    и одна на 30 000 ms дают среднее 310 ms — почти незаметно, а p95, max и
    список медленных показывают выброс сразу. Дашборд из одного среднего —
    это дашборд, который врёт.

    Второе свойство: сводка не зависит от порядка поступления спанов, потому
    что порядок уже зафиксирован в ingest_spans.
    """
    if not sessions:
        raise ValueError("nothing to summarize")

    rows = []
    for session_id in sorted(sessions):
        spans = sessions[session_id]
        verdict = judge_session(spans, rubric, max_steps)
        latency = session_latency_ms(spans)
        rows.append(
            {
                "session_id": session_id,
                "score": verdict["score"],
                "reasons": verdict["reasons"],
                "latency_ms": latency,
                "slow": latency > slow_ms,
            }
        )

    latencies = [row["latency_ms"] for row in rows]
    counts = categorize_failures(sessions, rubric, max_steps)
    # Сортировка с двумя ключами: убывание счётчика и алфавит при равенстве —
    # иначе топ причин будет прыгать между прогонами.
    top_reasons = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    return {
        "sessions": len(rows),
        "failure_rate": sum(1 for row in rows if row["reasons"]) / len(rows),
        "top_reasons": top_reasons,
        "score_mean": sum(row["score"] for row in rows) / len(rows),
        "latency_mean_ms": sum(latencies) / len(rows),
        "latency_p95_ms": latency_percentile(latencies, 95),
        "latency_max_ms": max(latencies),
        "slow_sessions": [row["session_id"] for row in rows if row["slow"]],
        "rows": rows,
    }


def worst_session(sessions, rubric, max_steps=MAX_STEPS):
    """Найти трейс, с которого начинать разбор: худшая оценка, при равенстве — медленнее.

    Возвращает {"session_id", "score", "latency_ms", "reasons"}.

    worst_session({}, {"no_tool_errors": 1.0})  ->  ValueError

    Порядок сравнения: сначала меньшая оценка судьи, потом БОЛЬШАЯ
    латентность, потом id по алфавиту. Третий ключ нужен не для красоты: без
    него две одинаково плохие сессии будут выигрывать по очереди в
    зависимости от порядка обхода словаря, и ссылка в тикете перестанет
    открывать то, о чём тикет.

    Зачем это в AI: сводка говорит «стало хуже», а разбирать всё равно надо
    один конкретный прогон. Это дверь в session replay.
    """
    if not sessions:
        raise ValueError("no sessions")

    def rank(session_id):
        verdict = judge_session(sessions[session_id], rubric, max_steps)
        return (verdict["score"], -session_latency_ms(sessions[session_id]), session_id)

    session_id = min(sorted(sessions), key=rank)
    verdict = judge_session(sessions[session_id], rubric, max_steps)
    return {
        "session_id": session_id,
        "score": verdict["score"],
        "latency_ms": session_latency_ms(sessions[session_id]),
        "reasons": verdict["reasons"],
    }
