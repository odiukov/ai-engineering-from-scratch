"""
Observability агентов: Langfuse, Phoenix, Opik

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l24-agent-observability-platforms
Разбор:  /check-code p14-l24-agent-observability-platforms
"""

import math
import re

RUBRIC_CHECKS = ("has_final_answer", "no_tool_errors", "within_step_budget", "no_pii_in_output")
MAX_STEPS = 5
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
