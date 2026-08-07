"""
Наблюдаемость LLM: трассы, критический путь и сэмплирование — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Чему это соответствует в настоящих инструментах:

    make_span        <-  tracer.start_span() в OpenTelemetry: id, parent_id,
                         времена и атрибуты gen_ai.*
    index_by_parent  <-  то, из чего Jaeger, Langfuse и Phoenix рисуют водопад
    trace_cost       <-  колонка "cost" в Langfuse и Helicone: свёртка
                         gen_ai.usage.* по всем спанам одной трассы
    critical_path    <-  "critical path" в Chrome Tracing и в трассах агентов
    keep_trace       <-  processor tail_sampling в OpenTelemetry Collector:
                         policies = [status_code, numeric_attribute, probabilistic]
    retention_cost   <-  счёт от Datadog-класса против зеркала в своём
                         data lake (сюжет Arize AX «zero-copy»)

Ни сети, ни OTel SDK: экспортёр всё равно только сериализует такие словари.
Часов тоже нет — все времена приходят параметрами. Спан, который сам зовёт
time.time(), невозможно протестировать.
"""

# Средний размер одной сохранённой трассы: промпт, ответ, метаданные.
BYTES_PER_TRACE = 4_500

# Цена хранения за гигабайт в месяц. Монолитный ингест (Datadog-класс)
# против зеркала в собственном object storage, которое читает Arize AX.
PRICE_PER_GB_MONTH_MONOLITHIC = 0.50
PRICE_PER_GB_MONTH_LAKE = 0.005

# Допустимые статусы спана. Всё остальное — сломанный инструментатор.
SPAN_STATUSES = ("ok", "error")


class TraceError(Exception):
    """Трасса собрана неправильно: кривые времена, дубли id, нет корня.

    Свой класс, а не ValueError и тем более не RuntimeError: заготовка
    бросает NotImplementedError, который наследуется от RuntimeError, и
    тест на родительский класс прошёл бы зелёным по пустому файлу.
    """


def make_span(span_id, parent_id, name, start_ms, end_ms, cost_usd=0.0, status="ok"):
    """Один спан трассы: кто, когда, сколько стоил, чем закончился.

    make_span("a", None, "agent", 0, 100)["duration_ms"]        ->  100
    make_span("b", "a", "llm", 10, 40, 0.002)["parent_id"]      ->  'a'
    make_span("c", "a", "tool", 40, 10)                         ->  TraceError

    Ловушка: end_ms < start_ms и неизвестный status — это не «странные
    данные», а сломанный инструментатор. Бросай TraceError, не молчи:
    спан с отрицательной длительностью ломает водопад в Langfuse и тихо
    занижает критический путь.

    parent_id=None означает корень трассы.
    """
    if end_ms < start_ms:
        raise TraceError(f"span {span_id}: end_ms {end_ms} < start_ms {start_ms}")
    if status not in SPAN_STATUSES:
        raise TraceError(f"span {span_id}: unknown status {status!r}")
    # duration считаем здесь один раз: все остальные функции читают готовое
    # поле и не могут разойтись в определении длительности.
    return {
        "span_id": span_id,
        "parent_id": parent_id,
        "name": name,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": end_ms - start_ms,
        "cost_usd": cost_usd,
        "status": status,
    }


def index_by_parent(spans):
    """Индекс «id родителя -> список его детей», дети по возрастанию end_ms.

    Корневые спаны лежат под ключом None.

    index_by_parent([root, a, b])[None]        ->  [root]
    index_by_parent([root, a, b])["root"]      ->  [a, b]

    Ловушка: два спана с одинаковым span_id — это склеенные трассы, дальше
    считать нельзя. TraceError.

    Сортировка нужна не для красоты: критический путь идёт от конца назад и
    обязан брать детей в детерминированном порядке. Ключ (end_ms, span_id),
    чтобы одинаковые времена не зависели от порядка в списке.
    """
    seen = set()
    index = {}
    for sp in spans:
        if sp["span_id"] in seen:
            raise TraceError(f"duplicate span_id {sp['span_id']!r}")
        seen.add(sp["span_id"])
        index.setdefault(sp["parent_id"], []).append(sp)
    for children in index.values():
        children.sort(key=lambda s: (s["end_ms"], s["span_id"]))
    return index


def trace_cost(spans):
    """Стоимость всей трассы: сумма cost_usd по её спанам.

    trace_cost([])                                  ->  0.0
    trace_cost([root, llm_a, llm_b])                ->  сумма трёх cost_usd

    Свёртка по трассе, а не по вызову — потому что в агенте бюджет сжигает
    не корень, а десятый tool-call внутри цикла. Отчёт «дорогие трассы»
    строится ровно этой суммой.
    """
    return float(sum(sp["cost_usd"] for sp in spans))


def critical_path(spans, root_id):
    """Критический путь: цепочка спанов, которая определяет длительность корня.

    Алгоритм — идём от конца корня НАЗАД по времени. Берём ребёнка, который
    заканчивается не позже текущего момента и позже остальных, спускаемся в
    него рекурсивно, переносим момент на его начало и повторяем на том же
    уровне, пока дети не кончатся.

    Родитель [0,100], дети [0,100] и [0,100] (параллельные):
        critical_path(...)  ->  ['root', 'b']        — только один из братьев
    Родитель [0,100], дети [0,50] и [50,100] (последовательные):
        critical_path(...)  ->  ['root', 'b', 'a']   — оба, от позднего к раннему

    Ловушка: спуститься в самого позднего ребёнка и на этом закончить —
    неверно. После возврата из ребёнка надо продолжить перебор БРАТЬЕВ с
    новым моментом времени, иначе последовательная цепочка потеряется.
    """
    index = index_by_parent(spans)
    by_id = {sp["span_id"]: sp for sp in spans}
    if root_id not in by_id:
        raise TraceError(f"no span with id {root_id!r}")

    def walk(node):
        out = [node["span_id"]]
        t = node["end_ms"]
        # reversed по возрастанию end_ms = обход от самого позднего ребёнка
        for child in reversed(index.get(node["span_id"], [])):
            if child["end_ms"] <= t:
                out.extend(walk(child))
                t = child["start_ms"]
        return out

    return walk(by_id[root_id])


def critical_path_ms(spans, root_id):
    """Длительность критического пути — время ответа, а не сумма работ.

    Считаем только самые глубокие звенья пути: те, у кого ни один ребёнок в
    путь не попал. Родителя складывать нельзя — его время это и есть время
    детей, иначе двойной счёт.

    Два параллельных ребёнка по 100 мс:      ->  100.0  (сумма длительностей 200)
    Два последовательных ребёнка по 50 мс:   ->  100.0  (сумма длительностей 100)

    Отсюда правило чтения водопада: агент с десятью параллельными
    tool-call'ами по 2 с отвечает за 2 с, а не за 20. Суммарная длительность
    спанов — это про потраченный ресурс, критический путь — про latency.
    """
    path = critical_path(spans, root_id)
    on_path = set(path)
    index = index_by_parent(spans)
    by_id = {sp["span_id"]: sp for sp in spans}
    total = 0.0
    for span_id in path:
        children = index.get(span_id, [])
        # если хоть один ребёнок лежит на пути, время этого спана уже
        # представлено детьми — складывать его ещё раз нельзя
        if any(c["span_id"] in on_path for c in children):
            continue
        total += by_id[span_id]["duration_ms"]
    return float(total)


def keep_trace(trace, rng, success_rate, cost_threshold):
    """Хвостовое сэмплирование одной трассы: (оставить?, причина).

    Правила строго по порядку:
      'error'      — есть спан со status='error', оставляем всегда;
      'expensive'  — trace_cost(trace) >= cost_threshold, оставляем всегда;
      'sampled'    — иначе монетка rng.random() < success_rate;
      'dropped'    — монетка не выпала.

    keep_trace([err_span], rng, 0.0, 1.0)   ->  (True, 'error')
    keep_trace([ok_span], rng, 0.0, 1.0)    ->  (False, 'dropped')

    Ловушка: монетку бросай ТОЛЬКО дойдя до третьего правила. Если дёрнуть
    rng заранее «на всякий случай», поток случайных чисел сместится и на тех
    же данных получится другой ответ — тест это ловит.

    rng — random.Random(seed). Глобальный random сюда не годится: решение о
    сэмплировании обязано воспроизводиться при разборе инцидента.
    """
    if any(sp["status"] == "error" for sp in trace):
        return (True, "error")
    if trace_cost(trace) >= cost_threshold:
        return (True, "expensive")
    if rng.random() < success_rate:
        return (True, "sampled")
    return (False, "dropped")


def sample_traces(traces, rng, success_rate, cost_threshold):
    """Прогнать хвостовое сэмплирование по списку трасс и собрать отчёт.

    Возвращает dict:
      kept, dropped                    — сколько трасс,
      kept_errors, dropped_errors      — второе обязано быть 0,
      kept_cost_usd, dropped_cost_usd  — сколько денег видно, сколько ослепло,
      by_reason                        — счётчик причин из keep_trace,
      kept_fraction                    — kept / всего (0.0 на пустом входе).

    sample_traces([], rng, 0.05, 1.0)["kept_fraction"]  ->  0.0

    dropped_cost_usd — самая полезная строка отчёта: она отвечает на вопрос
    «какую долю счёта мы больше не видим», а его задают ровно тогда, когда
    расходы уже выросли.
    """
    stats = {
        "kept": 0,
        "dropped": 0,
        "kept_errors": 0,
        "dropped_errors": 0,
        "kept_cost_usd": 0.0,
        "dropped_cost_usd": 0.0,
        "by_reason": {},
    }
    for trace in traces:
        keep, reason = keep_trace(trace, rng, success_rate, cost_threshold)
        stats["by_reason"][reason] = stats["by_reason"].get(reason, 0) + 1
        is_error = any(sp["status"] == "error" for sp in trace)
        cost = trace_cost(trace)
        if keep:
            stats["kept"] += 1
            stats["kept_cost_usd"] += cost
            stats["kept_errors"] += int(is_error)
        else:
            stats["dropped"] += 1
            stats["dropped_cost_usd"] += cost
            stats["dropped_errors"] += int(is_error)
    total = len(traces)
    stats["kept_fraction"] = stats["kept"] / total if total else 0.0
    return stats


def retention_cost(traces_per_day, kept_fraction, price_per_gb_month,
                   bytes_per_trace=BYTES_PER_TRACE, days=30):
    """Счёт за хранение оставленных трасс за `days` дней.

    retention_cost(1_000_000, 1.0, PRICE_PER_GB_MONTH_MONOLITHIC)  ->  67.5
    retention_cost(1_000_000, 1.0, PRICE_PER_GB_MONTH_LAKE)        ->  0.675

    Разбор первого примера: 1e6 трасс * 4500 Б = 4.5 ГБ в день, тридцать дней
    по $0.50 за гигабайт = $67.5.

    Ловушка на порядок величины: это $67.5 в МЕСЯЦ, а не в день. Фразу
    «полное хранение стоит сотни долларов в день» надо пересчитывать, а не
    повторять: сотни в день получатся только на десятках миллионов трасс.
    """
    gb_per_day = traces_per_day * kept_fraction * bytes_per_trace / 1e9
    return gb_per_day * days * price_per_gb_month
