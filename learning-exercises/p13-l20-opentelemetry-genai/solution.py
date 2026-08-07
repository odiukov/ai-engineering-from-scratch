"""
OpenTelemetry GenAI — трассировка вызовов инструментов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

OTel SDK делает всё это одной строкой `with tracer.start_as_current_span(...)`.
Здесь мы собираем спан руками, чтобы стало видно, из чего он состоит и какие
именно атрибуты требует семантическое соглашение GenAI (semconv 1.37+).
Соответствие настоящему API:

    new_span                    <-  tracer.start_span(name, kind, context)
    finish_span                 <-  span.end()
    traceparent                 <-  TraceContextTextMapPropagator.inject()
    parse_traceparent           <-  TraceContextTextMapPropagator.extract()
    missing_gen_ai_attributes   <-  то, что проверяют линтеры semconv
    capture_content_event       <-  span.add_event() под opt-in переключателем
    span_tree                   <-  дерево, которое рисует Jaeger / Langfuse
    trace_problems              <-  валидатор трассы перед экспортом

Экспортёр (OTLP, Jaeger, Datadog) не нужен: он всего лишь сериализует эти
словари. Мы работаем сразу со словарями.

Времени по часам здесь тоже нет: наносекунды приходят параметром. Спан,
который сам зовёт time.time_ns(), невозможно протестировать.
"""

# Направление спана. SERVER — входящий запрос, CLIENT — исходящий за границу
# процесса (LLM-провайдер, MCP-сервер), INTERNAL — шаг внутри агента.
SPAN_KINDS = ("INTERNAL", "CLIENT", "SERVER", "PRODUCER", "CONSUMER")

# Обязательные атрибуты по операциям. Порядок внутри кортежа — порядок из
# спецификации, и отчёт о недостающих отдаётся в нём же.
REQUIRED_ATTRS = {
    "chat": (
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
    ),
    "embeddings": (
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
    ),
    "execute_tool": (
        "gen_ai.operation.name",
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
    ),
    "invoke_agent": (
        "gen_ai.operation.name",
        "gen_ai.agent.name",
        "gen_ai.agent.id",
    ),
}

# События с содержимым запроса и ответа. Всё остальное — не content-событие.
CONTENT_EVENTS = (
    "gen_ai.content.prompt",
    "gen_ai.content.completion",
    "gen_ai.content.tool_call",
)

# W3C trace context: версия и длины идентификаторов в hex-символах.
TRACEPARENT_VERSION = "00"
TRACE_ID_HEX_LEN = 32
SPAN_ID_HEX_LEN = 16


def new_span(name, kind, start_ns, rng, parent=None, attrs=None):
    """Создать спан. Дочерний наследует traceId родителя, но не spanId.

    rng — random.Random, из него берутся идентификаторы. Глобальный random
    сюда не годится: два прогона теста обязаны дать одни и те же id.

    root = new_span("agent.invoke_agent", "INTERNAL", 1000, rng)
        ->  {"name": ..., "traceId": <32 hex>, "spanId": <16 hex>,
             "parentSpanId": None, "endTimeUnixNano": None, ...}
    child = new_span("llm.chat", "CLIENT", 1100, rng, parent=root)
        ->  child["traceId"] == root["traceId"]
            child["parentSpanId"] == root["spanId"]

    endTimeUnixNano равен None, пока спан не закрыт. Ноль сюда не годится:
    ноль — законное время, и "не закрыт" от "закрыт в нулевую наносекунду"
    было бы не отличить.

    Неизвестный kind — ValueError. Экспортёр молча выбросит спан с чужим
    видом, и в бэкенде его просто не будет.
    """
    if kind not in SPAN_KINDS:
        raise ValueError(f"unknown span kind: {kind}")
    # traceId рождается только у корня; у потомка он всегда родительский —
    # иначе трасса развалится на несколько несвязанных деревьев.
    trace_id = parent["traceId"] if parent else format(rng.getrandbits(128), "032x")
    return {
        "name": name,
        "kind": kind,
        "traceId": trace_id,
        "spanId": format(rng.getrandbits(64), "016x"),
        "parentSpanId": parent["spanId"] if parent else None,
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": None,
        "attributes": dict(attrs or {}),
        "events": [],
    }


def finish_span(span, end_ns):
    """Закрыть спан временем end_ns. Вернуть его же.

    finish_span(span, 2000)  ->  span["endTimeUnixNano"] == 2000

    Два отказа, оба ValueError:
      * спан уже закрыт — повторный end() в OTel тихо игнорируется, и потому
        двойное закрытие живёт в коде годами; здесь оно шумит сразу;
      * end_ns меньше начала — часы поехали назад, и длительность выйдет
        отрицательной. В бэкенде такой спан рисуется полосой нулевой длины,
        и найти причину потом почти невозможно.

    Нулевая длительность (end_ns == start_ns) законна: мгновенный шаг.
    """
    if span["endTimeUnixNano"] is not None:
        raise ValueError(f"span {span['name']} is already finished")
    if end_ns < span["startTimeUnixNano"]:
        raise ValueError(f"span {span['name']}: end {end_ns} is before start")
    span["endTimeUnixNano"] = end_ns
    return span


def traceparent(span, sampled=True):
    """Заголовок W3C traceparent для передачи контекста за границу процесса.

    traceparent(span)  ->  "00-<32 hex traceId>-<16 hex spanId>-01"
    traceparent(span, sampled=False)  ->  "...-00"

    Именно эту строку MCP-клиент кладёт в HTTP-заголовок, а для stdio — в
    поле `_meta.traceparent` JSON-RPC-запроса. Сервер её разбирает и
    продолжает ТУ ЖЕ трассу вместо того, чтобы завести свою.

    В заголовок идёт spanId текущего спана: для сервера он станет
    parentSpanId. Родительский id сюда подставлять нельзя — потеряется
    один уровень дерева.
    """
    if len(span["traceId"]) != TRACE_ID_HEX_LEN:
        raise ValueError(f"traceId must be {TRACE_ID_HEX_LEN} hex chars")
    if len(span["spanId"]) != SPAN_ID_HEX_LEN:
        raise ValueError(f"spanId must be {SPAN_ID_HEX_LEN} hex chars")
    flags = "01" if sampled else "00"
    return f"{TRACEPARENT_VERSION}-{span['traceId']}-{span['spanId']}-{flags}"


def parse_traceparent(header):
    """Разобрать заголовок traceparent. Словарь или ValueError.

    parse_traceparent("00-" + "a" * 32 + "-" + "b" * 16 + "-01")
        ->  {"traceId": "aaaa...", "spanId": "bbbb...", "sampled": True}

    Всё, что не по спецификации, — ValueError, и вот почему по каждому пункту:
      * не четыре поля или чужая версия — заголовок не наш;
      * не hex или неверная длина — id непригоден для корреляции;
      * ВЕРХНИЙ регистр запрещён: спецификация требует нижний, а бэкенды
        сравнивают id как строки, и "AB..." с "ab..." не склеятся в одну
        трассу;
      * traceId или spanId из одних нулей — зарезервированное «нет
        значения»; принять его значит склеить в одну трассу всех, кто
        забыл проставить контекст.
    """
    parts = header.split("-")
    if len(parts) != 4:
        raise ValueError(f"traceparent must have 4 fields, got {len(parts)}")
    version, trace_id, span_id, flags = parts
    if version != TRACEPARENT_VERSION:
        raise ValueError(f"unsupported traceparent version: {version}")
    for value, size, label in (
        (trace_id, TRACE_ID_HEX_LEN, "traceId"),
        (span_id, SPAN_ID_HEX_LEN, "spanId"),
    ):
        if len(value) != size:
            raise ValueError(f"{label} must be {size} hex chars, got {len(value)}")
        if value != value.lower():
            raise ValueError(f"{label} must be lowercase hex")
        try:
            int(value, 16)
        except ValueError:
            raise ValueError(f"{label} is not hex: {value}") from None
        if value == "0" * size:
            raise ValueError(f"{label} of all zeroes is reserved as 'unset'")
    if len(flags) != 2:
        raise ValueError(f"trace-flags must be 2 hex chars, got {flags!r}")
    # sampled — младший бит флагов, а не строгое равенство "01":
    # спецификация разрешает и другие биты в том же байте.
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "sampled": bool(int(flags, 16) & 1),
    }


def missing_gen_ai_attributes(span):
    """Каких обязательных gen_ai.* атрибутов не хватает спану. Список в порядке semconv.

    Пусть у спана есть только gen_ai.operation.name = "execute_tool".
    missing_gen_ai_attributes(span)
        ->  ["gen_ai.tool.name", "gen_ai.tool.call.id"]

    Полностью заполненный спан даёт [].

    Спан без gen_ai.operation.name — ValueError: по этому атрибуту бэкенд
    решает, как спан вообще читать, и без него список требований не
    определён. Незнакомая операция — тоже ValueError: обычно это опечатка
    ("tool_execute" вместо "execute_tool"), и она стоит целого дашборда.
    """
    attrs = span["attributes"]
    operation = attrs.get("gen_ai.operation.name")
    if operation is None:
        raise ValueError(f"span {span['name']} has no gen_ai.operation.name")
    if operation not in REQUIRED_ATTRS:
        raise ValueError(f"unknown gen_ai operation: {operation}")
    return [name for name in REQUIRED_ATTRS[operation] if name not in attrs]


def capture_content_event(span, event_name, content, at_ns, capture_content=False):
    """Добавить событие с содержимым — только если сбор содержимого включён.

    capture_content_event(span, "gen_ai.content.prompt", "hi", 1500)
        ->  False, событий у спана не прибавилось
    capture_content_event(span, "gen_ai.content.prompt", "hi", 1500, True)
        ->  True, span["events"][-1]["attributes"]["content"] == "hi"

    Сбор выключен по умолчанию не из экономии: в prompt лежат персональные
    данные пользователя, и трасса уезжает наружу, в чужой бэкенд. Включают
    его осознанно, через OTEL_SEMCONV_STABILITY_OPT_IN и переменные
    content-capture.

    Имя события проверяется ДО проверки флага. Иначе опечатка в имени
    отлежится в коде до того дня, когда сбор включат в проде.
    """
    if event_name not in CONTENT_EVENTS:
        raise ValueError(f"not a gen_ai content event: {event_name}")
    if not capture_content:
        return False
    span["events"].append(
        {
            "name": event_name,
            "timeUnixNano": at_ns,
            "attributes": {"content": content},
        }
    )
    return True


def span_tree(spans):
    """Собрать из плоского списка спанов дерево. Корневой узел или ValueError.

    Узел: {"span": <спан>, "children": [<узел>, ...]}.

    span_tree([root, llm, tool])
        ->  {"span": root, "children": [{"span": llm, "children": []},
                                        {"span": tool, "children": []}]}

    Дети упорядочены по времени начала, при совпадении — по spanId. Порядок
    спанов во входном списке на дерево не влияет: экспортёр отдаёт их в том
    порядке, в каком они закрылись, а закрываются родители последними.

    Четыре причины для ValueError — все означают, что трасса битая и рисовать
    её нельзя: разные traceId в одном списке, повторный spanId, ссылка на
    несуществующего родителя, не ровно один корень.
    """
    if not spans:
        raise ValueError("no spans to build a tree from")
    trace_ids = {s["traceId"] for s in spans}
    if len(trace_ids) != 1:
        raise ValueError(f"spans belong to {len(trace_ids)} different traces")
    by_id = {}
    for span in spans:
        if span["spanId"] in by_id:
            raise ValueError(f"duplicate spanId: {span['spanId']}")
        by_id[span["spanId"]] = span
    for span in spans:
        parent_id = span["parentSpanId"]
        if parent_id is not None and parent_id not in by_id:
            raise ValueError(f"span {span['name']}: parent {parent_id} is not here")
    roots = [s for s in spans if s["parentSpanId"] is None]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root span, got {len(roots)}")

    children = {}
    for span in spans:
        children.setdefault(span["parentSpanId"], []).append(span)

    def node(span):
        kids = sorted(
            children.get(span["spanId"], ()),
            key=lambda c: (c["startTimeUnixNano"], c["spanId"]),
        )
        return {"span": span, "children": [node(k) for k in kids]}

    return node(roots[0])


def trace_problems(spans):
    """Проверить трассу перед экспортом. Список претензий, пустой — значит норма.

    В отличие от span_tree ничего не бросает: это линтер, ему нужно
    перечислить ВСЕ беды разом, а не упасть на первой.

    trace_problems([root, child_ending_after_root])
        ->  ["llm.chat: ends after parent agent.invoke_agent"]

    Что проверяется, кроме атрибутов:
      * незакрытый спан — забытый finish_span; в бэкенде он не появится;
      * ребёнок начался раньше родителя или закончился позже него. Это
        главная смысловая проверка трассы: родитель по определению
        охватывает всё, что произошло внутри него. Нарушение означает, что
        спан прицепили не к тому родителю — а по красивой картинке в Jaeger
        это незаметно.
    """
    problems = []
    if len({s["traceId"] for s in spans}) > 1:
        problems.append("spans belong to more than one trace")
    by_id = {s["spanId"]: s for s in spans}
    for span in spans:
        label = span["name"]
        if span["endTimeUnixNano"] is None:
            problems.append(f"{label}: span is not finished")
        try:
            missing = missing_gen_ai_attributes(span)
        except ValueError as exc:
            # спан без операции или с чужой операцией — тоже претензия,
            # но линтер не должен из-за неё падать целиком
            problems.append(f"{label}: {exc}")
        else:
            problems.extend(f"{label}: missing {attr}" for attr in missing)
        parent_id = span["parentSpanId"]
        if parent_id is None:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            problems.append(f"{label}: parent {parent_id} is not in this trace")
            continue
        if span["startTimeUnixNano"] < parent["startTimeUnixNano"]:
            problems.append(f"{label}: starts before parent {parent['name']}")
        if (
            span["endTimeUnixNano"] is not None
            and parent["endTimeUnixNano"] is not None
            and span["endTimeUnixNano"] > parent["endTimeUnixNano"]
        ):
            problems.append(f"{label}: ends after parent {parent['name']}")
    return problems
