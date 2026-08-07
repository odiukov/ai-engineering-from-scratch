"""
Семантические конвенции OpenTelemetry GenAI — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Собираем руками то, что даёт OTel SDK с GenAI-инструментацией:
  * имя и kind спана по конвенции (`invoke_agent {gen_ai.agent.name}`,
    CLIENT против INTERNAL) — `describe_span`;
  * набор атрибутов `gen_ai.*` — `genai_attributes`;
  * W3C trace context: `format_traceparent`, `continue_trace` — тот самый
    заголовок, через который трейс переживает границу процесса;
  * дерево спанов с родительскими связями — `start_span`, `end_span`,
    `span_tree`;
  * контракт content capture (по умолчанию НЕ захватываем, продовый режим —
    внешнее хранилище и ссылка на спане) — `capture_content`.

Ни сети, ни экспортёра, ни сторонних пакетов. Время всегда приходит
параметром (start_ns/end_ns), никаких time.time() внутри.
"""

# Значения gen_ai.operation.name, которые перечисляет урок.
GEN_AI_OPERATIONS = ("chat", "completion", "create_agent", "invoke_agent", "tool_call")

# Значения gen_ai.provider.name из конвенции.
GEN_AI_PROVIDERS = ("anthropic", "openai", "aws.bedrock", "google.vertex")

# Режимы захвата содержимого. "off" — то, что конвенция требует по умолчанию.
CONTENT_MODES = ("off", "inline", "reference")


def describe_span(operation, target=None, remote=False):
    """Имя и kind спана по конвенции GenAI.

    describe_span("invoke_agent", "planner")
        ->  {"name": "invoke_agent planner", "kind": "INTERNAL"}
    describe_span("invoke_agent")
        ->  {"name": "invoke_agent", "kind": "INTERNAL"}
    describe_span("invoke_agent", "planner", remote=True)
        ->  {"name": "invoke_agent planner", "kind": "CLIENT"}
    describe_span("chat", "claude-x")
        ->  {"name": "chat claude-x", "kind": "CLIENT"}

    `target` — это gen_ai.agent.name для агентских спанов,
    gen_ai.request.model для модельных и имя инструмента для tool_call.
    Без него имя спана — просто операция, без хвоста и без пробела.

    Ловушка: remote влияет ТОЛЬКО на invoke_agent. Спаны chat/completion
    всегда CLIENT (это вызов удалённого API), а create_agent и tool_call
    всегда INTERNAL — они происходят внутри процесса, сколько бы ни
    передавали remote=True.
    """
    if operation not in GEN_AI_OPERATIONS:
        raise ValueError(f"unknown gen_ai.operation.name: {operation}")

    name = f"{operation} {target}" if target else operation

    if operation in ("chat", "completion"):
        kind = "CLIENT"
    elif operation == "invoke_agent":
        # CLIENT — только для удалённого агентского сервиса (Bedrock Agents,
        # Assistants API). Локальный ReAct-цикл — INTERNAL.
        kind = "CLIENT" if remote else "INTERNAL"
    else:
        kind = "INTERNAL"
    return {"name": name, "kind": kind}


def genai_attributes(
    provider,
    operation,
    request_model=None,
    response_model=None,
    agent_name=None,
    data_source_id=None,
):
    """Словарь атрибутов gen_ai.* — только те ключи, значение которых есть.

    genai_attributes("anthropic", "chat", request_model="claude-x")
        ->  {"gen_ai.provider.name": "anthropic",
             "gen_ai.operation.name": "chat",
             "gen_ai.request.model": "claude-x"}

    genai_attributes("openai", "chat", "gpt-x", response_model="gpt-x-0301")
        ->  ... + {"gen_ai.response.model": "gpt-x-0301"}

    genai_attributes("mistral", "chat")  ->  ValueError

    Ловушка: атрибут с пустым значением класть НЕЛЬЗЯ — в бэкенде
    "gen_ai.response.model": None и отсутствие ключа выглядят по-разному, и
    дашборд начинает считать None отдельной моделью. Нет значения — нет ключа.

    Зачем: response.model отличается от request.model, когда провайдер
    отроутил запрос. Без обоих атрибутов регрессию «нас перевели на другую
    ревизию модели» не увидеть.
    """
    if provider not in GEN_AI_PROVIDERS:
        raise ValueError(f"unknown gen_ai.provider.name: {provider}")
    if operation not in GEN_AI_OPERATIONS:
        raise ValueError(f"unknown gen_ai.operation.name: {operation}")

    pairs = (
        ("gen_ai.provider.name", provider),
        ("gen_ai.operation.name", operation),
        ("gen_ai.request.model", request_model),
        ("gen_ai.response.model", response_model),
        ("gen_ai.agent.name", agent_name),
        ("gen_ai.data_source.id", data_source_id),
    )
    return {key: value for key, value in pairs if value is not None}


def format_traceparent(trace_id, span_id, sampled=True):
    """Заголовок W3C traceparent: version-traceid-spanid-flags.

    format_traceparent("a" * 32, "b" * 16)
        ->  '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01'
    format_traceparent("a" * 32, "b" * 16, sampled=False)
        ->  '00-...-00'
    format_traceparent("A" * 32, "b" * 16)   ->  ValueError

    Требования W3C, за которые ловят чаще всего: 32 hex-символа на trace id,
    16 на span id, ТОЛЬКО нижний регистр, и ни то, ни другое не может быть
    из одних нулей — все нули означают «идентификатора нет».
    """
    hex_chars = "0123456789abcdef"
    for value, width, label in ((trace_id, 32, "trace_id"), (span_id, 16, "span_id")):
        if len(value) != width or any(ch not in hex_chars for ch in value):
            raise ValueError(f"{label} must be {width} lowercase hex chars")
        if set(value) == {"0"}:
            raise ValueError(f"{label} must not be all zeros")
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def continue_trace(header, new_trace_id=None):
    """Создать трейс: либо продолжить входящий traceparent, либо начать новый.

    Возвращает {"trace_id", "spans": [], "stack": [...], "remote_parent"}.

    continue_trace(None, "a" * 32)["trace_id"]      ->  'aaaa...'  (32 символа)
    continue_trace(None, "a" * 32)["remote_parent"] ->  None
    continue_trace("00-" + "a" * 32 + "-" + "b" * 16 + "-01")
        ->  trace_id = "aaaa...", remote_parent = "bbbbbbbbbbbbbbbb"
    continue_trace(None)                            ->  ValueError

    Стек открытых спанов всегда стартует пустым: удалённый родитель живёт в
    отдельном поле remote_parent, потому что закрывать его мы не будем — он
    закроется в чужом процессе.

    Ключевое свойство: trace_id НЕ рождается здесь, когда пришёл заголовок —
    он берётся из него. Сгенерируешь свой — и вызов в дочернем процессе
    окажется отдельным трейсом, а не веткой родительского; именно так
    ломается сквозной трейс через CLI-субпроцесс.

    Версия отличная от "00" — ValueError: разбирать неизвестный формат наугад
    хуже, чем отказаться.
    """
    if header is None:
        if not new_trace_id:
            raise ValueError("either header or new_trace_id is required")
        # Проверяем формат тем же кодом, что и на выходе. Span id здесь
        # фиктивный: нужна только валидация trace_id.
        format_traceparent(new_trace_id, "0" * 15 + "1")
        return {"trace_id": new_trace_id, "spans": [], "stack": [], "remote_parent": None}

    parts = header.split("-")
    if len(parts) != 4 or parts[0] != "00":
        raise ValueError(f"unsupported traceparent: {header}")
    _, trace_id, parent_span_id, flags = parts
    if len(flags) != 2 or any(ch not in "0123456789abcdef" for ch in flags):
        raise ValueError(f"bad traceparent flags: {flags}")
    format_traceparent(trace_id, parent_span_id, sampled=flags == "01")
    return {
        "trace_id": trace_id,
        "spans": [],
        "stack": [],
        "remote_parent": parent_span_id,
    }


def start_span(trace, span_id, name, kind, attributes, start_ns):
    """Открыть спан: родитель — тот, что сейчас на вершине стека трейса.

    trace = continue_trace(None, "a" * 32)
    start_span(trace, "b" * 16, "invoke_agent p", "INTERNAL", {}, 0)["parent_id"]
        ->  None
    затем start_span(trace, "c" * 16, "tool_call s", "INTERNAL", {}, 10)["parent_id"]
        ->  'bbbbbbbbbbbbbbbb'

    Возвращает сам спан: trace_id, span_id, parent_id, name, kind,
    attributes, start_ns, end_ns=None.

    Если стек пуст, а трейс продолжает входящий traceparent, родителем
    становится remote_parent — так ветка из другого процесса прирастает к
    родительскому спану.

    Ловушки. Первая: trace_id спан НЕ придумывает — берёт из трейса; трейс
    без trace_id — ValueError. Вторая: повторный span_id — ValueError, иначе
    дерево склеит два разных спана в один узел. Третья: attributes надо
    скопировать, иначе один словарь окажется общим у нескольких спанов и
    правка одного перепишет остальные.
    """
    if not trace.get("trace_id"):
        raise ValueError("trace_id must exist before the first span")
    if any(s["span_id"] == span_id for s in trace["spans"]):
        raise ValueError(f"duplicate span_id: {span_id}")

    # Пустой стек не значит «корень»: родитель мог приехать в traceparent.
    parent_id = trace["stack"][-1] if trace["stack"] else trace.get("remote_parent")
    span = {
        "trace_id": trace["trace_id"],
        "span_id": span_id,
        "parent_id": parent_id,
        "name": name,
        "kind": kind,
        "attributes": dict(attributes),
        "start_ns": start_ns,
        "end_ns": None,
    }
    trace["spans"].append(span)
    trace["stack"].append(span_id)
    return span


def end_span(trace, span_id, end_ns):
    """Закрыть спан. Закрыть можно только самый внутренний из открытых.

    Возвращает спан с заполненными end_ns и duration_ns.

    Ловушка, ради которой всё это: если разрешить закрывать родителя, пока
    открыт ребёнок, получится спан, который «переживает» родителя — в
    бэкенде это либо отрицательная длительность у ребёнка, либо оторванная
    ветка. Порядок строго LIFO, попытка закрыть не вершину — ValueError.

    Вторая ловушка: end_ns < start_ns — тоже ValueError, а не отрицательная
    длительность. Часы монотонные, время приходит параметром; отрицательная
    длительность означает перепутанные аргументы.
    """
    if not trace["stack"] or trace["stack"][-1] != span_id:
        raise ValueError(f"span {span_id} is not the innermost open span")
    span = next(s for s in trace["spans"] if s["span_id"] == span_id)
    if end_ns < span["start_ns"]:
        raise ValueError(f"end_ns {end_ns} precedes start_ns {span['start_ns']}")
    span["end_ns"] = end_ns
    span["duration_ns"] = end_ns - span["start_ns"]
    trace["stack"].pop()
    return span


def span_tree(trace):
    """Собрать дерево из плоского списка спанов: [{"span": ..., "children": [...]}].

    Корни — спаны без родителя, а также спаны, чей родитель пришёл из
    другого процесса (remote_parent из traceparent): межпроцессный трейс
    обязан выглядеть одним деревом, а не набором обрывков.

    Порядок детей — порядок открытия спанов.

    Ловушки. Первая: незакрытые спаны — ValueError; дерево из середины
    прогона врёт про длительности. Вторая: parent_id, которого нет ни среди
    спанов, ни в remote_parent, — тоже ValueError; это «orphaned tool span»
    из урока, и молча превращать его в корень нельзя, иначе поломка
    контекста никогда не обнаружится.
    """
    if trace["stack"]:
        raise ValueError(f"unfinished spans: {trace['stack']}")

    known = {s["span_id"] for s in trace["spans"]}
    remote_parent = trace.get("remote_parent")
    nodes = {s["span_id"]: {"span": s, "children": []} for s in trace["spans"]}

    roots = []
    for span in trace["spans"]:
        parent = span["parent_id"]
        if parent is None or parent == remote_parent:
            roots.append(nodes[span["span_id"]])
        elif parent in known:
            nodes[parent]["children"].append(nodes[span["span_id"]])
        else:
            raise ValueError(f"orphaned span {span['span_id']}: no parent {parent}")
    return roots


def capture_content(store, span, messages, mode="off"):
    """Контракт content capture: по умолчанию содержимое НЕ попадает в спан.

    Возвращает ссылку (str) в режиме "reference", иначе None.

    capture_content({}, span, ["secret"])                      ->  None,
        и в span["attributes"] не появилось ничего
    capture_content({}, span, ["hi"], mode="inline")           ->  None,
        span["attributes"]["gen_ai.input.messages"] == ["hi"]
    capture_content(store, span, ["secret"], mode="reference") ->  'content-1',
        store["content-1"] == ["secret"], а на спане только ссылка

    Ловушка и смысл: в продовом режиме "reference" на спане не должно быть
    самого текста — ни в одном атрибуте. Трейсы читает вся дежурная смена,
    а в промптах лежат PII и секреты. Содержимое уходит во внешнее
    хранилище, на спане — только идентификатор строки.

    Ссылки нумеруются от размера store, поэтому одинаковые сообщения из
    двух спанов не перезаписывают друг друга.
    """
    if mode not in CONTENT_MODES:
        raise ValueError(f"unknown content capture mode: {mode}")
    if mode == "off":
        return None
    if mode == "inline":
        span["attributes"]["gen_ai.input.messages"] = list(messages)
        return None
    ref = f"content-{len(store) + 1}"
    store[ref] = list(messages)
    span["attributes"]["gen_ai.input.messages_ref"] = ref
    return ref
