"""
Транспорты MCP: stdio, Streamable HTTP, SSE — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Транспорт — это всё, что лежит между «у меня есть словарь» и «словарь
оказался у собеседника». Здесь мы пишем руками обе рамки: построчную для
stdio и HTTP+SSE для удалённого варианта. Соответствие настоящему API:

    split_stdio      <-  буфер внутри stdio_client из mcp-python-sdk
    new_session_id   <-  secrets.token_hex(16) в StreamableHTTPSessionManager
    origin_allowed   <-  проверка Origin в TransportSecurityMiddleware
    sse_event        <-  сериализатор кадров EventSourceResponse
    parse_sse        <-  разбор потока на стороне клиента (httpx-sse)
    replay_after     <-  EventStore.replay_events_after (last-event-id)
    detect_transport <-  проба «старый сервер или новый» в клиенте
    handle_http      <-  ASGI-ручка POST/GET/DELETE на /mcp

Сокетов нет: HTTP здесь — это (метод, путь, заголовки, тело) на входе и
(статус, заголовки, тело) на выходе. Ровно то, что видит ручка.
"""

import json

JSONRPC = "2.0"

# Спецификация требует идентификатор сессии не короче 128 бит.
MIN_SESSION_BITS = 128

# Разрешённые HTTP-методы одной ручки Streamable HTTP.
ALLOWED_METHODS = ("POST", "GET", "DELETE")


def split_stdio(buffer, chunk):
    """Нарезать поток stdio на сообщения. Вернуть (сообщения, остаток).

    Формат stdio — одна строка = один JSON-объект, разделитель "\\n".
    Никаких Content-Length: длину знает перевод строки, и только он.

    split_stdio("", '{"id":1}\\n{"id":2}\\n')   ->  ([{...}, {...}], "")
    split_stdio("", '{"id":')                  ->  ([], '{"id":')
    split_stdio('{"id":', '1}\\n')              ->  ([{"id": 1}], "")

    Смысл остатка: чтение из трубы отдаёт БАЙТЫ, а не сообщения. Один
    read() легко возвращает полтора сообщения. Хвост без "\\n" — это ещё
    не сообщение, его надо донести до следующего вызова.

    Ловушка: `chunk.split("\\n")` даёт последним элементом хвост, и он
    почти всегда неполный. Обработаешь его как строку — потеряешь
    сообщение или получишь ValueError на ровном месте.
    """
    data = buffer + chunk
    parts = data.split("\n")
    # последний кусок — то, что осталось после последнего "\n"; он неполный
    rest = parts.pop()
    messages = [json.loads(line) for line in parts if line.strip()]
    return messages, rest


def new_session_id(rng, bits=MIN_SESSION_BITS):
    """Случайный идентификатор сессии в hex. Источник случайности — параметр.

    new_session_id(random.Random(0))  ->  строка из 32 hex-символов
    Один и тот же seed даёт одну и ту же последовательность — тест
    воспроизводим, а в бою на это место ставится secrets.token_hex.

    Спецификация требует не меньше 128 бит: идентификатор сессии — это по
    сути bearer-токен, угадать его не должно быть можно. Меньше 128 — сразу
    ValueError, потому что «почти случайный» тут хуже, чем никакой.

    Ловушка: id выдаёт СЕРВЕР. Принять идентификатор, предложенный
    клиентом, — значит позволить ему сесть в чужую сессию.
    """
    if bits < MIN_SESSION_BITS or bits % 4:
        raise ValueError(f"session id must be at least {MIN_SESSION_BITS} bits")
    # ширина в hex-символах: 4 бита на символ, ведущие нули значимы
    return format(rng.getrandbits(bits), "0{}x".format(bits // 4))


def origin_allowed(origin, allowlist):
    """Пускать ли запрос с таким Origin. Поддержан шаблон вида https://*.example.com.

    origin_allowed("http://localhost", ["http://localhost"])          ->  True
    origin_allowed("http://evil.example", ["http://localhost"])       ->  False
    origin_allowed("https://app.example.com", ["https://*.example.com"])  ->  True
    origin_allowed(None, ["http://localhost"])                        ->  True

    Зачем: браузер честно поставит Origin: http://evil.com на запрос к
    твоему localhost:1234/mcp, и same-origin policy тебя не спасёт —
    запрос-то кросс-доменный и разрешённый. Единственная защита — список.

    Origin отсутствует — значит запрос не из браузера (curl, SDK), и
    подделывать заголовок незачем: пускаем.

    Ловушки:
      * "https://evil.example.com.attacker.net" ЗАКАНЧИВАЕТСЯ не на
        ".example.com" — наивная проверка `".example.com" in origin`
        пропустит этого гостя;
      * шаблон "*.example.com" НЕ покрывает голый "example.com";
      * схема тоже сравнивается: http вместо https — другой origin.
    """
    if not origin:
        return True
    for pattern in allowlist:
        if pattern == origin:
            return True
        if "://" not in pattern:
            continue
        scheme, host_pattern = pattern.split("://", 1)
        if not host_pattern.startswith("*."):
            continue
        prefix = scheme + "://"
        if not origin.startswith(prefix):
            continue
        host = origin[len(prefix):]
        suffix = host_pattern[1:]  # "*.example.com" -> ".example.com"
        # endswith, а не in: суффикс обязан быть именно КОНЦОМ имени,
        # и перед ним обязана быть хотя бы одна метка
        if host.endswith(suffix) and len(host) > len(suffix):
            return True
    return False


def sse_event(data, event_id=None, event=None):
    """Собрать один кадр Server-Sent Events.

    sse_event('{"a":1}', 7)
        ->  'id: 7\\ndata: {"a":1}\\n\\n'
    sse_event("hello", event="ping")
        ->  'event: ping\\ndata: hello\\n\\n'
    sse_event("first\\nsecond")
        ->  'data: first\\ndata: second\\n\\n'

    Кадр заканчивается ПУСТОЙ строкой — именно она говорит получателю
    «событие целиком». Забудешь второй "\\n" — клиент будет ждать
    продолжения вечно.

    Ловушка: перевод строки внутри data нельзя оставить как есть, он
    оборвёт кадр. Каждая строка данных получает собственный префикс "data:".
    """
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event is not None:
        lines.append(f"event: {event}")
    for line in str(data).split("\n"):
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def parse_sse(text):
    """Разобрать поток SSE в список событий {"id", "event", "data"}.

    parse_sse('id: 7\\ndata: {"a":1}\\n\\n')
        ->  [{"id": "7", "event": None, "data": '{"a":1}'}]
    parse_sse(': keepalive\\n\\ndata: hi\\n\\n')
        ->  [{"id": None, "event": None, "data": "hi"}]

    Строка, начинающаяся с двоеточия, — комментарий: так шлют keepalive,
    чтобы прокси не убил простаивающее соединение. Событием она не является.

    Многострочная data склеивается обратно через "\\n": это ровно обратная
    операция к sse_event.

    Ловушка: id приходит СТРОКОЙ. Сравнивать его с числом бесполезно.
    """
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event = {"id": None, "event": None, "data": None}
        data_lines = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue  # комментарий/keepalive
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "data":
                data_lines.append(value)
            elif field in ("id", "event"):
                event[field] = value
        if not data_lines:
            continue  # блок без data — не событие
        event["data"] = "\n".join(data_lines)
        events.append(event)
    return events


def replay_after(events, last_event_id):
    """События, которые клиент пропустил, пока соединение лежало.

    events — список словарей с ключом "id" в порядке появления.

    replay_after(evts, None)   ->  все события
    replay_after(evts, "2")    ->  всё, что после события с id "2"
    replay_after(evts, "999")  ->  все события (такого id мы не помним)

    Идентификатор ПОСЛЕДНЕГО полученного, а не первого потерянного:
    заголовок так и называется, last-event-id. Отдать событие с этим id
    ещё раз — дубль в контексте модели.

    Неизвестный id — не повод молчать: отдаём всё и полагаемся на то, что
    клиент отбросит уже виденное по id.
    """
    if last_event_id is None:
        return list(events)
    for index, event in enumerate(events):
        if str(event.get("id")) == str(last_event_id):
            return list(events[index + 1:])
    return list(events)


def detect_transport(response):
    """Определить транспорт удалённого сервера по ответу на пробный POST.

    response — {"status": 200, "headers": {"Content-Type": ...}}.

    detect_transport({"status": 200, "headers": {"Content-Type": "application/json"}})
        ->  "streamable-http"
    detect_transport({"status": 200,
                      "headers": {"Content-Type": "text/event-stream",
                                  "Location": "/messages"}})
        ->  "http-sse-legacy"
    detect_transport({"status": 404, "headers": {}})
        ->  "unsupported"

    Старый двухэндпойнтный режим выдаёт себя связкой «SSE + Location»:
    сервер отвечает потоком и тут же говорит, куда слать POST-ы. Новый
    Streamable HTTP обходится одним адресом и Location не присылает.

    Ловушка: заголовки HTTP регистронезависимы. "content-type" и
    "Content-Type" — один и тот же заголовок.
    """
    if response.get("status") != 200:
        return "unsupported"
    headers = {k.lower(): v for k, v in (response.get("headers") or {}).items()}
    content_type = headers.get("content-type", "")
    is_sse = content_type.startswith("text/event-stream")
    if is_sse and "location" in headers:
        return "http-sse-legacy"
    if is_sse or content_type.startswith("application/json"):
        return "streamable-http"
    return "unsupported"


def handle_http(state, method, path, headers, body, rng):
    """Единая ручка Streamable HTTP. Вернуть (статус, заголовки, тело).

    state — {"endpoint": "/mcp", "allowlist": [...], "sessions": {},
             "handler": функция сообщение -> ответ или None}.

    handle_http(st, "POST", "/mcp", {"Origin": "http://localhost"}, msg, rng)
        ->  (200, {"Mcp-Session-Id": "<hex>", ...}, <ответ JSON-RPC>)
    handle_http(st, "GET", "/mcp", {"Mcp-Session-Id": sid}, None, rng)
        ->  (200, {"Content-Type": "text/event-stream", ...}, None)
    handle_http(st, "DELETE", "/mcp", {"Mcp-Session-Id": sid}, None, rng)
        ->  (204, {}, None)

    Порядок проверок важен и он такой:
      1. не наш путь            -> 404, сессию не заводим;
      2. Origin вне списка      -> 403, сессию не заводим;
      3. неизвестный session id -> 404: сессию отозвали, клиент обязан
         заново пройти initialize;
      4. метод не из ALLOWED_METHODS -> 405 с заголовком Allow.

    POST без Mcp-Session-Id — это первый запрос: сервер выдаёт новый id и
    возвращает его заголовком. POST с известным id новый НЕ выдаёт.

    Ловушки:
      * нотификация (ответа нет) — это 202 Accepted с пустым телом, а не
        200 с "result": null;
      * заголовки регистронезависимы, "mcp-session-id" тоже валиден.
    """
    incoming = {k.lower(): v for k, v in (headers or {}).items()}

    if path != state["endpoint"]:
        return 404, {}, None
    if not origin_allowed(incoming.get("origin"), state["allowlist"]):
        return 403, {}, {"error": "origin not allowed"}
    if method not in ALLOWED_METHODS:
        return 405, {"Allow": ", ".join(ALLOWED_METHODS)}, None

    sessions = state["sessions"]
    session_id = incoming.get("mcp-session-id")
    if session_id is not None and session_id not in sessions:
        # сессия отозвана или выдумана клиентом: только заново
        return 404, {}, None

    if method == "DELETE":
        if session_id is None:
            return 400, {}, {"error": "missing Mcp-Session-Id"}
        del sessions[session_id]
        return 204, {}, None

    if method == "GET":
        if session_id is None:
            return 400, {}, {"error": "missing Mcp-Session-Id"}
        return 200, {"Content-Type": "text/event-stream",
                     "Mcp-Session-Id": session_id}, None

    # POST
    if session_id is None:
        session_id = new_session_id(rng)
        sessions[session_id] = {"events": []}
    result = state["handler"](body)
    out_headers = {"Mcp-Session-Id": session_id}
    if result is None:
        # нотификация: принято, отвечать нечем
        return 202, out_headers, None
    out_headers["Content-Type"] = "application/json"
    return 200, out_headers, result
