"""
Свой MCP-клиент: хендшейк, сессии, общее пространство имён — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Клиент — это то место, где живёт вся оркестрация: он поднимает несколько
серверов, жмёт руку каждому, склеивает их списки инструментов в один
плоский набор для модели и разводит вызовы обратно по владельцам.
Соответствие настоящему API:

    handshake_messages  <-  ClientSession.initialize() из mcp-python-sdk
    new_session         <-  сам объект ClientSession (состояние на сервер)
    supports            <-  проверка session.server_capabilities перед вызовом
    merge_tools         <-  склейка tools/list нескольких серверов в хосте
    route_call          <-  выбор сессии по имени инструмента
    drain               <-  фоновый reader-поток, разбирающий stdout сервера
    apply_notification  <-  реакция на notifications/*_changed

Процессов здесь нет: subprocess.Popen лишь доставляет словари. Мы работаем
сразу со словарями, поэтому тесты воспроизводимы и ничего не висит.

Сессия — обычный словарь:

    {"name": "notes", "protocolVersion": ..., "capabilities": {...},
     "serverInfo": {...}, "tools": [...], "pending": {id: method},
     "stale": True, "dirty": [], "alive": True}
"""

JSONRPC = "2.0"

# Ревизия, которую клиент предлагает в initialize.
PROTOCOL_VERSION = "2025-11-25"

# Ревизии, с которыми клиент умеет разговаривать. Сервер вправе ответить
# более старой из этого списка — тогда работаем по ней.
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26")

# Политики разрешения коллизий в общем пространстве имён.
POLICIES = ("prefix", "first", "reject")


def handshake_messages(request_id, client_name, version, capabilities=None):
    """Два сообщения хендшейка: запрос initialize и нотификация о готовности.

    handshake_messages(1, "host", "0.1")
        ->  [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                         "clientInfo": {"name": "host", "version": "0.1"}}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"}]

    Второе сообщение — НОТИФИКАЦИЯ: ключа "id" в нём нет вовсе, и ответа на
    неё сервер не пришлёт. Ждать ответа на неё — классический дедлок клиента.

    Ловушка: capabilities нужно скопировать. Положишь ссылку на словарь
    вызывающего — его последующая правка тихо поменяет уже отправленное
    сообщение (и то, что клиент считает объявленным).
    """
    return [
        {
            "jsonrpc": JSONRPC,
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": dict(capabilities or {}),
                "clientInfo": {"name": client_name, "version": version},
            },
        },
        # нотификация: без "id", ответа не будет
        {"jsonrpc": JSONRPC, "method": "notifications/initialized"},
    ]


def new_session(name, init_result):
    """Состояние сессии из ответа сервера на initialize.

    new_session("notes", {"protocolVersion": "2025-11-25",
                          "capabilities": {"tools": {}},
                          "serverInfo": {"name": "notes", "version": "1.0.0"}})
        ->  {"name": "notes", "protocolVersion": "2025-11-25",
             "capabilities": {"tools": {}}, "serverInfo": {...},
             "tools": [], "pending": {}, "stale": True, "dirty": [],
             "alive": True}

    Версию протокола выбирает СЕРВЕР — из тех, что мы предложили. Ответ с
    версией не из SUPPORTED_PROTOCOLS означает, что говорить не о чем:
    ValueError прямо на хендшейке, а не загадочная ошибка через десять
    вызовов.

    stale=True с самого начала: tools/list мы ещё не звали, списка нет.
    """
    protocol = init_result.get("protocolVersion")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol version: {protocol}")
    return {
        "name": name,
        "protocolVersion": protocol,
        "capabilities": dict(init_result.get("capabilities") or {}),
        "serverInfo": dict(init_result.get("serverInfo") or {}),
        "tools": [],
        # id запроса -> метод; сюда кладём отправленное и ждём ответа
        "pending": {},
        "stale": True,
        "dirty": [],
        "alive": True,
    }


def supports(session, path):
    """Объявлял ли сервер такую возможность. Путь — через точку.

    supports(s, "tools")                ->  True, если сервер объявил tools
    supports(s, "resources.subscribe")  ->  True, только если внутри True
    supports(s, "prompts")              ->  False, если ключа нет

    Пустой словарь — это ДА для самого примитива и НЕТ для любого флага
    внутри: {"tools": {}} значит «инструменты есть, listChanged не шлю».

    Зачем это: вызвать resources/subscribe у сервера, который подписки не
    объявлял, — гарантированная -32601. Дешевле спросить заранее.
    """
    node = session.get("capabilities") or {}
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    # {} на конце пути — примитив объявлен; False/None — флаг опущен
    return True if node == {} else bool(node)


def merge_tools(sessions, policy="prefix"):
    """Склеить tools нескольких сессий в одно плоское пространство имён.

    Возвращает {видимое имя: {"server": имя сервера, "tool": описание}} в
    порядке обхода.

    merge_tools([notes, files])            ->  {"notes_list": ..., "search": ...}
    merge_tools([notes, files], "prefix")  ->  второй search станет "files/search"
    merge_tools([notes, files], "first")   ->  второй search просто выпадет
    merge_tools([notes, files], "reject")  ->  ValueError

    Три политики — это три реальных хоста: Claude Desktop и VS Code
    префиксуют, Cursor отказывается грузить второй сервер.

    Мёртвые сессии (alive=False) в набор не попадают: модели нельзя
    показывать инструмент, вызвать который невозможно.

    Ловушка: «тихий first-come» прячет коллизию. Модель зовёт search и
    попадает не туда, куда думает, — и никакой ошибки.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown collision policy: {policy}")

    namespace = {}
    for session in sessions:
        if not session.get("alive", True):
            continue
        for tool in session.get("tools", []):
            name = tool["name"]
            if name not in namespace:
                namespace[name] = {"server": session["name"], "tool": tool}
                continue
            if policy == "first":
                continue  # первый победил, второй молча выпал
            if policy == "reject":
                raise ValueError(
                    f"Tool name collision: {name} "
                    f"({namespace[name]['server']} vs {session['name']})"
                )
            alias = f"{session['name']}/{name}"
            if alias in namespace:
                raise ValueError(f"Prefixed name also collides: {alias}")
            namespace[alias] = {"server": session["name"], "tool": tool}
    return namespace


def route_call(namespace, request_id, tool_name, arguments):
    """Куда и что отправить по имени инструмента из общего набора.

    Возвращает (имя сервера, сообщение tools/call).

    route_call(ns, 5, "files/search", {"q": "mcp"})
        ->  ("files", {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                       "params": {"name": "search", "arguments": {"q": "mcp"}}})
    route_call(ns, 6, "no-such-tool", {})  ->  KeyError

    Ключевое: префикс — это выдумка КЛИЕНТА. Сервер про него ничего не
    знает, и в params["name"] обязано уехать исходное имя инструмента.
    Отправишь "files/search" — получишь -32602 «unknown tool».
    """
    entry = namespace[tool_name]  # KeyError наружу: модель назвала несуществующее
    return entry["server"], {
        "jsonrpc": JSONRPC,
        "id": request_id,
        "method": "tools/call",
        # имя из описания инструмента, а не видимый алиас
        "params": {"name": entry["tool"]["name"], "arguments": dict(arguments or {})},
    }


def drain(session, incoming):
    """Разобрать всё, что прочитал фоновый reader, по четырём корзинам.

    incoming — список разобранных сообщений; None означает EOF (сервер
    закрыл stdout или умер).

    Возвращает {"responses": {id: сообщение}, "notifications": [...],
                "server_requests": [...], "unmatched": [...]}.

    Сессия правится на месте: сматченные id уходят из pending, EOF гасит
    alive.

    Как различать (три разных сорта сообщений, и путать их дорого):
      * есть "method" и есть "id"  -> ЗАПРОС СЕРВЕРА к нам
        (sampling/createMessage, roots/list, elicitation/create) — на него
        надо ответить;
      * есть "method", "id" нет     -> нотификация, отвечать НЕЛЬЗЯ;
      * "method" нет, есть "id"     -> ответ на наш запрос.

    Ловушка: всё, что пришло после EOF, существовать не может. Читать
    дальше — значит обрабатывать мусор из чужого буфера.
    """
    result = {"responses": {}, "notifications": [], "server_requests": [], "unmatched": []}
    for message in incoming:
        if message is None:
            session["alive"] = False
            break  # после EOF ничего не бывает
        if "method" in message:
            if "id" in message:
                result["server_requests"].append(message)
            else:
                result["notifications"].append(message)
            continue
        request_id = message.get("id")
        if request_id in session.get("pending", {}):
            session["pending"].pop(request_id)
            result["responses"][request_id] = message
        else:
            # ответ на запрос, которого мы не помним: чужой, повторный
            # или пришедший после таймаута
            result["unmatched"].append(message)
    return result


def apply_notification(session, notification):
    """Отреагировать на нотификацию сервера. Вернуть ту же сессию.

    apply_notification(s, {"jsonrpc": "2.0",
                           "method": "notifications/tools/list_changed"})
        ->  сессия с stale=True: список инструментов надо перезапросить
    apply_notification(s, {"jsonrpc": "2.0",
                           "method": "notifications/resources/updated",
                           "params": {"uri": "notes://note-1"}})
        ->  сессия с "notes://note-1" в dirty

    Незнакомая нотификация — не ошибка: сервер может знать методы новее
    нашего клиента. Молча игнорируем, состояние не трогаем.

    Ловушка: сообщение с "id" — это ЗАПРОС, а не нотификация. Обработать
    его здесь значит не ответить на него никогда, и сервер повиснет.
    Поэтому ValueError.
    """
    if "id" in notification:
        raise ValueError("Not a notification: message carries an id")

    method = notification.get("method")
    params = notification.get("params") or {}
    if method == "notifications/tools/list_changed":
        session["stale"] = True
    elif method == "notifications/resources/updated":
        uri = params.get("uri")
        # повторное уведомление о том же ресурсе не должно плодить дубли
        if uri is not None and uri not in session["dirty"]:
            session["dirty"].append(uri)
    return session
