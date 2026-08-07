"""
Свой MCP-клиент: хендшейк, сессии, общее пространство имён

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l08-building-an-mcp-client
Разбор:  /check-code p13-l08-building-an-mcp-client
"""

JSONRPC = "2.0"
PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26")
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
