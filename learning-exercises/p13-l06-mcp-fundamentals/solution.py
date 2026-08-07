"""
Основы MCP: примитивы, жизненный цикл, JSON-RPC — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

MCP — это шесть примитивов, три фазы жизненного цикла и JSON-RPC 2.0 на
проводе. SDK (`mcp`, FastMCP) прячет и кодек, и согласование возможностей;
здесь мы собираем разбор транскрипта руками. Соответствие спецификации
2025-11-25:

    classify_message     <-  разделение request / response / notification
    primitive_of         <-  к какому примитиву относится метод
    owner_of             <-  кто примитив предоставляет: сервер или клиент
    negotiated_features  <-  результат обмена capabilities в initialize
    is_permitted         <-  что запрещено слать без объявленной возможности
    pair_messages        <-  сопоставление ответов запросам по id
    trace                <-  разметка транскрипта по фазам и примитивам
    transcript_stats     <-  сколько трафика уходит на цикл, а сколько на дело

Сети и транспорта нет: stdio и Streamable HTTP только доставляют словари,
а мы работаем сразу со словарями.
"""

JSONRPC_VERSION = "2.0"

PROTOCOL_VERSION = "2025-11-25"

# Три примитива сервера и три примитива клиента. Каждая возможность MCP
# принадлежит ровно одному из шести.
SERVER_PRIMITIVES = ("tools", "resources", "prompts")
CLIENT_PRIMITIVES = ("roots", "sampling", "elicitation")

# Методы фазы initialize. Всё остальное — фаза operation.
LIFECYCLE_METHODS = ("initialize", "notifications/initialized")


def classify_message(message):
    """Что это за сообщение JSON-RPC 2.0.

    Возвращает "request", "response", "error", "notification" или "invalid".

    classify_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        ->  "request"
    classify_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        ->  "notification"
    classify_message({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601,
                                                           "message": "..."}})
        ->  "error"

    Ловушка: id=0 — совершенно нормальный идентификатор. Нотификация
    отличается ОТСУТСТВИЕМ ключа "id", а не его ложностью: проверка
    `if not message.get("id")` объявит нотификацией первый же запрос сессии.

    Вторая ловушка: в ответе ровно один из ключей "result" и "error". Оба
    сразу — невалидное сообщение, а не «ну там же есть result».
    """
    if not isinstance(message, dict) or message.get("jsonrpc") != JSONRPC_VERSION:
        return "invalid"
    has_id = "id" in message
    has_method = "method" in message
    has_result = "result" in message
    has_error = "error" in message

    if has_method:
        # запрос или нотификация; result/error в них лежать не может
        if has_result or has_error:
            return "invalid"
        return "request" if has_id else "notification"
    if has_result and has_error:
        return "invalid"
    if not has_id:
        return "invalid"
    if has_result:
        return "response"
    if has_error:
        return "error"
    return "invalid"


def primitive_of(method):
    """К какому примитиву относится метод. None — если примитив незнакомый.

    primitive_of("tools/call")                    ->  "tools"
    primitive_of("resources/subscribe")           ->  "resources"
    primitive_of("sampling/createMessage")        ->  "sampling"
    primitive_of("notifications/tools/list_changed")  ->  "tools"
    primitive_of("initialize")                    ->  "lifecycle"
    primitive_of("tools/delete")                  ->  "tools"
    primitive_of("cron/schedule")                 ->  None

    Имена методов устроены как "<примитив>/<действие>", а нотификации — как
    "notifications/<примитив>/<событие>". Разбор по первому сегменту с
    поправкой на этот префикс.

    Несуществующее ДЕЙСТВИЕ примитива не отменяет: tools/delete адресован
    роутеру tools, и именно этот роутер отвечает на него -32601. Незнакомым
    бывает только сам примитив.

    Ловушка: notifications/initialized не про примитив, а про жизненный цикл,
    хотя тоже начинается с notifications/. Обрабатывай его отдельно.
    """
    if method in LIFECYCLE_METHODS:
        return "lifecycle"
    head = method.split("/")[0]
    if head == "notifications":
        parts = method.split("/")
        head = parts[1] if len(parts) > 1 else ""
        # notifications/progress и notifications/cancelled принадлежат
        # транспортному уровню, а не какому-то одному примитиву
        if head in ("progress", "cancelled", "message"):
            return "lifecycle"
    if head in SERVER_PRIMITIVES or head in CLIENT_PRIMITIVES:
        return head
    return None


def owner_of(primitive):
    """Кто предоставляет примитив: "server", "client" или None.

    owner_of("tools")      ->  "server"
    owner_of("sampling")   ->  "client"
    owner_of("lifecycle")  ->  None   (обе стороны, ничей)

    Разделение принципиально: sampling и elicitation объявляет КЛИЕНТ, и
    вызывает их сервер, а не наоборот. Сервер без клиентского sampling не
    имеет права слать sampling/createMessage — именно это и делает клиент
    без модели по-прежнему валидным клиентом MCP.
    """
    if primitive in SERVER_PRIMITIVES:
        return "server"
    if primitive in CLIENT_PRIMITIVES:
        return "client"
    return None


def negotiated_features(client_capabilities, server_capabilities):
    """Что стороны реально могут использовать после initialize.

    Возвращает отсортированный список строк: имя примитива и, через точку,
    каждый включённый под-флаг.

    negotiated_features({"sampling": {}, "roots": {"listChanged": True}},
                        {"tools": {"listChanged": True},
                         "resources": {"subscribe": True}})
        ->  ["resources", "resources.subscribe", "roots", "roots.listChanged",
             "sampling", "tools", "tools.listChanged"]

    Клиентские возможности берутся из client_capabilities, серверные — из
    server_capabilities. Возможность, объявленная не той стороной, не
    считается: сервер, написавший себе "sampling": {}, ничего этим не
    включает.

    Под-флаг со значением False объявлен, но выключен — в список он не идёт.
    """
    features = []
    for capabilities, primitives in (
        (client_capabilities, CLIENT_PRIMITIVES),
        (server_capabilities, SERVER_PRIMITIVES),
    ):
        for name, flags in capabilities.items():
            if name not in primitives:
                continue
            features.append(name)
            for flag, enabled in (flags or {}).items():
                if enabled:
                    features.append(f"{name}.{flag}")
    return sorted(features)


def is_permitted(method, client_capabilities, server_capabilities):
    """Можно ли вообще слать этот метод после согласования возможностей.

    is_permitted("tools/call", {}, {"tools": {}})            ->  True
    is_permitted("tools/call", {}, {})                       ->  False
    is_permitted("sampling/createMessage", {"sampling": {}}, {})  ->  True
    is_permitted("resources/subscribe", {}, {"resources": {}})    ->  False
    is_permitted("initialize", {}, {})                       ->  True

    Методы жизненного цикла разрешены всегда: без них согласовывать нечего.

    Ловушка: resources/subscribe требует не просто "resources", а под-флага
    "resources.subscribe". Проверка на уровне примитива пропустит подписку
    туда, где сервер её не поддерживает, и клиент будет ждать уведомлений,
    которых не будет.

    Метод незнакомого примитива — False. Согласовывать нечего: другая
    сторона про такой примитив вообще не слышала.
    """
    primitive = primitive_of(method)
    if primitive == "lifecycle":
        return True
    if primitive is None:
        return False
    features = negotiated_features(client_capabilities, server_capabilities)
    if primitive not in features:
        return False
    action = method.split("/")[-1]
    # под-флаги совпадают по имени с действием: subscribe -> resources.subscribe
    if action == "subscribe":
        return f"{primitive}.subscribe" in features
    if action == "list_changed":
        return f"{primitive}.listChanged" in features
    return True


def pair_messages(transcript):
    """Разложить транскрипт на пары запрос-ответ, нотификации и сироты.

    Возвращает словарь:
        {"pairs": [(запрос, ответ), ...],
         "notifications": [сообщение, ...],
         "pending": [запрос, ...],      запрос без ответа
         "orphans": [ответ, ...]}       ответ без запроса

    Порядок пар — порядок ЗАПРОСОВ, а не ответов: ответы в одном соединении
    приходят вперемешку, и восстановить порядок можно только по id.

    Ловушка: id=0 обязан сопоставляться так же, как любой другой. Словарь по
    id спасает от этого сам собой, а вот поиск через `if request_id` — нет.

    Невалидные сообщения не попадают никуда: их нельзя ни спарить, ни
    посчитать нотификацией.
    """
    requests = {}
    order = []
    notifications = []
    responses = {}
    orphans = []

    for message in transcript:
        kind = classify_message(message)
        if kind == "request":
            requests[message["id"]] = message
            order.append(message["id"])
        elif kind == "notification":
            notifications.append(message)
        elif kind in ("response", "error"):
            responses[message["id"]] = message

    for message_id, response in responses.items():
        if message_id not in requests:
            orphans.append(response)

    pairs = [(requests[i], responses[i]) for i in order if i in responses]
    pending = [requests[i] for i in order if i not in responses]
    return {
        "pairs": pairs,
        "notifications": notifications,
        "pending": pending,
        "orphans": orphans,
    }


def trace(transcript):
    """Разметить каждое сообщение транскрипта. Список словарей, порядок тот же.

    Каждая запись:
        {"kind": ..., "method": <строка или None>,
         "primitive": <строка или None>, "phase": "initialize" | "operation"}

    trace([init_request, init_response, initialized_notification, list_request])
        ->  фазы ["initialize", "initialize", "initialize", "operation"]

    Фаза переключается ПОСЛЕ notifications/initialized: сама нотификация ещё
    относится к рукопожатию, а всё, что за ней, — к работе.

    Ловушка: у ответа нет поля method, и его примитив берётся у запроса с тем
    же id. Ответ, разобранный в отрыве от запроса, ничего о себе не знает —
    поэтому trace опирается на pair_messages.
    """
    method_by_id = {}
    for request, _response in pair_messages(transcript)["pairs"]:
        method_by_id[request["id"]] = request["method"]

    out = []
    phase = "initialize"
    for message in transcript:
        kind = classify_message(message)
        method = message.get("method") if isinstance(message, dict) else None
        if method is None and kind in ("response", "error"):
            method = method_by_id.get(message.get("id"))
        primitive = primitive_of(method) if method else None
        out.append({"kind": kind, "method": method, "primitive": primitive, "phase": phase})
        if method == "notifications/initialized":
            phase = "operation"
    return out


def transcript_stats(transcript):
    """Сводка по транскрипту: сколько чего и какая доля ушла на рукопожатие.

    Возвращает словарь:
        {"request": n, "response": n, "error": n, "notification": n,
         "invalid": n, "lifecycle_share": доля от 0.0 до 1.0}

    transcript_stats([])
        ->  {"request": 0, ..., "lifecycle_share": 0.0}

    lifecycle_share — доля сообщений фазы initialize от всех сообщений. На
    долгой сессии она стремится к нулю: рукопожатие платится один раз, а
    вызовы идут тысячами. На сессии из трёх вызовов рукопожатие съедает
    половину трафика — вот почему транспорт держат открытым.

    Ключи счётчиков присутствуют всегда, даже нулевые: строчку метрик надо
    парсить, а не угадывать.
    """
    counts = {kind: 0 for kind in ("request", "response", "error", "notification", "invalid")}
    marked = trace(transcript)
    for entry in marked:
        counts[entry["kind"]] += 1
    if marked:
        lifecycle = sum(1 for entry in marked if entry["phase"] == "initialize")
        counts["lifecycle_share"] = lifecycle / len(marked)
    else:
        counts["lifecycle_share"] = 0.0
    return counts
