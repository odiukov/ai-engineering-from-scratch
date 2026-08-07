"""
Основы MCP: примитивы, жизненный цикл, JSON-RPC

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l06-mcp-fundamentals
Разбор:  /check-code p13-l06-mcp-fundamentals
"""

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2025-11-25"
SERVER_PRIMITIVES = ("tools", "resources", "prompts")
CLIENT_PRIMITIVES = ("roots", "sampling", "elicitation")
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
