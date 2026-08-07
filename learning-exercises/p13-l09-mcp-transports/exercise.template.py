"""
Транспорты MCP: stdio, Streamable HTTP, SSE

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l09-mcp-transports
Разбор:  /check-code p13-l09-mcp-transports
"""

import json

JSONRPC = "2.0"
MIN_SESSION_BITS = 128
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
