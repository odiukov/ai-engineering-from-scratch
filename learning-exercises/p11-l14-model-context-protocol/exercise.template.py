"""
Model Context Protocol

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l14-model-context-protocol
Разбор:  /check-code p11-l14-model-context-protocol
"""

PROTOCOL_VERSION = "2025-06-18"
JSONRPC_VERSION = "2.0"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
JSON_TYPES = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def make_request(method, params=None, request_id=None):
    """Собрать сообщение JSON-RPC 2.0.

    make_request("tools/list", request_id=1)
        ->  {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
    make_request("tools/call", {"name": "add"}, 2)
        ->  {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "add"}, "id": 2}
    make_request("notifications/initialized")
        ->  {"jsonrpc": "2.0", "method": "notifications/initialized"}

    request_id=None означает НОТИФИКАЦИЮ: ключа "id" в сообщении нет вовсе,
    и ответа на него сервер не шлёт.

    Ловушка: id=0 — совершенно нормальный идентификатор. Отличать
    нотификацию проверкой `if not request_id` нельзя, только `is None`.
    """
    raise NotImplementedError


def make_response(request_id, result):
    """Успешный ответ JSON-RPC.

    make_response(1, {"ok": True})
        ->  {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

    Ответ обязан нести ТОТ ЖЕ id, что и запрос: клиент по нему сопоставляет
    ответ с ожиданием. В одном соединении может лететь десяток запросов, и
    порядок ответов ничем не гарантирован.
    """
    raise NotImplementedError


def make_error(request_id, code, message, data=None):
    """Ответ-ошибка JSON-RPC.

    make_error(7, METHOD_NOT_FOUND, "Method not found")
        ->  {"jsonrpc": "2.0", "id": 7, "error": {"code": -32601, "message": "Method not found"}}

    В сообщении ровно один из ключей: либо "result", либо "error", никогда оба.
    Ключ "data" добавляется только если его передали.
    """
    raise NotImplementedError


def tool_schema(name, description, properties, required=()):
    """Описание инструмента ровно в том виде, в каком его отдаёт tools/list.

    tool_schema("add", "Add two integers",
                {"a": {"type": "integer"}, "b": {"type": "integer"}}, ("a", "b"))
        ->  {"name": "add", "description": "Add two integers",
             "inputSchema": {"type": "object",
                             "properties": {"a": {"type": "integer"},
                                            "b": {"type": "integer"}},
                             "required": ["a", "b"]}}

    В настоящем SDK это генерируется из аннотаций типов функции. Здесь пишем
    руками — заодно видно, что модель получает именно текст description, и
    он пишется для выбора инструмента моделью, а не для документации.

    Ловушка: properties и required нужно скопировать. Если сложить в схему
    ссылку на словарь вызывающего кода, его последующая правка тихо поменяет
    схему уже опубликованного инструмента.
    """
    raise NotImplementedError


def validate_arguments(schema, arguments):
    """Проверить аргументы вызова по inputSchema. Вернуть список претензий.

    Пустой список означает «всё в порядке».

    validate_arguments(add_schema, {"a": 1, "b": 2})   ->  []
    validate_arguments(add_schema, {"a": 1})           ->  ["missing required property: b"]
    validate_arguments(add_schema, {"a": 1, "b": "2"}) ->  ["b: expected integer, got str"]

    Ловушка: в Python bool — подкласс int, поэтому isinstance(True, int) даёт
    True. Без отдельной проверки {"a": True} проедет как integer.
    """
    raise NotImplementedError


def call_tool(server, request_id, params):
    """Выполнить tools/call: найти инструмент, проверить аргументы, вызвать.

    server — словарь:
        {"name": "demo-server", "version": "1.0.0",
         "tools": {"add": {"schema": <tool_schema>, "handler": <функция>}}}
    params — тело вызова: {"name": "add", "arguments": {"a": 1, "b": 2}}

    call_tool(srv, 1, {"name": "add", "arguments": {"a": 1, "b": 2}})
        ->  {"jsonrpc": "2.0", "id": 1,
             "result": {"content": [{"type": "text", "text": "3"}], "isError": False}}
    call_tool(srv, 2, {"name": "nope", "arguments": {}})
        ->  ответ с "error", код -32602

    Два разных сорта неудачи, и путать их нельзя:
      * нет такого инструмента или аргументы не по схеме — это ошибка
        ПРОТОКОЛА, поле "error", код INVALID_PARAMS; чинит её разработчик;
      * обработчик бросил исключение — это ошибка ИСПОЛНЕНИЯ, она приходит
        обычным "result" с флагом isError, и ДОЛЖНА дойти до модели, чтобы
        та могла попробовать иначе.

    Ловушка: упавший инструмент не имеет права уронить сервер. Любое
    исключение обработчика ловится и превращается в текстовый content.
    """
    raise NotImplementedError


def handle(server, request):
    """Роутер сервера: обработать одно сообщение. Ответ или None.

    Поддерживаемые методы: initialize, tools/list, tools/call.

    handle(srv, make_request("tools/list", request_id=1))
        ->  {"jsonrpc": "2.0", "id": 1, "result": {"tools": [<схема add>]}}
    handle(srv, make_request("notifications/initialized"))
        ->  None
    handle(srv, make_request("tools/delete", request_id=9))
        ->  ответ с "error", код -32601

    Ловушка: нотификация не порождает ответа НИКОГДА — даже когда метод
    неизвестен и ответить очень хочется. Отправишь ответ на нотификацию —
    клиент получит сообщение с id, которого не ждал.
    """
    raise NotImplementedError


def handle_batch(server, requests):
    """Обработать пачку сообщений. Вернуть список ответов без нотификаций.

    Если в пачке одни нотификации — вернуть пустой список, а не список из
    None. Транспорт в этом случае не шлёт клиенту ничего.

    handle_batch(srv, [ping_request, initialized_notification, list_request])
        ->  [<ответ на ping>, <ответ на list>]

    Порядок ответов совпадает с порядком запросов, но клиент на это
    полагаться не должен — именно поэтому в ответе есть id.
    """
    raise NotImplementedError
