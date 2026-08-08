"""
Model Context Protocol — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

MCP — это не магия, а JSON-RPC 2.0 с тремя примитивами. Пакет `mcp` и
`FastMCP` скрывают кодек и роутер за декораторами; здесь мы пишем их руками.
Соответствие настоящему SDK:

    make_request/make_response/make_error  <-  кодек JSON-RPC внутри mcp
    tool_schema                            <-  @mcp.tool() и type hints
    validate_arguments                     <-  проверка inputSchema на сервере
    handle                                 <-  диспетчер методов сервера
    handle_batch                           <-  разбор пачки сообщений транспорта

Сеть не нужна: транспорт (stdio, streamable HTTP) — это то, что доставляет
словари. Мы работаем сразу со словарями.
"""

# Ревизия спецификации, которую отдаём в ответе на initialize.
PROTOCOL_VERSION = "2025-06-18"

JSONRPC_VERSION = "2.0"

# Коды ошибок JSON-RPC 2.0. Числа не наши, они из спецификации, менять нельзя.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Соответствие типов JSON Schema питоновским. "number" принимает и int, и float.
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
    message = {"jsonrpc": JSONRPC_VERSION, "method": method}
    # params опускаем, а не шлём null: спецификация разрешает отсутствие
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    return message


def make_response(request_id, result):
    """Успешный ответ JSON-RPC.

    make_response(1, {"ok": True})
        ->  {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

    Ответ обязан нести ТОТ ЖЕ id, что и запрос: клиент по нему сопоставляет
    ответ с ожиданием. В одном соединении может лететь десяток запросов, и
    порядок ответов ничем не гарантирован.
    """
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def make_error(request_id, code, message, data=None):
    """Ответ-ошибка JSON-RPC.

    make_error(7, METHOD_NOT_FOUND, "Method not found")
        ->  {"jsonrpc": "2.0", "id": 7, "error": {"code": -32601, "message": "Method not found"}}

    В сообщении ровно один из ключей: либо "result", либо "error", никогда оба.
    Ключ "data" добавляется только если его передали.
    """
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": err}


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
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": dict(properties),
            # required приходит кортежем, а в JSON нужен массив
            "required": list(required),
        },
    }


def validate_arguments(schema, arguments):
    """Проверить аргументы вызова по inputSchema. Вернуть список претензий.

    Пустой список означает «всё в порядке».

    validate_arguments(add_schema, {"a": 1, "b": 2})   ->  []
    validate_arguments(add_schema, {"a": 1})           ->  ["missing required property: b"]
    validate_arguments(add_schema, {"a": 1, "b": "2"}) ->  ["b: expected integer, got str"]

    Ловушка: в Python bool — подкласс int, поэтому isinstance(True, int) даёт
    True. Без отдельной проверки {"a": True} проедет как integer.
    """
    props = schema["inputSchema"]["properties"]
    problems = [
        f"missing required property: {name}"
        for name in schema["inputSchema"]["required"]
        if name not in arguments
    ]
    for name, value in arguments.items():
        spec = props.get(name)
        if spec is None:
            problems.append(f"unknown property: {name}")
            continue
        expected = spec["type"]
        allowed = JSON_TYPES[expected]
        # bool отсекаем руками: True прошёл бы как integer и как number
        ok = isinstance(value, allowed) and not (
            expected in ("integer", "number") and isinstance(value, bool)
        )
        if not ok:
            problems.append(f"{name}: expected {expected}, got {type(value).__name__}")
    return problems


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
    name = params.get("name")
    tool = server["tools"].get(name)
    if tool is None:
        return make_error(request_id, INVALID_PARAMS, f"Unknown tool: {name}")

    arguments = params.get("arguments") or {}
    problems = validate_arguments(tool["schema"], arguments)
    if problems:
        return make_error(request_id, INVALID_PARAMS, "; ".join(problems))

    try:
        value = tool["handler"](**arguments)
    except Exception as exc:  # ловим всё: падение инструмента не роняет сервер
        return make_response(
            request_id,
            {
                "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
                "isError": True,
            },
        )
    return make_response(
        request_id,
        {"content": [{"type": "text", "text": str(value)}], "isError": False},
    )


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
    is_notification = "id" not in request
    request_id = request.get("id")
    method = request.get("method")

    if request.get("jsonrpc") != JSONRPC_VERSION:
        response = make_error(request_id, INVALID_REQUEST, "Invalid JSON-RPC version")
    elif method == "initialize":
        params = request.get("params")
        required = ("protocolVersion", "capabilities", "clientInfo")
        missing = [name for name in required if not isinstance(params, dict) or name not in params]
        if missing:
            response = make_error(
                request_id,
                INVALID_PARAMS,
                f"Missing initialize params: {', '.join(missing)}",
            )
        elif not isinstance(params["protocolVersion"], str):
            response = make_error(request_id, INVALID_PARAMS, "protocolVersion must be a string")
        elif not isinstance(params["capabilities"], dict):
            response = make_error(request_id, INVALID_PARAMS, "capabilities must be an object")
        elif not isinstance(params["clientInfo"], dict):
            response = make_error(request_id, INVALID_PARAMS, "clientInfo must be an object")
        else:
            response = make_response(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {"name": server["name"], "version": server["version"]},
                    "capabilities": {"tools": {}},
                },
            )
    elif method == "tools/list":
        # отдаём только схемы: handler — питоновская функция, её не сериализовать
        response = make_response(
            request_id, {"tools": [t["schema"] for t in server["tools"].values()]}
        )
    elif method == "tools/call":
        response = call_tool(server, request_id, request.get("params") or {})
    else:
        response = make_error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

    return None if is_notification else response


def handle_batch(server, requests):
    """Обработать пачку сообщений. Вернуть список ответов без нотификаций.

    Если в пачке одни нотификации — вернуть пустой список, а не список из
    None. Транспорт в этом случае не шлёт клиенту ничего.

    handle_batch(srv, [ping_request, initialized_notification, list_request])
        ->  [<ответ на ping>, <ответ на list>]

    Порядок ответов совпадает с порядком запросов, но клиент на это
    полагаться не должен — именно поэтому в ответе есть id.
    """
    return [r for r in (handle(server, req) for req in requests) if r is not None]
