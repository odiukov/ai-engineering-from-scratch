"""
Свой MCP-сервер: реестры, content-блоки, диспетчер — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Здесь мы собираем руками ровно то, что FastMCP и TypeScript SDK прячут за
декораторами. Соответствие настоящему API:

    annotations         <-  @app.tool(annotations={...})
    needs_confirmation  <-  логика хоста, решающая показать диалог согласия
    initialize_result   <-  автоматический capabilities из реестров сервера
    tool_content        <-  превращение возврата питоновской функции в content
    call_tool           <-  исполнение tools/call с isError вместо падения
    dispatch            <-  роутер методов сервера (dict[str, Callable] в SDK)
    serve_lines         <-  цикл stdio-транспорта: строка со stdin, строка в stdout

Сети здесь нет и не будет: транспорт — это то, что доставляет строки. Мы
работаем сразу со строками и словарями, поэтому всё воспроизводимо и
тестируется без единого процесса.

Сервер описывается обычным словарём:

    {
      "name": "notes", "version": "1.0.0", "subscribe": False,
      "tools":     {"notes_list": {"tool": <описание>, "handler": <функция>}},
      "resources": {"notes://note-1": {"name": ..., "mimeType": ..., "text": ...}},
      "prompts":   {"review_note": {"description": ..., "messages": [...]}},
    }
"""

import json

# Ревизия спецификации, которую сервер объявляет в ответе на initialize.
PROTOCOL_VERSION = "2025-11-25"

JSONRPC = "2.0"

# Коды ошибок JSON-RPC 2.0. Числа не наши, менять нельзя.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def annotations(read_only=False, destructive=False, idempotent=False, open_world=False):
    """Подсказки о безопасности инструмента в том виде, в каком их ждёт tools/list.

    annotations(read_only=True)
        ->  {"readOnlyHint": True}
    annotations(destructive=True, idempotent=True)
        ->  {"destructiveHint": True, "idempotentHint": True}
    annotations()
        ->  {}

    Соглашение спецификации: подсказка со значением False равносильна её
    отсутствию, поэтому в словарь попадают ТОЛЬКО поднятые флаги. Клиент по
    ним решает, показывать ли диалог подтверждения и можно ли повторить
    вызов после таймаута.

    Ловушка: имена полей в JSON — camelCase (`readOnlyHint`), а не
    питоновский snake_case. Клиент ищет именно их.
    """
    # порядок фиксируем, чтобы словарь сравнивался предсказуемо в тестах
    # и не прыгал в логах между запусками
    flags = (
        ("readOnlyHint", read_only),
        ("destructiveHint", destructive),
        ("idempotentHint", idempotent),
        ("openWorldHint", open_world),
    )
    return {key: True for key, value in flags if value}


def needs_confirmation(tool):
    """Спросить ли пользователя перед вызовом этого инструмента.

    needs_confirmation({"name": "notes_list",
                        "annotations": {"readOnlyHint": True}})   ->  False
    needs_confirmation({"name": "notes_delete",
                        "annotations": {"destructiveHint": True}}) ->  True
    needs_confirmation({"name": "notes_create"})                   ->  True

    Правило безопасного клиента: молчание сервера — не разрешение. Нет
    подсказок вовсе — считаем вызов опасным и спрашиваем.

    Ловушка: сервер может прислать оба флага сразу. destructiveHint сильнее
    readOnlyHint — «читает и при этом всё сносит» надо трактовать в пользу
    пользователя, а не сервера.
    """
    hints = tool.get("annotations") or {}
    if hints.get("destructiveHint"):
        return True
    # readOnlyHint снимает подтверждение; его отсутствие — нет
    return not hints.get("readOnlyHint", False)


def initialize_result(server):
    """Ответ на initialize: версия протокола, capabilities, serverInfo.

    Пустой реестр НЕ порождает capability: клиент по этому набору гейтит
    фичи, и объявить prompts без единого prompt — значит обещать то, чего нет.

    initialize_result({"name": "notes", "version": "1.0.0",
                       "tools": {"a": ...}, "resources": {}, "prompts": {}})
        ->  {"protocolVersion": "2025-11-25",
             "capabilities": {"tools": {"listChanged": True}},
             "serverInfo": {"name": "notes", "version": "1.0.0"}}

    Флаг subscribe у ресурсов берётся из server["subscribe"] и по умолчанию
    False: подписки — это отдельная работа (урок 10), обещать их зря нельзя.
    """
    capabilities = {}
    if server.get("tools"):
        capabilities["tools"] = {"listChanged": True}
    if server.get("resources"):
        capabilities["resources"] = {
            "listChanged": True,
            "subscribe": bool(server.get("subscribe", False)),
        }
    if server.get("prompts"):
        # у промптов listChanged обычно не поддерживают: набор статичен
        capabilities["prompts"] = {"listChanged": False}
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": capabilities,
        "serverInfo": {"name": server["name"], "version": server["version"]},
    }


def tool_content(value):
    """Превратить возврат питоновского обработчика в список content-блоков.

    tool_content("Found 2 notes")
        ->  [{"type": "text", "text": "Found 2 notes"}]
    tool_content({"id": "note-1"})
        ->  [{"type": "text", "text": '{"id": "note-1"}'}]
    tool_content([{"type": "text", "text": "a"}])
        ->  [{"type": "text", "text": "a"}]      (уже готовые блоки)

    Список считается готовыми блоками, только если КАЖДЫЙ его элемент —
    словарь с ключом "type". Список чего угодно другого — это данные, и он
    уезжает одним текстовым блоком через json.dumps.

    Ловушка: голую строку клиенту отдавать нельзя, поле content — всегда
    массив. Именно поэтому у SDK возврат `-> str` всё равно превращается в
    один текстовый блок.
    """
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, list) and value and all(
        isinstance(block, dict) and "type" in block for block in value
    ):
        # копируем блоки: реестр сервера не должен уехать наружу по ссылке
        return [dict(block) for block in value]
    # sort_keys ради воспроизводимости, ensure_ascii=False ради кириллицы
    return [{"type": "text", "text": json.dumps(value, ensure_ascii=False, sort_keys=True)}]


def call_tool(registry, name, arguments):
    """Выполнить инструмент. Вернуть тело result для tools/call.

    registry — {имя: {"tool": <описание>, "handler": <функция>}}.

    call_tool(reg, "echo", {"text": "hi"})
        ->  {"content": [{"type": "text", "text": "hi"}], "isError": False}
    call_tool(reg, "boom", {})
        ->  {"content": [{"type": "text", "text": "ValueError: ..."}], "isError": True}
    call_tool(reg, "no-such-tool", {})
        ->  KeyError

    Две разные неудачи, и путать их нельзя:
      * инструмента нет — это ошибка ПРОТОКОЛА, её должен увидеть
        разработчик клиента, поэтому наружу летит KeyError, а диспетчер
        превратит его в JSON-RPC error с кодом -32602;
      * обработчик упал — это ошибка ИСПОЛНЕНИЯ, она приходит обычным
        result с isError: True и ДОЛЖНА дойти до модели, чтобы та могла
        попробовать иначе.

    Ловушка: упавший инструмент не имеет права уронить сервер. Любое
    исключение обработчика ловится и становится текстовым блоком.
    """
    entry = registry[name]  # KeyError наружу — это осознанно
    try:
        value = entry["handler"](**(arguments or {}))
    except Exception as exc:  # noqa: BLE001 — падение инструмента не роняет сервер
        return {
            "content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}],
            "isError": True,
        }
    return {"content": tool_content(value), "isError": False}


def dispatch(server, message):
    """Роутер сервера: одно входящее сообщение -> ответ или None.

    Поддержаны initialize, ping, tools/list, tools/call, resources/list,
    resources/read, prompts/list, prompts/get.

    dispatch(srv, {"jsonrpc": "2.0", "id": 1, "method": "ping"})
        ->  {"jsonrpc": "2.0", "id": 1, "result": {}}
    dispatch(srv, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        ->  None
    dispatch(srv, {"jsonrpc": "2.0", "id": 9, "method": "tools/delete"})
        ->  ответ с "error", код -32601

    Метод примитива, которого у сервера нет, — это -32601, а не пустой
    список: сервер не объявлял capability, значит метода для клиента не
    существует.

    Ловушки:
      * нотификация (нет ключа "id") не порождает ответа НИКОГДА, даже
        когда метод неизвестен и ответить очень хочется;
      * id=0 — законный идентификатор, отличать нотификацию проверкой
        `if not message.get("id")` нельзя, только по наличию ключа.
    """
    is_notification = "id" not in message
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    def ok(result):
        return {"jsonrpc": JSONRPC, "id": request_id, "result": result}

    def fail(code, text):
        return {"jsonrpc": JSONRPC, "id": request_id,
                "error": {"code": code, "message": text}}

    if message.get("jsonrpc") != JSONRPC:
        response = fail(INVALID_REQUEST, "Invalid JSON-RPC version")
    elif method == "initialize":
        response = ok(initialize_result(server))
    elif method == "ping":
        response = ok({})
    elif method == "tools/list" and server.get("tools"):
        # наружу уходят только описания: handler — питоновская функция,
        # её не сериализовать
        response = ok({"tools": [e["tool"] for e in server["tools"].values()]})
    elif method == "tools/call" and server.get("tools"):
        try:
            response = ok(call_tool(server["tools"], params.get("name"), params.get("arguments")))
        except KeyError:
            response = fail(INVALID_PARAMS, f"Unknown tool: {params.get('name')}")
    elif method == "resources/list" and server.get("resources"):
        response = ok({"resources": [
            {"uri": uri, "name": meta["name"],
             "mimeType": meta.get("mimeType", "text/plain")}
            for uri, meta in server["resources"].items()
        ]})
    elif method == "resources/read" and server.get("resources"):
        meta = server["resources"].get(params.get("uri"))
        if meta is None:
            response = fail(INVALID_PARAMS, f"Unknown resource: {params.get('uri')}")
        else:
            response = ok({"contents": [{
                "uri": params["uri"],
                "mimeType": meta.get("mimeType", "text/plain"),
                "text": meta["text"],
            }]})
    elif method == "prompts/list" and server.get("prompts"):
        response = ok({"prompts": [
            {"name": name, "description": p["description"],
             "arguments": p.get("arguments", [])}
            for name, p in server["prompts"].items()
        ]})
    elif method == "prompts/get" and server.get("prompts"):
        prompt = server["prompts"].get(params.get("name"))
        if prompt is None:
            response = fail(INVALID_PARAMS, f"Unknown prompt: {params.get('name')}")
        else:
            response = ok({"description": prompt["description"],
                           "messages": prompt["messages"]})
    else:
        response = fail(METHOD_NOT_FOUND, f"Method not found: {method}")

    return None if is_notification else response


def serve_lines(server, lines):
    """Цикл stdio-транспорта: строки со stdin -> строки для stdout.

    Одна строка — один JSON-объект, разделитель "\\n". Ни длин, ни рамок.

    serve_lines(srv, ['{"jsonrpc":"2.0","id":1,"method":"ping"}'])
        ->  ['{"jsonrpc": "2.0", "id": 1, "result": {}}']
    serve_lines(srv, ['{"jsonrpc":"2.0","method":"notifications/initialized"}'])
        ->  []
    serve_lines(srv, ['не json'])
        ->  строка с ошибкой -32700 и "id": null

    Правила транспорта:
      * пустые строки пропускаем молча, это не сообщения;
      * битый JSON — код -32700 и "id": null, потому что id прочитать было
        неоткуда;
      * нотификации не дают выходных строк, поэтому длина ответа обычно
        МЕНЬШЕ длины входа.

    Ловушка: в stdout нельзя писать ничего, кроме JSON-RPC. Один print()
    для отладки — и клиент падает на разборе. Логи идут в stderr.
    """
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            # id неизвестен, поэтому по спецификации null
            out.append(json.dumps({
                "jsonrpc": JSONRPC, "id": None,
                "error": {"code": PARSE_ERROR, "message": "Parse error"},
            }))
            continue
        response = dispatch(server, message)
        if response is not None:
            out.append(json.dumps(response))
    return out
