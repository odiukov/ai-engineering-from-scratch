"""
Свой MCP-сервер: реестры, content-блоки, диспетчер

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p13-l07-building-an-mcp-server
Разбор:  /check-code p13-l07-building-an-mcp-server
"""

import json

PROTOCOL_VERSION = "2025-11-25"
JSONRPC = "2.0"
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
