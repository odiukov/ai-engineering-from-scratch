"""Тесты к уроку «Свой MCP-сервер». Правь exercise.py."""

import json

import pytest

from exercise import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    annotations,
    call_tool,
    dispatch,
    initialize_result,
    needs_confirmation,
    serve_lines,
    tool_content,
)

NOTES = {
    "note-1": {"title": "MCP overview", "body": "Primitives, lifecycle, JSON-RPC."},
    "note-2": {"title": "Tool schemas", "body": "Atomic beats monolithic."},
}


def _notes_list():
    return [f"{k}: {v['title']}" for k, v in sorted(NOTES.items())]


def _notes_create(title):
    if not title:
        raise ValueError("title must not be empty")
    return {"id": "note-3", "title": title}


def make_server(tools=True, resources=True, prompts=True, subscribe=False):
    """Демо-сервер. Собирается ВНУТРИ теста, чтобы сбор тестов не падал."""
    server = {
        "name": "notes",
        "version": "1.0.0",
        "subscribe": subscribe,
        "tools": {},
        "resources": {},
        "prompts": {},
    }
    if tools:
        server["tools"] = {
            "notes_list": {
                "tool": {
                    "name": "notes_list",
                    "description": "List every note.",
                    "inputSchema": {"type": "object", "properties": {}},
                    "annotations": {"readOnlyHint": True},
                },
                "handler": _notes_list,
            },
            "notes_create": {
                "tool": {
                    "name": "notes_create",
                    "description": "Create a note.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                },
                "handler": _notes_create,
            },
        }
    if resources:
        server["resources"] = {
            "notes://note-1": {
                "name": "MCP overview",
                "mimeType": "text/markdown",
                "text": "Primitives, lifecycle, JSON-RPC.",
            }
        }
    if prompts:
        server["prompts"] = {
            "review_note": {
                "description": "Review one note.",
                "arguments": [{"name": "note_id", "required": True}],
                "messages": [{"role": "user", "content": {"type": "text", "text": "Review it"}}],
            }
        }
    return server


# ------------------------------------------------------------- annotations
def test_annotations_keeps_only_raised_flags():
    assert annotations(read_only=True) == {"readOnlyHint": True}
    assert annotations(read_only=True, destructive=False) == {"readOnlyHint": True}


def test_annotations_without_arguments_is_empty():
    """Подсказка со значением False равносильна её отсутствию."""
    assert annotations() == {}


def test_annotations_uses_camel_case_wire_names():
    got = annotations(destructive=True, idempotent=True, open_world=True)
    assert got == {"destructiveHint": True, "idempotentHint": True, "openWorldHint": True}


# -------------------------------------------------------- needs_confirmation
def test_read_only_tool_needs_no_confirmation():
    assert needs_confirmation({"name": "notes_list", "annotations": {"readOnlyHint": True}}) is False


def test_destructive_tool_needs_confirmation():
    assert needs_confirmation({"name": "notes_delete", "annotations": {"destructiveHint": True}}) is True


def test_tool_without_annotations_is_treated_as_unsafe():
    """Молчание сервера — не разрешение."""
    assert needs_confirmation({"name": "notes_create"}) is True


def test_destructive_beats_read_only():
    """Сервер прислал оба флага — решаем в пользу пользователя."""
    tool = {"name": "notes_purge", "annotations": {"readOnlyHint": True, "destructiveHint": True}}
    assert needs_confirmation(tool) is True


# ------------------------------------------------------- initialize_result
def test_initialize_reports_protocol_version_and_server_info():
    result = initialize_result(make_server())
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"] == {"name": "notes", "version": "1.0.0"}


def test_empty_registry_declares_no_capability():
    """Объявить prompts без единого prompt — значит обещать несуществующее."""
    caps = initialize_result(make_server(prompts=False))["capabilities"]
    assert "prompts" not in caps and "tools" in caps
    bare = initialize_result(make_server(tools=False, resources=False, prompts=False))
    assert bare["capabilities"] == {}


def test_subscribe_is_false_unless_the_server_supports_it():
    caps = initialize_result(make_server())["capabilities"]
    assert caps["resources"]["subscribe"] is False


def test_subscribe_is_declared_when_supported():
    caps = initialize_result(make_server(subscribe=True))["capabilities"]
    assert caps["resources"]["subscribe"] is True


# ------------------------------------------------------------ tool_content
def test_string_becomes_one_text_block():
    assert tool_content("Found 2 notes") == [{"type": "text", "text": "Found 2 notes"}]


def test_dict_is_serialized_into_a_text_block():
    blocks = tool_content({"id": "note-1"})
    assert len(blocks) == 1
    assert json.loads(blocks[0]["text"]) == {"id": "note-1"}


def test_ready_blocks_pass_through():
    blocks = [{"type": "text", "text": "a"}, {"type": "image", "data": "AA==", "mimeType": "image/png"}]
    assert tool_content(blocks) == blocks


def test_ready_blocks_are_copied_not_aliased():
    """Реестр сервера не должен уехать наружу по ссылке."""
    blocks = [{"type": "text", "text": "a"}]
    out = tool_content(blocks)
    out[0]["text"] = "changed"
    assert blocks[0]["text"] == "a"


def test_list_of_plain_data_is_not_mistaken_for_blocks():
    blocks = tool_content(["note-1", "note-2"])
    assert len(blocks) == 1 and blocks[0]["type"] == "text"
    assert json.loads(blocks[0]["text"]) == ["note-1", "note-2"]


# --------------------------------------------------------------- call_tool
def test_call_tool_wraps_the_result_in_content():
    result = call_tool(make_server()["tools"], "notes_create", {"title": "hi"})
    assert result["isError"] is False
    assert json.loads(result["content"][0]["text"])["title"] == "hi"


def test_failing_handler_returns_is_error_instead_of_raising():
    """Ошибка исполнения обязана дойти до модели, а не уронить сервер."""
    result = call_tool(make_server()["tools"], "notes_create", {"title": ""})
    assert result["isError"] is True
    assert "ValueError" in result["content"][0]["text"]


def test_unknown_tool_raises_key_error():
    """Нет инструмента — это ошибка протокола, не ошибка исполнения."""
    with pytest.raises(KeyError):
        call_tool(make_server()["tools"], "notes_teleport", {})


def test_call_tool_accepts_missing_arguments_key():
    result = call_tool(make_server()["tools"], "notes_list", None)
    assert result["isError"] is False and len(result["content"]) == 1


# ---------------------------------------------------------------- dispatch
def test_ping_answers_with_the_same_id():
    response = dispatch(make_server(), {"jsonrpc": "2.0", "id": 7, "method": "ping"})
    assert response == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_notification_produces_no_response():
    """Даже на незнакомый метод: у нотификации нет id, ответ слать некуда."""
    server = make_server()
    assert dispatch(server, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert dispatch(server, {"jsonrpc": "2.0", "method": "notifications/nonsense"}) is None


def test_zero_is_a_valid_request_id():
    """`if not message.get("id")` спутает id=0 с нотификацией."""
    response = dispatch(make_server(), {"jsonrpc": "2.0", "id": 0, "method": "ping"})
    assert response is not None and response["id"] == 0


def test_unknown_method_is_method_not_found():
    response = dispatch(make_server(), {"jsonrpc": "2.0", "id": 9, "method": "tools/delete"})
    assert response["error"]["code"] == METHOD_NOT_FOUND
    assert "result" not in response


def test_undeclared_capability_is_rejected_not_answered_empty():
    """Сервер без промптов не объявлял capability — метода для клиента нет."""
    server = make_server(prompts=False)
    response = dispatch(server, {"jsonrpc": "2.0", "id": 1, "method": "prompts/list"})
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_wrong_jsonrpc_version_is_invalid_request():
    response = dispatch(make_server(), {"jsonrpc": "1.0", "id": 1, "method": "ping"})
    assert response["error"]["code"] == -32600


def test_tools_list_hides_python_handlers():
    """Наружу уходят описания; функцию-обработчик не сериализовать."""
    tools = dispatch(make_server(), {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in tools} == {"notes_list", "notes_create"}
    assert all("handler" not in t for t in tools)
    json.dumps(tools)


def test_tools_call_routes_to_the_handler():
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
           "params": {"name": "notes_list", "arguments": {}}}
    result = dispatch(make_server(), msg)["result"]
    assert result["isError"] is False


def test_unknown_target_becomes_invalid_params_not_is_error():
    """Несуществующий инструмент и несуществующий ресурс — оба -32602."""
    server = make_server()
    call = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "nope", "arguments": {}}}
    read = {"jsonrpc": "2.0", "id": 5, "method": "resources/read",
            "params": {"uri": "notes://ghost"}}
    assert dispatch(server, call)["error"]["code"] == INVALID_PARAMS
    assert dispatch(server, read)["error"]["code"] == INVALID_PARAMS


def test_resources_read_returns_contents_with_the_uri():
    msg = {"jsonrpc": "2.0", "id": 4, "method": "resources/read",
           "params": {"uri": "notes://note-1"}}
    contents = dispatch(make_server(), msg)["result"]["contents"]
    assert contents[0]["uri"] == "notes://note-1"
    assert contents[0]["mimeType"] == "text/markdown"


# ------------------------------------------------------------- serve_lines
def test_serve_lines_answers_a_request():
    out = serve_lines(make_server(), ['{"jsonrpc":"2.0","id":1,"method":"ping"}'])
    assert [json.loads(line) for line in out] == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def test_serve_lines_stays_silent_on_notifications():
    """Длина выхода меньше длины входа — это нормально."""
    lines = ['{"jsonrpc":"2.0","method":"notifications/initialized"}',
             '{"jsonrpc":"2.0","id":1,"method":"ping"}']
    assert len(serve_lines(make_server(), lines)) == 1


def test_broken_json_becomes_parse_error_with_null_id():
    out = serve_lines(make_server(), ["{not json"])
    message = json.loads(out[0])
    assert message["error"]["code"] == PARSE_ERROR
    assert message["id"] is None


def test_blank_lines_are_skipped():
    assert serve_lines(make_server(), ["", "   ", "\n"]) == []


def test_every_output_line_is_a_single_json_object():
    """В stdout нельзя писать ничего, кроме JSON-RPC-конвертов."""
    lines = ['{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
             '{"jsonrpc":"2.0","method":"notifications/initialized"}',
             '{"jsonrpc":"2.0","id":2,"method":"tools/list"}']
    out = serve_lines(make_server(), lines)
    assert len(out) == 2
    assert all("\n" not in line and isinstance(json.loads(line), dict) for line in out)
