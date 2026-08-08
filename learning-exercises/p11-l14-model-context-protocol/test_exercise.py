"""Тесты к уроку «Model Context Protocol». Правь exercise.py."""

import json

from exercise import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    call_tool,
    handle,
    handle_batch,
    make_error,
    make_request,
    make_response,
    tool_schema,
    validate_arguments,
)


# Схемы строятся лениво, внутри функций: если собрать их на уровне модуля,
# незаполненная заготовка уронит СБОР тестов, а не сами тесты, и проверка
# «N failed == N passed» перестанет что-либо значить.
def add_schema():
    return tool_schema(
        "add",
        "Add two integers",
        {"a": {"type": "integer"}, "b": {"type": "integer"}},
        ("a", "b"),
    )


def echo_schema():
    return tool_schema(
        "echo",
        "Echo the text back",
        {"text": {"type": "string"}, "loud": {"type": "boolean"}},
        ("text",),
    )


def boom(**_kwargs):
    raise ZeroDivisionError("division by zero")


def make_server():
    return {
        "name": "demo-server",
        "version": "1.0.0",
        "tools": {
            "add": {"schema": add_schema(), "handler": lambda a, b: a + b},
            "echo": {
                "schema": echo_schema(),
                "handler": lambda text, loud=False: text.upper() if loud else text,
            },
            "boom": {
                "schema": tool_schema("boom", "Always fails", {}, ()),
                "handler": boom,
            },
        },
    }


def initialize_params():
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "exercise-client", "version": "1.0.0"},
    }


# ------------------------------------------------------------ make_request
def test_request_carries_the_protocol_version_and_method():
    assert make_request("tools/list", request_id=1) == {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1,
    }


def test_request_omits_params_when_there_are_none():
    """Спецификация разрешает отсутствие params; слать null — лишний шум."""
    assert "params" not in make_request("tools/list", request_id=1)


def test_request_without_id_is_a_notification():
    assert "id" not in make_request("notifications/initialized")


def test_request_with_id_zero_is_not_a_notification():
    """Ловушка: 0 ложен в булевом смысле, но это законный идентификатор."""
    assert make_request("ping", request_id=0)["id"] == 0


# --------------------------------------------- make_response / make_error
def test_response_echoes_the_request_id():
    assert make_response(42, {"ok": True}) == {
        "jsonrpc": "2.0",
        "id": 42,
        "result": {"ok": True},
    }


def test_error_carries_code_and_message():
    err = make_error(7, METHOD_NOT_FOUND, "Method not found")
    assert err["error"] == {"code": -32601, "message": "Method not found"}


def test_error_omits_data_unless_given():
    assert "data" not in make_error(1, INVALID_PARAMS, "bad")["error"]
    assert make_error(1, INVALID_PARAMS, "bad", {"field": "a"})["error"]["data"] == {"field": "a"}


def test_message_never_has_result_and_error_at_once():
    """Ровно одно из двух — иначе клиент не знает, что делать."""
    ok = make_response(1, "x")
    bad = make_error(1, INTERNAL_ERROR, "boom")
    assert ("result" in ok) and ("error" not in ok)
    assert ("error" in bad) and ("result" not in bad)


# ------------------------------------------------------------- tool_schema
def test_schema_wraps_properties_into_an_object_schema():
    assert add_schema()["inputSchema"]["type"] == "object"
    assert add_schema()["inputSchema"]["required"] == ["a", "b"]


def test_schema_is_json_serialisable():
    """Схема уходит по проводу, значит в ней не должно быть питоновских объектов."""
    schema = add_schema()
    assert json.loads(json.dumps(schema)) == schema


def test_schema_copies_properties_instead_of_aliasing_them():
    props = {"a": {"type": "integer"}}
    schema = tool_schema("t", "d", props, ("a",))
    props["b"] = {"type": "string"}
    assert "b" not in schema["inputSchema"]["properties"]


# ------------------------------------------------------ validate_arguments
def test_valid_arguments_produce_no_complaints():
    assert validate_arguments(add_schema(), {"a": 1, "b": 2}) == []


def test_missing_required_property_is_reported():
    assert validate_arguments(add_schema(), {"a": 1}) == ["missing required property: b"]


def test_wrong_type_is_reported_with_both_types():
    problems = validate_arguments(add_schema(), {"a": 1, "b": "2"})
    assert problems == ["b: expected integer, got str"]


def test_unknown_property_is_reported():
    problems = validate_arguments(add_schema(), {"a": 1, "b": 2, "c": 3})
    assert problems == ["unknown property: c"]


def test_boolean_does_not_sneak_through_as_an_integer():
    """В Python bool — подкласс int, и без отдельной проверки True прошёл бы."""
    assert validate_arguments(add_schema(), {"a": True, "b": 2}) == [
        "a: expected integer, got bool"
    ]


def test_optional_property_may_be_absent():
    assert validate_arguments(echo_schema(), {"text": "hi"}) == []


def test_number_accepts_both_int_and_float():
    schema = tool_schema("f", "d", {"x": {"type": "number"}}, ("x",))
    assert validate_arguments(schema, {"x": 1}) == []
    assert validate_arguments(schema, {"x": 1.5}) == []


# ---------------------------------------------------------------- call_tool
def test_call_returns_the_result_as_text_content():
    result = call_tool(make_server(), 1, {"name": "add", "arguments": {"a": 1, "b": 2}})
    assert result["id"] == 1
    assert result["result"] == {"content": [{"type": "text", "text": "3"}], "isError": False}


def test_call_of_an_unknown_tool_is_a_protocol_error():
    result = call_tool(make_server(), 5, {"name": "nope", "arguments": {}})
    assert result["error"]["code"] == INVALID_PARAMS
    assert result["id"] == 5


def test_call_with_bad_arguments_never_reaches_the_handler():
    calls = []
    server = make_server()
    server["tools"]["add"]["handler"] = lambda a, b: calls.append((a, b))
    result = call_tool(server, 6, {"name": "add", "arguments": {"a": 1}})
    assert result["error"]["code"] == INVALID_PARAMS
    assert calls == []


def test_a_crashing_tool_answers_with_is_error_not_with_a_protocol_error():
    """Модель обязана увидеть, что инструмент упал, — иначе она не исправится."""
    result = call_tool(make_server(), 7, {"name": "boom", "arguments": {}})
    assert "error" not in result
    assert result["result"]["isError"] is True
    assert "ZeroDivisionError" in result["result"]["content"][0]["text"]


def test_a_crashing_tool_does_not_crash_the_server():
    server = make_server()
    call_tool(server, 1, {"name": "boom", "arguments": {}})
    later = call_tool(server, 2, {"name": "add", "arguments": {"a": 2, "b": 3}})
    assert later["result"]["content"][0]["text"] == "5"


# ------------------------------------------------------------------- handle
def test_initialize_reports_the_protocol_version_and_server_name():
    request = make_request("initialize", initialize_params(), request_id=1)
    result = handle(make_server(), request)["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["serverInfo"]["name"] == "demo-server"


def test_initialize_requires_protocol_capabilities_and_client_info():
    for missing in ("protocolVersion", "capabilities", "clientInfo"):
        params = initialize_params()
        del params[missing]
        response = handle(make_server(), make_request("initialize", params, request_id=1))
        assert response["error"]["code"] == INVALID_PARAMS
        assert missing in response["error"]["message"]


def test_initialize_without_params_is_invalid():
    response = handle(make_server(), make_request("initialize", request_id=1))
    assert response["error"]["code"] == INVALID_PARAMS


def test_tools_list_returns_every_schema_and_no_handlers():
    result = handle(make_server(), make_request("tools/list", request_id=2))["result"]
    names = [t["name"] for t in result["tools"]]
    assert names == ["add", "echo", "boom"]
    assert json.loads(json.dumps(result)) == result


def test_unknown_method_is_method_not_found():
    response = handle(make_server(), make_request("tools/delete", request_id=9))
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_error_response_carries_the_same_id_as_the_request():
    """Без совпадения id клиент не поймёт, на какой из запросов пришла ошибка."""
    for request_id in (0, 1, 77, "abc"):
        response = handle(make_server(), make_request("nope", request_id=request_id))
        assert response["id"] == request_id


def test_a_notification_gets_no_answer_at_all():
    assert handle(make_server(), make_request("notifications/initialized")) is None


def test_a_notification_with_an_unknown_method_still_gets_no_answer():
    """Соблазн ответить ошибкой велик, но спецификация это запрещает."""
    assert handle(make_server(), make_request("tools/delete")) is None


def test_a_notification_still_runs_the_tool():
    seen = []
    server = make_server()
    server["tools"]["add"]["handler"] = lambda a, b: seen.append(a + b)
    response = handle(
        server,
        make_request("tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}}),
    )
    assert response is None
    assert seen == [3]


def test_wrong_protocol_version_is_an_invalid_request():
    bad = {"jsonrpc": "1.0", "method": "tools/list", "id": 3}
    assert handle(make_server(), bad)["error"]["code"] == INVALID_REQUEST


def test_handle_routes_tools_call_through_to_the_handler():
    request = make_request("tools/call", {"name": "echo", "arguments": {"text": "hi", "loud": True}}, 4)
    assert handle(make_server(), request)["result"]["content"][0]["text"] == "HI"


# -------------------------------------------------------------- handle_batch
def test_batch_skips_notifications_but_keeps_the_rest():
    server = make_server()
    batch = [
        make_request("initialize", initialize_params(), request_id=1),
        make_request("notifications/initialized"),
        make_request("tools/list", request_id=2),
    ]
    responses = handle_batch(server, batch)
    assert [r["id"] for r in responses] == [1, 2]


def test_batch_of_only_notifications_returns_nothing():
    """Не список из None, а пустой список: транспорт не шлёт ничего."""
    server = make_server()
    batch = [make_request("notifications/initialized"), make_request("notifications/cancelled")]
    assert handle_batch(server, batch) == []


def test_batch_preserves_ids_so_answers_can_be_matched_out_of_order():
    server = make_server()
    batch = [
        make_request("tools/call", {"name": "add", "arguments": {"a": 1, "b": 1}}, "x"),
        make_request("tools/call", {"name": "add", "arguments": {"a": 2, "b": 2}}, "y"),
    ]
    by_id = {r["id"]: r["result"]["content"][0]["text"] for r in handle_batch(server, batch)}
    assert by_id == {"x": "2", "y": "4"}
