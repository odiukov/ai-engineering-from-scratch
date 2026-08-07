"""Тесты к уроку «Свой MCP-клиент». Правь exercise.py."""

import pytest

from exercise import (
    PROTOCOL_VERSION,
    apply_notification,
    drain,
    handshake_messages,
    merge_tools,
    new_session,
    route_call,
    supports,
)


def init_result(caps=None, protocol=PROTOCOL_VERSION, name="notes"):
    return {
        "protocolVersion": protocol,
        "capabilities": caps if caps is not None else {"tools": {"listChanged": True}},
        "serverInfo": {"name": name, "version": "1.0.0"},
    }


def session(name, tools=(), caps=None, alive=True):
    """Готовая сессия. Собирается ВНУТРИ теста, без вызова exercise."""
    return {
        "name": name,
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": caps if caps is not None else {"tools": {}},
        "serverInfo": {"name": name, "version": "1.0.0"},
        "tools": [{"name": t, "description": t, "inputSchema": {}} for t in tools],
        "pending": {},
        "stale": False,
        "dirty": [],
        "alive": alive,
    }


# ------------------------------------------------------ handshake_messages
def test_handshake_sends_initialize_then_initialized():
    request, note = handshake_messages(1, "host", "0.1")
    assert request["method"] == "initialize"
    assert note["method"] == "notifications/initialized"


def test_initialized_is_a_notification_without_id():
    """Ждать ответа на нотификацию — классический дедлок клиента."""
    _, note = handshake_messages(1, "host", "0.1")
    assert "id" not in note


def test_handshake_declares_protocol_and_client_info():
    request, _ = handshake_messages(42, "host", "0.1")
    assert request["id"] == 42
    assert request["params"]["protocolVersion"] == PROTOCOL_VERSION
    assert request["params"]["clientInfo"] == {"name": "host", "version": "0.1"}


def test_handshake_defaults_to_empty_capabilities():
    request, _ = handshake_messages(1, "host", "0.1")
    assert request["params"]["capabilities"] == {}


def test_handshake_copies_capabilities():
    """Правка словаря вызывающего не должна менять отправленное сообщение."""
    caps = {"roots": {"listChanged": True}}
    request, _ = handshake_messages(1, "host", "0.1", caps)
    caps["sampling"] = {}
    assert "sampling" not in request["params"]["capabilities"]


# ------------------------------------------------------------- new_session
def test_new_session_starts_stale_and_alive():
    s = new_session("notes", init_result())
    assert s["alive"] is True and s["stale"] is True and s["tools"] == []


def test_new_session_keeps_server_capabilities():
    s = new_session("notes", init_result({"resources": {"subscribe": True}}))
    assert s["capabilities"] == {"resources": {"subscribe": True}}


def test_new_session_accepts_an_older_supported_protocol():
    """Версию выбирает сервер — из предложенных клиентом."""
    s = new_session("notes", init_result(protocol="2025-03-26"))
    assert s["protocolVersion"] == "2025-03-26"


def test_unknown_protocol_version_fails_at_handshake():
    with pytest.raises(ValueError):
        new_session("notes", init_result(protocol="1999-01-01"))


def test_new_session_has_no_pending_requests():
    assert new_session("notes", init_result())["pending"] == {}


# ---------------------------------------------------------------- supports
def test_declared_primitive_is_supported():
    assert supports(session("notes", caps={"tools": {}}), "tools") is True


def test_missing_primitive_is_not_supported():
    assert supports(session("notes", caps={"tools": {}}), "prompts") is False


def test_empty_capability_object_means_no_inner_flags():
    """{"tools": {}} — инструменты есть, listChanged не шлю."""
    s = session("notes", caps={"tools": {}})
    assert supports(s, "tools") is True
    assert supports(s, "tools.listChanged") is False


def test_nested_flag_is_read_through_the_dotted_path():
    s = session("notes", caps={"resources": {"subscribe": True, "listChanged": False}})
    assert supports(s, "resources.subscribe") is True
    assert supports(s, "resources.listChanged") is False


# ------------------------------------------------------------- merge_tools
def test_tools_without_collisions_keep_their_names():
    ns = merge_tools([session("notes", ["notes_list"]), session("files", ["read_file"])])
    assert set(ns) == {"notes_list", "read_file"}
    assert ns["read_file"]["server"] == "files"


def test_prefix_policy_renames_only_the_later_tool():
    ns = merge_tools([session("notes", ["search"]), session("files", ["search"])])
    assert set(ns) == {"search", "files/search"}
    assert ns["search"]["server"] == "notes"


def test_first_come_policy_silently_drops_the_duplicate():
    ns = merge_tools([session("notes", ["search"]), session("files", ["search"])], "first")
    assert set(ns) == {"search"}
    assert ns["search"]["server"] == "notes"


def test_reject_policy_refuses_to_load_the_second_server():
    with pytest.raises(ValueError):
        merge_tools([session("notes", ["search"]), session("files", ["search"])], "reject")


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        merge_tools([session("notes", ["search"])], "guess")


def test_dead_session_contributes_no_tools():
    """Модели нельзя показывать инструмент, вызвать который невозможно."""
    ns = merge_tools([session("notes", ["search"], alive=False), session("files", ["search"])])
    assert set(ns) == {"search"}
    assert ns["search"]["server"] == "files"


# -------------------------------------------------------------- route_call
def test_route_call_picks_the_owning_server():
    ns = merge_tools([session("notes", ["notes_list"]), session("files", ["read_file"])])
    server, message = route_call(ns, 5, "read_file", {"path": "a.txt"})
    assert server == "files"
    assert message["method"] == "tools/call"


def test_prefix_is_stripped_before_the_message_leaves():
    """Префикс — выдумка клиента; сервер знает инструмент под своим именем."""
    ns = merge_tools([session("notes", ["search"]), session("files", ["search"])])
    server, message = route_call(ns, 6, "files/search", {"q": "mcp"})
    assert server == "files"
    assert message["params"]["name"] == "search"


def test_route_call_carries_the_request_id_and_arguments():
    ns = merge_tools([session("notes", ["notes_list"])])
    _, message = route_call(ns, 11, "notes_list", {"tag": "mcp"})
    assert message["id"] == 11
    assert message["params"]["arguments"] == {"tag": "mcp"}


def test_unknown_tool_name_raises_key_error():
    ns = merge_tools([session("notes", ["notes_list"])])
    with pytest.raises(KeyError):
        route_call(ns, 12, "notes_teleport", {})


# ------------------------------------------------------------------- drain
def test_response_is_matched_to_a_pending_request():
    s = session("notes")
    s["pending"] = {4: "tools/call"}
    out = drain(s, [{"jsonrpc": "2.0", "id": 4, "result": {"content": []}}])
    assert set(out["responses"]) == {4}
    assert s["pending"] == {}


def test_notification_is_never_treated_as_a_response():
    s = session("notes")
    out = drain(s, [{"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}])
    assert out["notifications"] and out["responses"] == {}


def test_server_request_is_separated_from_a_notification():
    """У обоих есть "method"; отличает наличие id — на запрос надо ответить."""
    s = session("notes")
    out = drain(s, [
        {"jsonrpc": "2.0", "id": 9, "method": "sampling/createMessage", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/resources/updated", "params": {}},
    ])
    assert len(out["server_requests"]) == 1 and len(out["notifications"]) == 1


def test_response_with_an_unknown_id_goes_to_unmatched():
    s = session("notes")
    out = drain(s, [{"jsonrpc": "2.0", "id": 99, "result": {}}])
    assert out["unmatched"] and out["responses"] == {}


def test_eof_marks_the_session_dead():
    s = session("notes")
    drain(s, [None])
    assert s["alive"] is False


def test_nothing_after_eof_is_processed():
    """Байты после EOF — мусор из чужого буфера, а не сообщения."""
    s = session("notes")
    s["pending"] = {1: "ping"}
    out = drain(s, [None, {"jsonrpc": "2.0", "id": 1, "result": {}}])
    assert out["responses"] == {} and s["pending"] == {1: "ping"}


# ------------------------------------------------------- apply_notification
def test_tools_list_changed_marks_the_list_stale():
    s = session("notes")
    apply_notification(s, {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    assert s["stale"] is True


def test_resource_updated_records_the_uri():
    s = session("notes")
    apply_notification(s, {"jsonrpc": "2.0", "method": "notifications/resources/updated",
                           "params": {"uri": "notes://note-1"}})
    assert s["dirty"] == ["notes://note-1"]


def test_repeated_update_of_the_same_uri_does_not_duplicate():
    s = session("notes")
    note = {"jsonrpc": "2.0", "method": "notifications/resources/updated",
            "params": {"uri": "notes://note-1"}}
    apply_notification(s, note)
    apply_notification(s, note)
    assert s["dirty"] == ["notes://note-1"]


def test_unknown_notification_leaves_the_session_untouched():
    """Сервер может знать методы новее нашего клиента."""
    s = session("notes")
    before = (s["stale"], list(s["dirty"]))
    apply_notification(s, {"jsonrpc": "2.0", "method": "notifications/quantum/entangled"})
    assert (s["stale"], s["dirty"]) == before


def test_a_request_is_not_a_notification():
    """Обработать запрос как нотификацию — значит никогда на него не ответить."""
    s = session("notes")
    with pytest.raises(ValueError):
        apply_notification(s, {"jsonrpc": "2.0", "id": 3, "method": "roots/list"})
