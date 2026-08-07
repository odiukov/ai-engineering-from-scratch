"""Тесты к уроку «Основы MCP: примитивы, жизненный цикл, JSON-RPC». Правь exercise.py."""

import pytest

from exercise import (
    CLIENT_PRIMITIVES,
    PROTOCOL_VERSION,
    SERVER_PRIMITIVES,
    classify_message,
    is_permitted,
    negotiated_features,
    owner_of,
    pair_messages,
    primitive_of,
    trace,
    transcript_stats,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CLIENT_CAPS = {"roots": {"listChanged": True}, "sampling": {}, "elicitation": {}}
SERVER_CAPS = {
    "tools": {"listChanged": True},
    "resources": {"subscribe": True, "listChanged": True},
    "prompts": {"listChanged": False},
}


def request(message_id, method, params=None):
    body = {"jsonrpc": "2.0", "id": message_id, "method": method}
    if params is not None:
        body["params"] = params
    return body


def response(message_id, result):
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def error(message_id, code, message):
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def notification(method, params=None):
    body = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    return body


# Полная сессия: рукопожатие, работа, ошибка на неизвестном методе.
# id начинается с нуля намеренно — на нём ломаются наивные проверки.
SESSION = [
    request(0, "initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": CLIENT_CAPS}),
    response(0, {"protocolVersion": PROTOCOL_VERSION, "capabilities": SERVER_CAPS}),
    notification("notifications/initialized"),
    request(1, "tools/list"),
    response(1, {"tools": []}),
    request(2, "tools/call", {"name": "notes_search", "arguments": {"query": "mcp"}}),
    response(2, {"content": [{"type": "text", "text": "2 notes"}], "isError": False}),
    notification("notifications/tools/list_changed"),
    request(3, "tools/delete", {"name": "notes_search"}),
    error(3, -32601, "Method not found"),
]


# ------------------------------------------------------- classify_message
def test_request_and_notification_differ_only_by_the_id_key():
    assert classify_message(request(1, "tools/list")) == "request"
    assert classify_message(notification("notifications/initialized")) == "notification"


def test_id_zero_is_a_request_not_a_notification():
    """`if not message.get("id")` объявил бы нотификацией первый запрос сессии."""
    assert classify_message(request(0, "initialize")) == "request"


def test_result_and_error_are_told_apart():
    assert classify_message(response(1, {})) == "response"
    assert classify_message(error(1, -32601, "Method not found")) == "error"


def test_result_and_error_together_are_invalid():
    """В ответе ровно один из двух ключей — так сказано в спецификации."""
    both = {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {"code": -1, "message": "x"}}
    assert classify_message(both) == "invalid"


def test_malformed_envelopes_are_invalid():
    assert classify_message({"jsonrpc": "1.0", "id": 1, "method": "tools/list"}) == "invalid"
    assert classify_message({"jsonrpc": "2.0"}) == "invalid"


# ------------------------------------------------------------ primitive_of
def test_method_prefix_names_the_primitive():
    assert primitive_of("tools/call") == "tools"
    assert primitive_of("resources/subscribe") == "resources"
    assert primitive_of("sampling/createMessage") == "sampling"


def test_notification_prefix_is_skipped_when_finding_the_primitive():
    assert primitive_of("notifications/tools/list_changed") == "tools"
    assert primitive_of("notifications/resources/updated") == "resources"


def test_handshake_methods_belong_to_the_lifecycle_not_a_primitive():
    """notifications/initialized тоже начинается с notifications/, но примитива нет."""
    assert primitive_of("initialize") == "lifecycle"
    assert primitive_of("notifications/initialized") == "lifecycle"


def test_unknown_action_keeps_its_primitive_unknown_primitive_does_not():
    """tools/delete адресован роутеру tools — он и отвечает на него -32601."""
    assert primitive_of("tools/delete") == "tools"
    assert primitive_of("cron/schedule") is None


# ---------------------------------------------------------------- owner_of
def test_each_side_owns_its_own_three_primitives():
    """sampling объявляет клиент, а вызывает сервер — не наоборот."""
    assert all(owner_of(p) == "server" for p in SERVER_PRIMITIVES)
    assert all(owner_of(p) == "client" for p in CLIENT_PRIMITIVES)


def test_the_six_primitives_do_not_overlap():
    assert set(SERVER_PRIMITIVES) & set(CLIENT_PRIMITIVES) == set()
    assert len(SERVER_PRIMITIVES) + len(CLIENT_PRIMITIVES) == 6
    # у каждого из шести ровно один владелец, и он не None
    assert {owner_of(p) for p in SERVER_PRIMITIVES + CLIENT_PRIMITIVES} == {
        "server",
        "client",
    }


def test_lifecycle_belongs_to_neither_side():
    assert owner_of("lifecycle") is None


# ------------------------------------------------------ negotiated_features
def test_both_sides_contribute_their_own_primitives():
    features = negotiated_features(CLIENT_CAPS, SERVER_CAPS)
    assert "sampling" in features
    assert "tools" in features


def test_sub_flags_are_listed_with_a_dot():
    features = negotiated_features(CLIENT_CAPS, SERVER_CAPS)
    assert "resources.subscribe" in features
    assert "roots.listChanged" in features


def test_a_disabled_sub_flag_is_not_a_feature():
    """prompts.listChanged объявлен со значением False — значит выключен."""
    features = negotiated_features(CLIENT_CAPS, SERVER_CAPS)
    assert "prompts" in features
    assert "prompts.listChanged" not in features


def test_a_capability_declared_by_the_wrong_side_does_not_count():
    """Сервер, написавший себе "sampling", ничего этим не включает."""
    assert negotiated_features({}, {"sampling": {}, "tools": {}}) == ["tools"]
    assert negotiated_features({}, {}) == []


def test_result_is_sorted_and_therefore_stable():
    assert negotiated_features(CLIENT_CAPS, SERVER_CAPS) == sorted(
        negotiated_features(CLIENT_CAPS, SERVER_CAPS)
    )


# ------------------------------------------------------------ is_permitted
def test_lifecycle_is_always_permitted():
    assert is_permitted("initialize", {}, {}) is True
    assert is_permitted("notifications/initialized", {}, {}) is True


def test_an_undeclared_or_unknown_primitive_is_forbidden():
    assert is_permitted("tools/call", CLIENT_CAPS, {}) is False
    assert is_permitted("tools/call", {}, {"tools": {}}) is True
    assert is_permitted("cron/schedule", CLIENT_CAPS, SERVER_CAPS) is False


def test_server_may_not_sample_without_the_client_declaring_it():
    """Клиент без модели остаётся валидным клиентом MCP именно из-за этого."""
    assert is_permitted("sampling/createMessage", {}, SERVER_CAPS) is False
    assert is_permitted("sampling/createMessage", {"sampling": {}}, {}) is True


def test_subscribe_needs_the_sub_flag_not_just_the_primitive():
    assert is_permitted("resources/subscribe", {}, {"resources": {}}) is False
    assert is_permitted("resources/subscribe", {}, {"resources": {"subscribe": True}}) is True


def test_list_changed_notification_needs_its_own_flag():
    assert is_permitted("notifications/prompts/list_changed", {}, SERVER_CAPS) is False
    assert is_permitted("notifications/tools/list_changed", {}, SERVER_CAPS) is True



# ------------------------------------------------------------ pair_messages
def test_every_request_finds_its_response():
    paired = pair_messages(SESSION)
    assert [req["id"] for req, _ in paired["pairs"]] == [0, 1, 2, 3]
    assert paired["pending"] == []
    assert paired["orphans"] == []


def test_an_error_counts_as_the_answer_to_its_request():
    paired = pair_messages(SESSION)
    last_request, last_response = paired["pairs"][-1]
    assert last_request["method"] == "tools/delete"
    assert last_response["error"]["code"] == -32601


def test_responses_arriving_out_of_order_still_pair_up():
    """В одном соединении порядок ответов ничем не гарантирован."""
    shuffled = [request(0, "tools/list"), request(1, "prompts/list"),
                response(1, {}), response(0, {})]
    paired = pair_messages(shuffled)
    assert [req["id"] for req, _ in paired["pairs"]] == [0, 1]


def test_unmatched_messages_end_up_in_pending_and_orphans():
    pending = pair_messages(SESSION[:-1])
    assert [req["id"] for req in pending["pending"]] == [3]

    orphaned = pair_messages([response(99, {})])
    assert len(orphaned["orphans"]) == 1
    assert orphaned["pairs"] == []


def test_notifications_are_collected_separately():
    paired = pair_messages(SESSION)
    methods = [n["method"] for n in paired["notifications"]]
    assert methods == ["notifications/initialized", "notifications/tools/list_changed"]


# ------------------------------------------------------------------- trace
def test_phase_switches_after_the_initialized_notification():
    phases = [entry["phase"] for entry in trace(SESSION)]
    assert phases[:3] == ["initialize"] * 3
    assert set(phases[3:]) == {"operation"}


def test_a_response_borrows_the_method_of_its_request():
    """У ответа нет поля method — оно берётся у запроса с тем же id."""
    marked = trace(SESSION)
    assert marked[4]["kind"] == "response"
    assert marked[4]["method"] == "tools/list"
    assert marked[4]["primitive"] == "tools"


def test_trace_keeps_the_transcript_order_and_length():
    assert len(trace(SESSION)) == len(SESSION)


def test_an_error_is_traced_with_the_method_of_its_request():
    marked = trace(SESSION)
    assert marked[-1]["kind"] == "error"
    assert marked[-1]["method"] == "tools/delete"
    assert marked[-1]["primitive"] == "tools"


def test_an_empty_transcript_traces_to_nothing():
    assert trace([]) == []


# -------------------------------------------------------- transcript_stats
def test_counters_are_present_even_when_zero():
    stats = transcript_stats([])
    assert stats["request"] == 0 and stats["invalid"] == 0
    assert stats["lifecycle_share"] == APPROX(0.0)


def test_counts_match_the_session():
    stats = transcript_stats(SESSION)
    assert (stats["request"], stats["response"], stats["error"], stats["notification"]) == (
        4,
        3,
        1,
        2,
    )


def test_lifecycle_share_is_the_handshake_fraction_and_dilutes_over_time():
    """Три сообщения рукопожатия из десяти; на длинной сессии — доли процента."""
    assert transcript_stats(SESSION)["lifecycle_share"] == APPROX(0.3)
    long_session = SESSION + [request(9, "tools/call"), response(9, {})] * 20
    assert transcript_stats(long_session)["lifecycle_share"] < 0.1
