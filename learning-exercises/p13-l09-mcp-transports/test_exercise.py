"""Тесты к уроку «Транспорты MCP». Правь exercise.py."""

import json
import random

import pytest

from exercise import (
    detect_transport,
    handle_http,
    new_session_id,
    origin_allowed,
    parse_sse,
    replay_after,
    split_stdio,
    sse_event,
)

MESSAGES = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
]

STREAM = "".join(json.dumps(m) + "\n" for m in MESSAGES)


def echo_handler(message):
    """Заглушка сервера: нотификации без ответа, запросы — эхом."""
    if message is None or "id" not in message:
        return None
    return {"jsonrpc": "2.0", "id": message["id"], "result": {"echo": message["method"]}}


def make_state():
    return {
        "endpoint": "/mcp",
        "allowlist": ["http://localhost", "https://claude.ai"],
        "sessions": {},
        "handler": echo_handler,
    }


# -------------------------------------------------------------- split_stdio
def test_complete_lines_become_messages():
    messages, rest = split_stdio("", STREAM)
    assert messages == MESSAGES and rest == ""


def test_partial_tail_is_kept_in_the_buffer():
    messages, rest = split_stdio("", '\n{"id": 1}\n{"id": ')
    assert messages == [{"id": 1}] and rest == '{"id": '


def test_message_split_across_chunks_is_reassembled():
    messages_a, rest = split_stdio("", '{"id":')
    messages_b, rest = split_stdio(rest, " 1}")
    messages_c, rest = split_stdio(rest, "\n")
    assert messages_a == [] and messages_b == []
    assert messages_c == [{"id": 1}] and rest == ""


def test_any_random_chunking_yields_the_same_messages():
    """Границы read() случайны; набор сообщений от них зависеть не должен."""
    rng = random.Random(0)
    for _ in range(20):
        buffer, seen, pos = "", [], 0
        while pos < len(STREAM):
            size = rng.randint(1, 9)
            got, buffer = split_stdio(buffer, STREAM[pos:pos + size])
            seen.extend(got)
            pos += size
        assert seen == MESSAGES and buffer == ""


# ----------------------------------------------------------- new_session_id
def test_session_id_is_thirty_two_hex_characters():
    session_id = new_session_id(random.Random(0))
    assert len(session_id) == 32
    int(session_id, 16)


def test_same_seed_gives_the_same_session_id():
    assert new_session_id(random.Random(7)) == new_session_id(random.Random(7))


def test_consecutive_ids_from_one_rng_differ():
    rng = random.Random(0)
    assert new_session_id(rng) != new_session_id(rng)


def test_short_session_id_is_refused():
    """Идентификатор сессии — это bearer-токен; 64 бита угадываются."""
    with pytest.raises(ValueError):
        new_session_id(random.Random(0), bits=64)


# ----------------------------------------------------------- origin_allowed
def test_exact_origin_from_the_allowlist_passes():
    assert origin_allowed("http://localhost", ["http://localhost"]) is True


def test_unknown_origin_is_rejected():
    assert origin_allowed("http://evil.example", ["http://localhost"]) is False


def test_wildcard_matches_a_subdomain():
    assert origin_allowed("https://app.example.com", ["https://*.example.com"]) is True


def test_wildcard_matches_only_real_subdomains():
    """Наивная проверка через `in` пропустила бы суффиксную атаку."""
    assert origin_allowed("https://evil.example.com.attacker.net",
                          ["https://*.example.com"]) is False
    assert origin_allowed("https://example.com", ["https://*.example.com"]) is False


def test_missing_origin_means_the_caller_is_not_a_browser():
    assert origin_allowed(None, ["http://localhost"]) is True


# --------------------------------------------------------- sse_event/parse
def test_event_frame_ends_with_a_blank_line():
    """Пустая строка — единственный признак «кадр целиком»."""
    assert sse_event("hi") == "data: hi\n\n"


def test_event_id_and_name_precede_the_data():
    frame = sse_event("hi", event_id=7, event="message")
    assert frame == "id: 7\nevent: message\ndata: hi\n\n"


def test_newline_inside_data_gets_its_own_prefix():
    """Иначе перевод строки внутри данных оборвал бы кадр."""
    assert sse_event("first\nsecond") == "data: first\ndata: second\n\n"


def test_parse_reads_id_event_and_data():
    events = parse_sse('id: 7\nevent: message\ndata: {"a":1}\n\n')
    assert events == [{"id": "7", "event": "message", "data": '{"a":1}'}]


def test_keepalive_comment_is_not_an_event():
    events = parse_sse(": keepalive\n\ndata: hi\n\n")
    assert len(events) == 1 and events[0]["data"] == "hi"


def test_multiline_data_round_trips():
    frame = sse_event("first\nsecond", event_id=3)
    assert parse_sse(frame)[0]["data"] == "first\nsecond"


def test_parsed_id_is_a_string_not_a_number():
    """Заголовок last-event-id тоже строка; сравнивать с int бесполезно."""
    assert parse_sse(sse_event("hi", event_id=12))[0]["id"] == "12"


# ------------------------------------------------------------ replay_after
def test_replay_without_last_id_returns_everything():
    events = [{"id": "1", "data": "a"}, {"id": "2", "data": "b"}]
    assert replay_after(events, None) == events


def test_replay_skips_the_acknowledged_event():
    """Отдать событие с этим id ещё раз — дубль в контексте модели."""
    events = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    assert replay_after(events, "2") == [{"id": "3"}]


def test_replay_after_the_newest_event_is_empty():
    events = [{"id": "1"}, {"id": "2"}]
    assert replay_after(events, "2") == []


def test_unknown_last_id_replays_everything():
    events = [{"id": "1"}, {"id": "2"}]
    assert replay_after(events, "999") == events


# -------------------------------------------------------- detect_transport
def test_single_endpoint_response_means_streamable_http():
    assert detect_transport({"status": 200,
                             "headers": {"Content-Type": "application/json"}}) == "streamable-http"
    assert detect_transport({"status": 200,
                             "headers": {"Content-Type": "text/event-stream"}}) == "streamable-http"


def test_sse_with_location_means_the_legacy_two_endpoint_mode():
    response = {"status": 200, "headers": {"content-type": "text/event-stream",
                                           "location": "/messages"}}
    assert detect_transport(response) == "http-sse-legacy"


def test_non_200_probe_is_unsupported():
    assert detect_transport({"status": 404, "headers": {}}) == "unsupported"


# --------------------------------------------------------------- handle_http
def test_first_post_receives_a_fresh_session_id():
    state = make_state()
    status, headers, body = handle_http(
        state, "POST", "/mcp", {"Origin": "http://localhost"},
        {"jsonrpc": "2.0", "id": 1, "method": "ping"}, random.Random(0))
    assert status == 200 and body["result"] == {"echo": "ping"}
    assert len(headers["Mcp-Session-Id"]) == 32 and len(state["sessions"]) == 1


def test_second_post_reuses_the_echoed_session_id():
    state = make_state()
    _, headers, _ = handle_http(state, "POST", "/mcp", {"Origin": "http://localhost"},
                                {"jsonrpc": "2.0", "id": 1, "method": "ping"}, random.Random(0))
    sid = headers["Mcp-Session-Id"]
    _, headers2, _ = handle_http(state, "POST", "/mcp", {"mcp-session-id": sid},
                                 {"jsonrpc": "2.0", "id": 2, "method": "ping"}, random.Random(1))
    assert headers2["Mcp-Session-Id"] == sid and len(state["sessions"]) == 1


def test_revoked_session_forces_a_new_handshake():
    """Клиент видит 404 и обязан заново пройти initialize."""
    state = make_state()
    status, _, _ = handle_http(state, "POST", "/mcp", {"Mcp-Session-Id": "de" * 16},
                               {"jsonrpc": "2.0", "id": 1, "method": "ping"}, random.Random(0))
    assert status == 404


def test_rejected_requests_never_create_a_session():
    """Сначала путь и Origin, только потом всё остальное."""
    state = make_state()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    bad_origin, _, _ = handle_http(state, "POST", "/mcp", {"Origin": "http://evil.example"},
                                   msg, random.Random(0))
    bad_path, _, _ = handle_http(state, "POST", "/rpc", {"Origin": "http://localhost"},
                                 msg, random.Random(0))
    assert (bad_origin, bad_path) == (403, 404) and state["sessions"] == {}


def test_delete_terminates_the_session():
    state = make_state()
    _, headers, _ = handle_http(state, "POST", "/mcp", {"Origin": "http://localhost"},
                                {"jsonrpc": "2.0", "id": 1, "method": "ping"}, random.Random(0))
    sid = headers["Mcp-Session-Id"]
    status, _, _ = handle_http(state, "DELETE", "/mcp", {"Mcp-Session-Id": sid}, None,
                               random.Random(0))
    assert status == 204 and state["sessions"] == {}


def test_get_opens_the_event_stream_for_a_known_session():
    state = make_state()
    _, headers, _ = handle_http(state, "POST", "/mcp", {"Origin": "http://localhost"},
                                {"jsonrpc": "2.0", "id": 1, "method": "ping"}, random.Random(0))
    status, out, _ = handle_http(state, "GET", "/mcp",
                                 {"Mcp-Session-Id": headers["Mcp-Session-Id"]}, None,
                                 random.Random(0))
    assert status == 200 and out["Content-Type"] == "text/event-stream"


def test_notification_is_accepted_with_no_body():
    """202 Accepted, а не 200 с "result": null."""
    state = make_state()
    status, _, body = handle_http(state, "POST", "/mcp", {"Origin": "http://localhost"},
                                  {"jsonrpc": "2.0", "method": "notifications/initialized"},
                                  random.Random(0))
    assert status == 202 and body is None


def test_unsupported_http_method_reports_what_is_allowed():
    state = make_state()
    status, headers, _ = handle_http(state, "PUT", "/mcp", {"Origin": "http://localhost"},
                                     None, random.Random(0))
    assert status == 405 and "POST" in headers["Allow"]
