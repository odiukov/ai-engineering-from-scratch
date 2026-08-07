"""Тесты к уроку «MCP Apps — интерактивные UI-ресурсы ui://». Правь exercise.py."""

import pytest

from exercise import (
    MCP_APP_MIME,
    accept_message,
    csp_findings,
    csp_header,
    dispatch_host_call,
    is_ui_uri,
    review_permissions,
    tool_result_with_ui,
    ui_resource_contents,
)

HOST = "https://host.example.com"
UI = "https://ui.example.com"
STRICT = {"defaultSrc": "'self'", "scriptSrc": "'self'", "connectSrc": "'self'"}


# ------------------------------------------------------------------ is_ui_uri
def test_ui_scheme_is_accepted():
    assert is_ui_uri("ui://notes/timeline") is True


def test_https_resource_is_not_a_ui_resource():
    assert is_ui_uri("https://notes.example.com/timeline") is False


def test_bare_scheme_without_a_path_is_rejected():
    assert is_ui_uri("ui://") is False


def test_uri_with_whitespace_is_rejected():
    """Пробел внутри адреса — почти всегда склеенная строка, а не URI."""
    assert is_ui_uri("ui://notes/time line") is False


# -------------------------------------------------------- ui_resource_contents
def test_resource_read_carries_the_app_profile_mime():
    """Без профиля хост покажет HTML текстом и iframe не создаст."""
    got = ui_resource_contents("ui://notes/timeline", "<!doctype html>")
    assert got["contents"][0]["mimeType"] == MCP_APP_MIME
    assert MCP_APP_MIME != "text/html"


def test_resource_read_keeps_uri_and_html():
    got = ui_resource_contents("ui://notes/timeline", "<h1>x</h1>")
    assert got["contents"][0]["uri"] == "ui://notes/timeline"
    assert got["contents"][0]["text"] == "<h1>x</h1>"


def test_non_ui_uri_cannot_be_served_as_an_app():
    with pytest.raises(ValueError):
        ui_resource_contents("https://notes.example.com", "<h1>x</h1>")


# ----------------------------------------------------------------- csp_header
def test_camel_case_keys_become_kebab_directives():
    assert csp_header({"defaultSrc": "'self'"}) == "default-src 'self'"


def test_directives_are_joined_with_semicolons():
    got = csp_header({"defaultSrc": "'self'", "scriptSrc": "'self' 'unsafe-inline'"})
    assert got == "default-src 'self'; script-src 'self' 'unsafe-inline'"


def test_header_is_stable_regardless_of_key_order():
    """Заголовок должен быть воспроизводимым — иначе его не захэшировать."""
    a = csp_header({"scriptSrc": "'self'", "defaultSrc": "'self'"})
    b = csp_header({"defaultSrc": "'self'", "scriptSrc": "'self'"})
    assert a == b


# --------------------------------------------------------------- csp_findings
def test_strict_policy_has_no_findings():
    assert csp_findings(STRICT) == []


def test_wildcard_connect_src_is_an_exfiltration_channel():
    assert csp_findings({"defaultSrc": "'self'", "connectSrc": "*"}) == [
        "wildcard_connect_src"
    ]


def test_missing_default_src_is_flagged():
    assert "missing_default_src" in csp_findings({"scriptSrc": "'self'"})


def test_unsafe_inline_is_a_finding_but_not_the_same_as_a_wildcard():
    got = csp_findings({"defaultSrc": "'self'", "scriptSrc": "'self' 'unsafe-inline'"})
    assert got == ["unsafe_inline_script"]


def test_findings_are_sorted_and_can_stack():
    got = csp_findings({"connectSrc": "*", "scriptSrc": "* 'unsafe-inline'"})
    assert got == sorted(got)
    assert len(got) == 4


# --------------------------------------------------------- review_permissions
def test_empty_request_asks_for_nothing():
    assert review_permissions([]) == {"prompt": [], "rejected": []}


def test_known_permission_still_needs_a_user_prompt():
    assert review_permissions(["camera"])["prompt"] == ["camera"]


def test_unknown_permission_is_rejected_not_silently_granted():
    """KNOWN_PERMISSIONS — allowlist: «неизвестно» значит «нет»."""
    got = review_permissions(["camera", "gpu"])
    assert got["rejected"] == ["gpu"] and got["prompt"] == ["camera"]


def test_duplicates_collapse():
    assert review_permissions(["camera", "camera"])["prompt"] == ["camera"]


# ------------------------------------------------------- tool_result_with_ui
def test_tool_result_carries_text_and_ui_resource_blocks():
    got = tool_result_with_ui("Вот таймлайн:", "ui://notes/timeline", STRICT, [])
    assert got["content"][0] == {"type": "text", "text": "Вот таймлайн:"}
    assert got["content"][1] == {"type": "ui_resource", "uri": "ui://notes/timeline"}


def test_meta_ui_binds_the_resource_to_the_tool_result():
    got = tool_result_with_ui("t", "ui://notes/timeline", STRICT, ["camera"])
    assert got["_meta"]["ui"]["resourceUri"] == "ui://notes/timeline"
    assert got["_meta"]["ui"]["permissions"] == ["camera"]


def test_wildcard_connect_src_blocks_the_whole_result():
    with pytest.raises(ValueError):
        tool_result_with_ui("t", "ui://notes/timeline", {"connectSrc": "*"}, [])


def test_unknown_permission_blocks_the_whole_result():
    with pytest.raises(ValueError):
        tool_result_with_ui("t", "ui://notes/timeline", STRICT, ["gpu"])


# --------------------------------------------------------------- accept_message
def test_matching_origin_and_known_method_are_accepted():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.close"}
    assert accept_message(UI, UI, msg) == (True, "ok")


def test_message_from_a_foreign_origin_is_rejected():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.callTool"}
    assert accept_message("https://evil.example.com", UI, msg) == (
        False,
        "origin_mismatch",
    )


def test_wildcard_allowed_origin_is_a_config_error_not_a_permission():
    """По каналу летят вызовы инструментов — принимать их от всех нельзя."""
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.close"}
    assert accept_message(UI, "*", msg) == (False, "wildcard_origin")


def test_method_outside_the_allowlist_is_rejected():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.evalJavaScript"}
    assert accept_message(UI, UI, msg) == (False, "unknown_method")


def test_wrong_jsonrpc_version_is_rejected():
    msg = {"jsonrpc": "1.0", "id": 1, "method": "host.close"}
    assert accept_message(UI, UI, msg) == (False, "bad_jsonrpc")


def test_message_without_id_is_malformed():
    assert accept_message(UI, UI, {"jsonrpc": "2.0", "method": "host.close"}) == (
        False,
        "malformed",
    )


def test_origin_is_checked_before_the_payload_is_even_parsed():
    """Разбирать тело сообщения от чужого origin незачем и опасно."""
    assert accept_message("https://evil.example.com", UI, "не dict вовсе") == (
        False,
        "origin_mismatch",
    )


# ------------------------------------------------------------ dispatch_host_call
def test_accepted_call_reaches_its_handler():
    msg = {"jsonrpc": "2.0", "id": 7, "method": "host.callTool", "params": {"n": 1}}
    got = dispatch_host_call(msg, UI, UI, {"host.callTool": lambda p: p["n"] + 1})
    assert got == {"jsonrpc": "2.0", "id": 7, "result": 2}


def test_rejected_origin_becomes_a_jsonrpc_error_with_the_id_preserved():
    msg = {"jsonrpc": "2.0", "id": 7, "method": "host.close"}
    got = dispatch_host_call(msg, "https://evil.example.com", UI, {})
    assert got["id"] == 7
    assert got["error"]["message"] == "origin_mismatch"
    assert "result" not in got


def test_unknown_method_gets_the_method_not_found_code():
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.evalJavaScript"}
    got = dispatch_host_call(msg, UI, UI, {"host.close": lambda p: None})
    assert got["error"]["code"] == -32601


def test_allowed_method_without_a_handler_is_still_an_error():
    """Allowlist разрешает звать; реализация может быть не подключена."""
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.getPrompt"}
    got = dispatch_host_call(msg, UI, UI, {})
    assert got["error"]["message"] == "no_handler"


def test_handler_is_not_called_when_the_origin_is_wrong():
    calls = []
    msg = {"jsonrpc": "2.0", "id": 1, "method": "host.callTool", "params": {}}
    dispatch_host_call(msg, HOST, UI, {"host.callTool": lambda p: calls.append(p)})
    assert calls == []
