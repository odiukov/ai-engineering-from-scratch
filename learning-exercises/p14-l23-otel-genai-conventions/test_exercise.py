"""Тесты к уроку «Семантические конвенции OpenTelemetry GenAI». Правь exercise.py."""

import pytest

from exercise import (
    capture_content,
    continue_trace,
    describe_span,
    end_span,
    format_traceparent,
    genai_attributes,
    span_tree,
    start_span,
)

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
REMOTE_SPAN = "00f067aa0ba902b7"
HEADER = f"00-{TRACE_ID}-{REMOTE_SPAN}-01"


def sid(n):
    """Валидный 16-символьный span id из номера."""
    return f"{n:016x}"


def names(nodes):
    """Плоский список имён с указанием глубины — удобно сравнивать деревья."""
    out = []

    def walk(node, depth):
        out.append((depth, node["span"]["name"]))
        for child in node["children"]:
            walk(child, depth + 1)

    for node in nodes:
        walk(node, 0)
    return out


# ------------------------------------------------------------ describe_span
def test_named_agent_goes_into_the_span_name():
    """С именем — "invoke_agent planner", без имени — просто "invoke_agent"."""
    assert describe_span("invoke_agent", "planner")["name"] == "invoke_agent planner"
    assert describe_span("invoke_agent")["name"] == "invoke_agent"


def test_remote_agent_service_is_client_and_local_one_is_internal():
    assert describe_span("invoke_agent", "planner", remote=True)["kind"] == "CLIENT"
    assert describe_span("invoke_agent", "planner")["kind"] == "INTERNAL"


def test_chat_span_is_client_even_without_remote_flag():
    """Вызов модели — всегда обращение к удалённому API."""
    assert describe_span("chat", "claude-x")["kind"] == "CLIENT"


def test_tool_span_stays_internal_regardless_of_remote():
    assert describe_span("tool_call", "search", remote=True)["kind"] == "INTERNAL"


def test_unknown_operation_is_rejected():
    with pytest.raises(ValueError):
        describe_span("run_agent", "planner")


# --------------------------------------------------------- genai_attributes
def test_missing_values_produce_no_keys_at_all():
    """None-атрибут бэкенд считает отдельным значением — ключа быть не должно."""
    attrs = genai_attributes("anthropic", "chat", request_model="claude-x")
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert "gen_ai.response.model" not in attrs
    assert None not in attrs.values()


def test_routed_response_model_is_reported_next_to_the_request_model():
    attrs = genai_attributes("openai", "chat", "gpt-x", response_model="gpt-x-0301")
    assert attrs["gen_ai.request.model"] == "gpt-x"
    assert attrs["gen_ai.response.model"] == "gpt-x-0301"


def test_rag_span_records_the_data_source():
    attrs = genai_attributes("anthropic", "tool_call", data_source_id="kb-42")
    assert attrs["gen_ai.data_source.id"] == "kb-42"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        genai_attributes("mistral", "chat", request_model="large")


# ------------------------------------------------------ format_traceparent
def test_traceparent_has_four_dash_separated_fields():
    assert format_traceparent(TRACE_ID, REMOTE_SPAN).split("-") == [
        "00",
        TRACE_ID,
        REMOTE_SPAN,
        "01",
    ]


def test_unsampled_flag_is_zero_zero():
    assert format_traceparent(TRACE_ID, REMOTE_SPAN, sampled=False).endswith("-00")


def test_uppercase_hex_is_rejected():
    """W3C требует нижний регистр; бэкенды не нормализуют за тебя."""
    with pytest.raises(ValueError):
        format_traceparent(TRACE_ID.upper(), REMOTE_SPAN)


def test_all_zero_or_short_ids_are_rejected():
    """Все нули означают «идентификатора нет», а длина фиксированная."""
    with pytest.raises(ValueError):
        format_traceparent("0" * 32, REMOTE_SPAN)
    with pytest.raises(ValueError):
        format_traceparent(TRACE_ID, "abc")


# ---------------------------------------------------------- continue_trace
def test_incoming_header_supplies_the_trace_id_and_the_remote_parent():
    trace = continue_trace(HEADER)
    assert trace["trace_id"] == TRACE_ID
    assert trace["remote_parent"] == REMOTE_SPAN


def test_trace_id_is_not_born_inside_a_span():
    """Спан наследует trace_id входящего заголовка, а не придумывает свой."""
    trace = continue_trace(HEADER)
    span = start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    assert span["trace_id"] == TRACE_ID
    assert span["parent_id"] == REMOTE_SPAN


def test_root_trace_has_no_remote_parent():
    trace = continue_trace(None, TRACE_ID)
    assert trace["remote_parent"] is None
    assert start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)["parent_id"] is None


def test_trace_without_header_and_without_id_is_an_error():
    with pytest.raises(ValueError):
        continue_trace(None)


def test_unparsable_traceparent_is_rejected():
    """Неизвестная версия и обрезанный заголовок — оба отказ, а не догадки."""
    with pytest.raises(ValueError):
        continue_trace(f"99-{TRACE_ID}-{REMOTE_SPAN}-01")
    with pytest.raises(ValueError):
        continue_trace(f"00-{TRACE_ID}-{REMOTE_SPAN}")


# ------------------------------------------------------ start_span/end_span
def test_nested_span_takes_the_open_span_as_parent():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    child = start_span(trace, sid(2), "tool_call search", "INTERNAL", {}, 10)
    assert child["parent_id"] == sid(1)


def test_sibling_span_reattaches_to_the_parent_after_close():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    start_span(trace, sid(2), "tool_call a", "INTERNAL", {}, 10)
    assert end_span(trace, sid(2), 20)["duration_ns"] == 10
    assert start_span(trace, sid(3), "tool_call b", "INTERNAL", {}, 30)["parent_id"] == sid(1)


def test_duplicate_span_id_is_rejected():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "a", "INTERNAL", {}, 0)
    with pytest.raises(ValueError):
        start_span(trace, sid(1), "b", "INTERNAL", {}, 5)


def test_span_attributes_are_not_shared_between_spans():
    """Один и тот же словарь на два спана — и правка одного перепишет второй."""
    trace = continue_trace(None, TRACE_ID)
    shared = {"gen_ai.provider.name": "anthropic"}
    first = start_span(trace, sid(1), "a", "INTERNAL", shared, 0)
    first["attributes"]["gen_ai.request.model"] = "claude-x"
    end_span(trace, sid(1), 1)
    second = start_span(trace, sid(2), "b", "INTERNAL", shared, 2)
    assert "gen_ai.request.model" not in second["attributes"]


def test_child_span_cannot_outlive_its_parent():
    """Закрывать можно только самый внутренний спан — иначе ветка оторвётся."""
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    start_span(trace, sid(2), "tool_call a", "INTERNAL", {}, 10)
    with pytest.raises(ValueError):
        end_span(trace, sid(1), 20)


def test_end_before_start_is_rejected():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "a", "INTERNAL", {}, 100)
    with pytest.raises(ValueError):
        end_span(trace, sid(1), 50)


# --------------------------------------------------------------- span_tree
def test_tree_nests_tool_spans_under_the_agent_span():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    start_span(trace, sid(2), "tool_call search", "INTERNAL", {}, 10)
    end_span(trace, sid(2), 20)
    start_span(trace, sid(3), "chat claude-x", "CLIENT", {}, 30)
    end_span(trace, sid(3), 40)
    end_span(trace, sid(1), 50)
    assert names(span_tree(trace)) == [
        (0, "invoke_agent p"),
        (1, "tool_call search"),
        (1, "chat claude-x"),
    ]


def test_cross_process_trace_renders_as_one_tree():
    """Ветка из другого процесса цепляется к remote_parent, а не рвётся."""
    trace = continue_trace(HEADER)
    start_span(trace, sid(1), "invoke_agent sub", "INTERNAL", {}, 0)
    start_span(trace, sid(2), "tool_call a", "INTERNAL", {}, 5)
    end_span(trace, sid(2), 10)
    end_span(trace, sid(1), 15)
    assert names(span_tree(trace)) == [(0, "invoke_agent sub"), (1, "tool_call a")]


def test_unfinished_span_blocks_the_tree():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    with pytest.raises(ValueError):
        span_tree(trace)


def test_orphaned_span_is_an_error_not_a_second_root():
    """Оторванный tool-спан обязан быть замечен, а не тихо всплыть в корень."""
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    end_span(trace, sid(1), 10)
    trace["spans"][0]["parent_id"] = sid(99)
    with pytest.raises(ValueError):
        span_tree(trace)


def test_tree_keeps_children_in_start_order():
    trace = continue_trace(None, TRACE_ID)
    start_span(trace, sid(1), "invoke_agent p", "INTERNAL", {}, 0)
    for n, at in ((2, 10), (3, 20), (4, 30)):
        start_span(trace, sid(n), f"tool_call t{n}", "INTERNAL", {}, at)
        end_span(trace, sid(n), at + 5)
    end_span(trace, sid(1), 100)
    assert [d for d, _ in names(span_tree(trace))] == [0, 1, 1, 1]
    assert names(span_tree(trace))[1][1] == "tool_call t2"


# ---------------------------------------------------------- capture_content
def make_span():
    trace = continue_trace(None, TRACE_ID)
    span = start_span(trace, sid(1), "chat claude-x", "CLIENT", {}, 0)
    return trace, span


def test_content_is_not_captured_by_default():
    """Конвенция: инструментация НЕ пишет prompt/response, пока не попросили."""
    _, span = make_span()
    assert capture_content({}, span, ["card 4111 1111"]) is None
    assert span["attributes"] == {}


def test_inline_mode_puts_messages_on_the_span():
    _, span = make_span()
    capture_content({}, span, ["hi"], mode="inline")
    assert span["attributes"]["gen_ai.input.messages"] == ["hi"]


def test_reference_mode_keeps_the_text_off_the_span():
    """Продовый режим: в трейсе только идентификатор, текст — во внешнем store."""
    store = {}
    _, span = make_span()
    ref = capture_content(store, span, ["card 4111 1111"], mode="reference")
    assert store[ref] == ["card 4111 1111"]
    assert "4111" not in str(span)


def test_reference_ids_do_not_collide_between_spans():
    store = {}
    _, first = make_span()
    _, second = make_span()
    a = capture_content(store, first, ["one"], mode="reference")
    b = capture_content(store, second, ["two"], mode="reference")
    assert a != b and store[a] == ["one"] and store[b] == ["two"]


def test_captured_messages_are_copied_not_aliased():
    store = {}
    _, span = make_span()
    messages = ["one"]
    ref = capture_content(store, span, messages, mode="reference")
    messages.append("two")
    assert store[ref] == ["one"]


def test_unknown_capture_mode_is_rejected():
    _, span = make_span()
    with pytest.raises(ValueError):
        capture_content({}, span, ["hi"], mode="on")
