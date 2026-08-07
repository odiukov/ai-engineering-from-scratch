"""Тесты к уроку «OpenTelemetry GenAI — трассировка вызовов инструментов». Правь exercise.py."""

import random

import pytest

from exercise import (
    REQUIRED_ATTRS,
    capture_content_event,
    finish_span,
    missing_gen_ai_attributes,
    new_span,
    parse_traceparent,
    span_tree,
    trace_problems,
    traceparent,
)

AGENT_ATTRS = {
    "gen_ai.operation.name": "invoke_agent",
    "gen_ai.agent.name": "research-agent",
    "gen_ai.agent.id": "agent_42",
}

TOOL_ATTRS = {
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.name": "get_weather",
    "gen_ai.tool.call.id": "call_01",
}

CHAT_ATTRS = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.model": "claude-sonnet",
    "gen_ai.response.model": "claude-sonnet-2026-02-01",
    "gen_ai.response.id": "resp_1",
    "gen_ai.usage.input_tokens": 120,
    "gen_ai.usage.output_tokens": 45,
}


def rng():
    return random.Random(20260807)


def sample_trace():
    """Корень с двумя детьми; ребёнок закрывается раньше родителя."""
    r = rng()
    root = new_span("agent.invoke_agent", "INTERNAL", 1000, r, attrs=dict(AGENT_ATTRS))
    llm = new_span("llm.chat", "CLIENT", 1100, r, parent=root, attrs=dict(CHAT_ATTRS))
    tool = new_span("tool.execute", "INTERNAL", 1300, r, parent=root, attrs=dict(TOOL_ATTRS))
    finish_span(llm, 1200)
    finish_span(tool, 1400)
    finish_span(root, 1500)
    return root, llm, tool


# ------------------------------------------------------------------ new_span
def test_child_span_inherits_the_trace_id():
    root, llm, _ = sample_trace()
    assert llm["traceId"] == root["traceId"]


def test_child_span_gets_its_own_span_id_and_points_at_the_parent():
    root, llm, _ = sample_trace()
    assert root["parentSpanId"] is None
    assert llm["spanId"] != root["spanId"]
    assert llm["parentSpanId"] == root["spanId"]


def test_ids_are_reproducible_for_the_same_seed():
    """Идентификаторы берутся из rng, а не из глобального random."""
    a = new_span("x", "INTERNAL", 0, random.Random(7))
    b = new_span("x", "INTERNAL", 0, random.Random(7))
    assert (a["traceId"], a["spanId"]) == (b["traceId"], b["spanId"])


def test_unknown_span_kind_is_rejected():
    with pytest.raises(ValueError):
        new_span("x", "OUTGOING", 0, rng())


# --------------------------------------------------------------- finish_span
def test_finishing_records_the_end_time():
    span = new_span("x", "INTERNAL", 1000, rng())
    assert finish_span(span, 2000)["endTimeUnixNano"] == 2000


def test_unfinished_span_has_none_not_zero():
    """Ноль — законное время; «не закрыт» обязан отличаться от «закрыт в нуле»."""
    assert new_span("x", "INTERNAL", 0, rng())["endTimeUnixNano"] is None


def test_zero_duration_span_is_allowed():
    span = new_span("x", "INTERNAL", 1000, rng())
    assert finish_span(span, 1000)["endTimeUnixNano"] == 1000


def test_finishing_twice_is_rejected():
    span = finish_span(new_span("x", "INTERNAL", 1000, rng()), 2000)
    with pytest.raises(ValueError):
        finish_span(span, 3000)


def test_end_before_start_is_rejected():
    with pytest.raises(ValueError):
        finish_span(new_span("x", "INTERNAL", 2000, rng()), 1000)


# --------------------------------------------------------------- traceparent
def test_traceparent_carries_this_span_id_not_the_parents():
    """В заголовок идёт текущий спан: для сервера он станет родителем."""
    root, llm, _ = sample_trace()
    carried = parse_traceparent(traceparent(llm))
    assert carried["spanId"] == llm["spanId"]
    assert carried["traceId"] == root["traceId"]


def test_traceparent_encodes_the_sampled_flag():
    root, _, _ = sample_trace()
    assert traceparent(root, sampled=False).endswith("-00")
    assert parse_traceparent(traceparent(root, sampled=False))["sampled"] is False


# --------------------------------------------------------- parse_traceparent
def test_parse_rejects_a_header_with_wrong_field_count():
    with pytest.raises(ValueError):
        parse_traceparent("00-" + "a" * 32 + "-" + "b" * 16)


def test_parse_rejects_an_unsupported_version():
    with pytest.raises(ValueError):
        parse_traceparent("01-" + "a" * 32 + "-" + "b" * 16 + "-01")


def test_parse_rejects_uppercase_hex():
    """Бэкенды сравнивают id как строки: "AB..." и "ab..." — разные трассы."""
    with pytest.raises(ValueError):
        parse_traceparent("00-" + "A" * 32 + "-" + "b" * 16 + "-01")


def test_parse_rejects_the_all_zero_trace_id():
    with pytest.raises(ValueError):
        parse_traceparent("00-" + "0" * 32 + "-" + "b" * 16 + "-01")


def test_parse_rejects_a_short_span_id():
    with pytest.raises(ValueError):
        parse_traceparent("00-" + "a" * 32 + "-" + "b" * 8 + "-01")


# ------------------------------------------------- missing_gen_ai_attributes
def test_complete_chat_span_is_missing_nothing():
    _, llm, _ = sample_trace()
    assert missing_gen_ai_attributes(llm) == []


def test_missing_attributes_come_back_in_semconv_order():
    span = new_span("tool.execute", "INTERNAL", 0, rng(),
                    attrs={"gen_ai.operation.name": "execute_tool"})
    assert missing_gen_ai_attributes(span) == [
        "gen_ai.tool.name",
        "gen_ai.tool.call.id",
    ]


def test_operations_require_different_attribute_sets():
    """У tool-спана нет request.model, у chat-спана нет tool.name."""
    assert "gen_ai.request.model" not in REQUIRED_ATTRS["execute_tool"]
    assert "gen_ai.tool.name" not in REQUIRED_ATTRS["chat"]
    # без этой половины тест ничего не проверяет: он прошёл бы и на заготовке
    _, llm, tool = sample_trace()
    assert missing_gen_ai_attributes(llm) == missing_gen_ai_attributes(tool) == []


def test_span_without_operation_name_cannot_be_checked():
    span = new_span("mystery", "INTERNAL", 0, rng(), attrs={"gen_ai.tool.name": "x"})
    with pytest.raises(ValueError):
        missing_gen_ai_attributes(span)


def test_misspelled_operation_is_rejected():
    """Опечатка tool_execute вместо execute_tool стоит целого дашборда."""
    span = new_span("t", "INTERNAL", 0, rng(),
                    attrs={"gen_ai.operation.name": "tool_execute"})
    with pytest.raises(ValueError):
        missing_gen_ai_attributes(span)


# ------------------------------------------------------ capture_content_event
def test_content_is_not_captured_by_default():
    """PII в prompt не должны уезжать в чужой бэкенд без явного согласия."""
    span = new_span("llm.chat", "CLIENT", 0, rng(), attrs=dict(CHAT_ATTRS))
    added = capture_content_event(span, "gen_ai.content.prompt", "SSN 123-45-6789", 5)
    assert added is False
    assert span["events"] == []


def test_content_is_captured_when_opted_in():
    span = new_span("llm.chat", "CLIENT", 0, rng(), attrs=dict(CHAT_ATTRS))
    added = capture_content_event(
        span, "gen_ai.content.completion", "hello", 5, capture_content=True
    )
    assert added is True
    assert span["events"][-1]["attributes"]["content"] == "hello"
    assert span["events"][-1]["timeUnixNano"] == 5


def test_a_misspelled_event_name_is_rejected_even_with_capture_off():
    """Иначе опечатка проснётся в тот день, когда сбор включат в проде."""
    span = new_span("llm.chat", "CLIENT", 0, rng(), attrs=dict(CHAT_ATTRS))
    with pytest.raises(ValueError):
        capture_content_event(span, "gen_ai.prompt.content", "hi", 5)


# ---------------------------------------------------------------- span_tree
def test_tree_nests_children_under_their_parent():
    root, llm, tool = sample_trace()
    tree = span_tree([root, llm, tool])
    assert tree["span"] is root
    assert [c["span"]["name"] for c in tree["children"]] == ["llm.chat", "tool.execute"]


def test_tree_does_not_depend_on_the_order_spans_were_exported():
    root, llm, tool = sample_trace()
    assert span_tree([tool, root, llm]) == span_tree([root, llm, tool])


def test_tree_rejects_spans_from_two_traces():
    root, llm, _ = sample_trace()
    other = new_span("other.root", "INTERNAL", 0, random.Random(1))
    with pytest.raises(ValueError):
        span_tree([root, llm, other])


def test_tree_rejects_a_forest():
    """Две вершины без родителя — это не одна трасса."""
    root, llm, _ = sample_trace()
    orphan = dict(llm, parentSpanId=None)
    with pytest.raises(ValueError):
        span_tree([root, orphan])


def test_tree_rejects_a_dangling_parent_reference():
    root, llm, _ = sample_trace()
    with pytest.raises(ValueError):
        span_tree([root, dict(llm, parentSpanId="dead" * 4)])


# ------------------------------------------------------------ trace_problems
def test_a_well_formed_trace_has_no_problems():
    root, llm, tool = sample_trace()
    assert trace_problems([root, llm, tool]) == []


def test_child_outliving_its_parent_is_reported():
    """Родитель по определению охватывает ребёнка; иначе спан прицеплен не туда."""
    root, llm, tool = sample_trace()
    llm["endTimeUnixNano"] = root["endTimeUnixNano"] + 1
    assert any("ends after parent" in p for p in trace_problems([root, llm, tool]))


def test_child_starting_before_its_parent_is_reported():
    root, llm, tool = sample_trace()
    llm["startTimeUnixNano"] = root["startTimeUnixNano"] - 1
    assert any("starts before parent" in p for p in trace_problems([root, llm, tool]))


def test_unfinished_span_is_reported():
    root, llm, tool = sample_trace()
    llm["endTimeUnixNano"] = None
    assert any("not finished" in p for p in trace_problems([root, llm, tool]))


def test_missing_required_attribute_is_reported_per_span():
    root, llm, tool = sample_trace()
    del tool["attributes"]["gen_ai.tool.call.id"]
    assert "tool.execute: missing gen_ai.tool.call.id" in trace_problems([root, llm, tool])


def test_linter_collects_every_problem_instead_of_stopping_at_the_first():
    root, llm, tool = sample_trace()
    llm["endTimeUnixNano"] = None
    del tool["attributes"]["gen_ai.tool.call.id"]
    assert len(trace_problems([root, llm, tool])) >= 2
