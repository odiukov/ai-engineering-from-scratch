"""Тесты к уроку «OpenAI Agents SDK: хендоффы, guardrails, трейсинг». Правь exercise.py."""

import pytest

from exercise import (
    SENSITIVE_ATTRIBUTES,
    handoff_tool_name,
    make_agent,
    redact_spans,
    run_guarded,
    run_guardrails,
    run_turn,
    session_prompt,
    visible_tools,
)


# ------------------------------------------------- детерминированные policy
def billing_policy(user_input):
    return {"kind": "final", "text": f"billing handled: {user_input}"}


def support_policy(user_input):
    return {"kind": "final", "text": f"support handled: {user_input}"}


def triage_policy(user_input):
    """Роутер. Хендофф без ключа "input": контекст должен уехать как есть."""
    low = user_input.lower()
    if "refund" in low or "invoice" in low:
        return {"kind": "handoff", "to": "billing"}
    if "crash" in low or "bug" in low:
        return {"kind": "handoff", "to": "support"}
    return {"kind": "final", "text": "not sure how to help"}


def lost_handoff_policy(user_input):
    return {"kind": "handoff", "to": "legal"}


def confused_policy(user_input):
    return {"kind": "reflect"}


def issue_refund(amount):
    return f"refunded {amount}"


def refund_limit(args):
    return (args["amount"] <= 100, f"amount {args['amount']} over the 100 limit")


def refund_policy_for(amount):
    """Модель просит инструмент, потом финализирует по его результату."""

    def policy(user_input):
        if user_input.startswith("tool "):
            return {"kind": "final", "text": f"done: {user_input}"}
        return {"kind": "tool", "tool": "issue_refund", "args": {"amount": amount}}

    return policy


def missing_tool_policy(user_input):
    return {"kind": "tool", "tool": "wire_transfer", "args": {}}


def recording_policy(calls):
    def policy(user_input):
        calls.append(user_input)
        return {"kind": "final", "text": "ok"}

    return policy


# --------------------------------------------------------- фабрики агентов
def triage_agent():
    billing = make_agent("billing", "handle refunds and invoices", billing_policy)
    support = make_agent("support", "handle bugs and errors", support_policy)
    return make_agent(
        "triage", "route to the right specialist", triage_policy, handoffs=(billing, support)
    )


def refund_agent(amount, guarded):
    tool = {
        "name": "issue_refund",
        "description": "Issue a refund of the given amount.",
        "fn": issue_refund,
    }
    if guarded:
        tool["guardrail"] = refund_limit
    return make_agent("billing", "refund customers", refund_policy_for(amount), tools=(tool,))


def ping_pong_agent():
    """Два агента, которые перекидывают запрос друг другу без конца."""
    pong = make_agent("pong", "bounce back", lambda text: {"kind": "handoff", "to": "ping"})
    ping = make_agent(
        "ping", "bounce", lambda text: {"kind": "handoff", "to": "pong"}, handoffs=(pong,)
    )
    pong["handoffs"] = (ping,)
    return ping


def block_ssn(text):
    return ("ssn" not in text.lower(), "refuses to process social security numbers")


def cap_length(text):
    return (len(text) < 40, f"output is {len(text)} chars")


def always_pass(text):
    return (True, "ok")


# ------------------------------------------------------- handoff_tool_name
def test_handoff_tool_name_uses_the_transfer_prefix():
    assert handoff_tool_name("billing") == "transfer_to_billing"


def test_handoff_tool_name_normalizes_spaces_and_case():
    assert handoff_tool_name("Billing Agent") == "transfer_to_billing_agent"
    assert handoff_tool_name("billing-agent") == "transfer_to_billing_agent"


def test_handoff_tool_name_rejects_a_name_the_model_cannot_type():
    """Модель вызывает инструмент по имени: кириллица и слэш непроизносимы."""
    with pytest.raises(ValueError):
        handoff_tool_name("биллинг")
    with pytest.raises(ValueError):
        handoff_tool_name("billing/agent")


def test_handoff_tool_name_rejects_an_empty_name():
    with pytest.raises(ValueError):
        handoff_tool_name("   ")


# ------------------------------------------------------------- make_agent
def test_make_agent_keeps_instructions_tools_and_handoffs():
    agent = triage_agent()
    assert agent["instructions"] == "route to the right specialist"
    assert agent["tools"] == ()
    assert [target["name"] for target in agent["handoffs"]] == ["billing", "support"]


def test_make_agent_rejects_a_name_that_cannot_become_a_transfer_tool():
    with pytest.raises(ValueError):
        make_agent("billing agent!", "handle refunds", billing_policy)


def test_make_agent_rejects_a_tool_without_a_description():
    """Без описания модель выбирает инструмент наугад."""
    tool = {"name": "issue_refund", "description": "  ", "fn": issue_refund}
    with pytest.raises(ValueError):
        make_agent("billing", "refund customers", billing_policy, tools=(tool,))


def test_make_agent_rejects_a_tool_named_like_a_handoff():
    """Два разных действия под одним именем — модель не различит их вообще."""
    support = make_agent("support", "handle bugs", support_policy)
    tool = {
        "name": "transfer_to_support",
        "description": "Pretend to transfer to support.",
        "fn": issue_refund,
    }
    with pytest.raises(ValueError):
        make_agent("triage", "route", triage_policy, tools=(tool,), handoffs=(support,))


def test_make_agent_rejects_two_handoffs_to_the_same_agent():
    billing = make_agent("billing", "handle refunds", billing_policy)
    with pytest.raises(ValueError):
        make_agent("triage", "route", triage_policy, handoffs=(billing, billing))


# ----------------------------------------------------------- visible_tools
def test_visible_tools_shows_handoffs_next_to_function_tools():
    """Для модели transfer_to_billing — такой же инструмент, как issue_refund."""
    billing = make_agent("billing", "handle refunds", billing_policy)
    tool = {
        "name": "issue_refund",
        "description": "Issue a refund of the given amount.",
        "fn": issue_refund,
    }
    agent = make_agent("triage", "route", triage_policy, tools=(tool,), handoffs=(billing,))
    assert visible_tools(agent) == ["issue_refund", "transfer_to_billing"]


def test_visible_tools_is_empty_without_tools_or_handoffs():
    assert visible_tools(make_agent("billing", "handle refunds", billing_policy)) == []



# --------------------------------------------------------- run_guardrails
def test_run_guardrails_emits_one_span_per_check():
    spans, tripped = run_guardrails((("a", always_pass), ("b", always_pass)), "hi", "input")
    assert [span["name"] for span in spans] == ["input_guardrail.a", "input_guardrail.b"]
    assert tripped is None


def test_run_guardrails_reports_the_stage_and_reason_of_a_trip():
    spans, tripped = run_guardrails((("pii", block_ssn),), "share my ssn", "input")
    assert tripped["stage"] == "input"
    assert tripped["name"] == "pii"
    assert "social security" in tripped["reason"]
    assert spans[0]["attributes"]["passed"] is False


def test_run_guardrails_does_not_run_checks_after_the_first_trip():
    """Сработавший guardrail останавливает цепочку — второй даже не зовётся."""
    seen = []

    def nosy(text):
        seen.append(text)
        return (True, "ok")

    spans, tripped = run_guardrails((("pii", block_ssn), ("nosy", nosy)), "my ssn", "input")
    assert seen == []
    assert len(spans) == 1


def test_run_guardrails_rejects_an_unknown_stage():
    with pytest.raises(ValueError):
        run_guardrails((("pii", block_ssn),), "hi", "inputt")


# --------------------------------------------------------------- run_turn
def test_run_turn_hands_the_original_request_to_the_target_agent():
    """Хендофф передаёт контекст: исходный запрос не теряется по дороге."""
    result = run_turn(triage_agent(), "I need a refund for invoice 4711")
    assert result["agent"] == "billing"
    assert result["output"] == "billing handled: I need a refund for invoice 4711"
    assert result["hops"] == 1



def test_run_turn_traces_the_handoff_under_its_transfer_tool_name():
    spans = run_turn(triage_agent(), "refund please")["spans"]
    assert spans[0]["name"] == "handoff.transfer_to_billing"
    assert spans[0]["attributes"] == {"from": "triage", "to": "billing"}


def test_run_turn_stops_a_handoff_ping_pong_at_the_hop_limit():
    """Handoff drift: A -> B -> A -> B без счётчика не заканчивается никогда."""
    result = run_turn(ping_pong_agent(), "hello", max_hops=3)
    assert result["stop_reason"] == "hop_limit"
    assert result["hops"] == 3


def test_run_turn_reports_a_missing_handoff_without_crashing():
    agent = make_agent("triage", "route", lost_handoff_policy)
    result = run_turn(agent, "anything")
    assert result["stop_reason"] == "unknown_handoff"
    assert "legal" in result["output"]


def test_run_turn_feeds_the_tool_result_back_to_the_model():
    result = run_turn(refund_agent(50, guarded=False), "refund 50 please")
    assert result["output"] == "done: tool issue_refund returned: refunded 50"
    assert [span["name"] for span in result["spans"]] == ["tool.issue_refund", "llm.billing"]


def test_run_turn_lets_a_tool_guardrail_stop_the_call_before_the_side_effect():
    """Guardrail останавливает цепочку ДО вызова fn: денег никто не вернул."""
    receipts = []

    def paying_refund(amount):
        receipts.append(amount)
        return f"refunded {amount}"

    tool = {
        "name": "issue_refund",
        "description": "Issue a refund of the given amount.",
        "fn": paying_refund,
        "guardrail": refund_limit,
    }
    agent = make_agent("billing", "refund customers", refund_policy_for(500), tools=(tool,))
    result = run_turn(agent, "refund 500 please")
    assert result["stop_reason"] == "tool_guardrail"
    assert receipts == []
    assert result["spans"][-1]["name"] == "tool_guardrail.issue_refund"


def test_run_turn_reports_an_unknown_tool_and_an_unknown_decision():
    unknown_tool = run_turn(make_agent("billing", "refunds", missing_tool_policy), "hi")
    assert unknown_tool["stop_reason"] == "unknown_tool"
    confused = run_turn(make_agent("billing", "refunds", confused_policy), "hi")
    assert confused["stop_reason"] == "bad_decision"


# ------------------------------------------------------------ run_guarded
def test_run_guarded_returns_the_agent_output_when_nothing_trips():
    result = run_guarded(
        triage_agent(),
        "I need a refund for invoice 4711",
        input_guardrails=(("pii", block_ssn),),
        output_guardrails=(("pass", always_pass),),
    )
    assert result["tripped"] is None
    assert result["output"].startswith("billing handled")
    assert result["wasted_llm_calls"] == 0


def test_blocking_input_guardrail_spends_nothing_on_the_main_model():
    """blocking=True — проверка первой, основная модель не вызывается вовсе."""
    calls = []
    agent = make_agent("triage", "route", recording_policy(calls))
    result = run_guarded(
        agent, "share my ssn", input_guardrails=(("pii", block_ssn),), blocking=True
    )
    assert result["stop_reason"] == "input_guardrail"
    assert result["llm_calls"] == 0
    assert calls == []


def test_parallel_input_guardrail_wastes_the_work_it_discards():
    """blocking=False — латентность ниже, но токены на выброшенный ответ сгорели."""
    calls = []
    agent = make_agent("triage", "route", recording_policy(calls))
    result = run_guarded(
        agent, "share my ssn", input_guardrails=(("pii", block_ssn),), blocking=False
    )
    assert result["stop_reason"] == "input_guardrail"
    assert result["output"] == ""
    assert result["wasted_llm_calls"] == 1
    assert calls == ["share my ssn"]


def test_output_guardrail_withholds_the_answer_it_rejected():
    result = run_guarded(
        triage_agent(),
        "I need a refund for invoice 4711",
        output_guardrails=(("length", cap_length),),
    )
    assert result["tripped"]["stage"] == "output"
    assert result["output"] == ""
    assert result["wasted_llm_calls"] >= 1
    assert result["spans"][-1]["name"] == "output_guardrail.length"


def test_run_guarded_lets_the_second_turn_see_the_first():
    session = []
    run_guarded(triage_agent(), "I need a refund for invoice 4711", session=session)
    result = run_guarded(triage_agent(), "and the bug too", session=session)
    assert len(session) == 2
    assert "ASSISTANT: billing handled" in result["spans"][-1]["attributes"]["output"]


def test_a_tripped_turn_is_not_written_into_the_session():
    """Ответа не было — истории тоже быть не должно."""
    session = []
    result = run_guarded(
        triage_agent(), "share my ssn", input_guardrails=(("pii", block_ssn),), session=session
    )
    assert result["tripped"]["stage"] == "input"
    assert session == []


# --------------------------------------------------------- session_prompt
def test_session_prompt_without_history_is_just_the_new_message():
    assert session_prompt([], "hi") == "USER: hi"


def test_session_prompt_keeps_prior_turns_in_order():
    session = [
        {"user": "hi", "assistant": "hello"},
        {"user": "refund", "assistant": "done"},
    ]
    assert session_prompt(session, "again").splitlines() == [
        "USER: hi",
        "ASSISTANT: hello",
        "USER: refund",
        "ASSISTANT: done",
        "USER: again",
    ]


def test_session_prompt_drops_the_oldest_turns_past_max_turns():
    session = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(5)]
    prompt = session_prompt(session, "now", max_turns=2)
    assert "q0" not in prompt
    assert "USER: q3" in prompt
    with pytest.raises(ValueError):
        session_prompt(session, "now", max_turns=0)


# ----------------------------------------------------------- redact_spans
def test_redact_spans_swaps_sensitive_content_for_a_reference():
    spans = [{"name": "llm.billing", "attributes": {"output": "card 4111"}}]
    redacted, store = redact_spans(spans)
    assert redacted[0]["attributes"]["output"] == "ref:1"
    assert store == {"ref:1": "card 4111"}


def test_redact_spans_keeps_the_structural_attributes():
    """Имена спанов и passed/from/to секретов не содержат и остаются как есть."""
    spans = [
        {"name": "handoff.transfer_to_billing", "attributes": {"from": "triage", "to": "billing"}},
        {"name": "input_guardrail.pii", "attributes": {"passed": False, "reason": "ssn"}},
    ]
    redacted, store = redact_spans(spans)
    assert redacted == spans
    assert store == {}


def test_redact_spans_does_not_mutate_the_original_trace():
    """Спаны уходят в несколько приёмников: порча на месте роняет второй."""
    spans = [{"name": "tool.issue_refund", "attributes": {"args": {"amount": 50}}}]
    redact_spans(spans)
    assert spans[0]["attributes"]["args"] == {"amount": 50}


def test_redact_spans_covers_every_sensitive_attribute_name():
    spans = [{"name": "s", "attributes": {key: f"secret {key}" for key in SENSITIVE_ATTRIBUTES}}]
    redacted, store = redact_spans(spans)
    assert set(redacted[0]["attributes"].values()) == set(store)
    assert len(store) == len(SENSITIVE_ATTRIBUTES)
