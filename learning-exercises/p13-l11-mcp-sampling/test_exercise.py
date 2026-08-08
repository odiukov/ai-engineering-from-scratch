"""Тесты к уроку «MCP sampling». Правь exercise.py."""

import pytest

from exercise import (
    SAMPLING_METHOD,
    SamplingBudgetExceeded,
    create_message_request,
    model_preferences,
    pick_model,
    review_request,
    run_sampling_loop,
    sampling_result,
    spend_sample,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

CATALOG = [
    {"name": "haiku", "cost": 1.0, "speed": 1.0, "intelligence": 0.3},
    {"name": "sonnet", "cost": 0.6, "speed": 0.6, "intelligence": 0.8},
    {"name": "opus", "cost": 0.1, "speed": 0.2, "intelligence": 1.0},
]


def canned_client(texts, stop_reasons=None):
    """Фейковый клиент: отдаёт заготовленные ответы по порядку."""
    seen = []

    def client(request):
        index = len(seen)
        seen.append(request)
        reason = (stop_reasons or ["endTurn"] * len(texts))[index]
        return sampling_result(request["id"], texts[index], "haiku", reason)

    client.seen = seen
    return client


# --------------------------------------------------------- model_preferences
def test_priorities_are_independent_not_normalized_to_a_sum():
    prefs = model_preferences(0.8, 0.7, 0.9)
    assert prefs["costPriority"] == APPROX(0.8)
    assert prefs["speedPriority"] == APPROX(0.7)
    assert prefs["intelligencePriority"] == APPROX(0.9)


def test_all_zero_priorities_are_valid():
    assert model_preferences(0, 0, 0) == {
        "costPriority": 0, "speedPriority": 0, "intelligencePriority": 0
    }


def test_priority_outside_zero_to_one_is_an_error():
    with pytest.raises(ValueError):
        model_preferences(-1, 1, 1)
    with pytest.raises(ValueError):
        model_preferences(0, 1.1, 0)


def test_hints_are_objects_with_a_name_field():
    prefs = model_preferences(0, 0, 1, hints=["claude-3-5-sonnet"])
    assert prefs["hints"] == [{"name": "claude-3-5-sonnet"}]


def test_empty_hints_are_not_sent_at_all():
    assert "hints" not in model_preferences(1, 1, 1)


# --------------------------------------------------- create_message_request
def test_request_uses_the_sampling_method_and_the_given_id():
    request = create_message_request(42, ["Pick five files"])
    assert request["method"] == SAMPLING_METHOD and request["id"] == 42


def test_plain_string_becomes_a_typed_user_message():
    params = create_message_request(1, ["Pick five files"])["params"]
    assert params["messages"] == [
        {"role": "user", "content": {"type": "text", "text": "Pick five files"}}]


def test_prepared_message_objects_pass_through():
    message = {"role": "assistant", "content": {"type": "text", "text": "ok"}}
    params = create_message_request(1, [message])["params"]
    assert params["messages"] == [message]


def test_absent_optional_fields_are_omitted_not_nulled():
    params = create_message_request(1, ["hi"])["params"]
    assert "systemPrompt" not in params and "modelPreferences" not in params
    assert "tools" not in params


def test_unknown_include_context_mode_is_refused():
    """Опечатка тут молча превратилась бы в «сервер просил лишнего»."""
    with pytest.raises(ValueError):
        create_message_request(1, ["hi"], include_context="everything")


def test_non_positive_max_tokens_is_refused():
    with pytest.raises(ValueError):
        create_message_request(1, ["hi"], max_tokens=0)


def test_tools_in_sampling_are_carried_through():
    """SEP-1577: клиент прогонит цикл вызова инструментов внутри sampling."""
    tools = [{"name": "fetch_url", "description": "d", "inputSchema": {}}]
    assert create_message_request(1, ["hi"], tools=tools)["params"]["tools"] == tools


# ---------------------------------------------------------------- pick_model
def test_intelligence_priority_picks_the_smartest_model():
    assert pick_model(CATALOG, model_preferences(0, 0, 1)) == "opus"


def test_cost_priority_picks_the_cheapest_model():
    assert pick_model(CATALOG, model_preferences(1, 0, 0)) == "haiku"


def test_hint_is_a_preference_not_only_a_tie_breaker():
    assert pick_model(CATALOG, model_preferences(0, 0, 1, hints=["haiku"])) == "haiku"


def test_hints_match_model_name_substrings_in_preference_order():
    prefs = model_preferences(0, 0, 1, hints=["missing", "son"])
    assert pick_model(CATALOG, prefs) == "sonnet"


def test_empty_catalog_is_an_error():
    with pytest.raises(ValueError):
        pick_model([], model_preferences(1, 1, 1))


# ----------------------------------------------------------- sampling_result
def test_completion_always_comes_back_as_assistant():
    result = sampling_result(42, "done", "haiku")["result"]
    assert result["role"] == "assistant"
    assert result["content"] == {"type": "text", "text": "done"}


def test_result_reports_the_model_actually_used():
    """Она вполне может не совпасть с hints сервера."""
    assert sampling_result(1, "x", "haiku")["result"]["model"] == "haiku"


def test_result_carries_the_request_id():
    assert sampling_result(7, "x", "haiku")["id"] == 7


def test_provider_specific_stop_reason_is_allowed():
    assert sampling_result(1, "x", "haiku", stop_reason="length")["result"]["stopReason"] == "length"


def test_tool_use_is_a_standard_stop_reason():
    assert sampling_result(1, "x", "haiku", stop_reason="toolUse")["result"]["stopReason"] == "toolUse"


def test_empty_stop_reason_is_refused():
    with pytest.raises(ValueError):
        sampling_result(1, "x", "haiku", stop_reason="")


# ------------------------------------------------------------- spend_sample
def test_first_spend_counts_one():
    budget = {}
    assert spend_sample(budget, "tool:summarize", 5) == 1
    assert budget["tool:summarize"] == 1


def test_spending_past_the_limit_raises():
    budget = {"tool:summarize": 5}
    with pytest.raises(SamplingBudgetExceeded):
        spend_sample(budget, "tool:summarize", 5)


def test_keys_have_independent_budgets():
    budget = {"tool:a": 5}
    assert spend_sample(budget, "tool:b", 5) == 1


def test_zero_limit_denies_the_very_first_call():
    with pytest.raises(SamplingBudgetExceeded):
        spend_sample({}, "tool:a", 0)


# --------------------------------------------------------- run_sampling_loop
def test_loop_collects_the_text_of_every_round():
    client = canned_client(["files: a, b", "the summary"])
    out = run_sampling_loop([["pick"], ["summarize"]], client)
    assert out["texts"] == ["files: a, b", "the summary"] and out["rounds"] == 2


def test_every_round_uses_a_distinct_request_id():
    """Два sampling с одним id — клиент не поймёт, на какой отвечает."""
    client = canned_client(["a", "b"])
    run_sampling_loop([["pick"], ["summarize"]], client)
    ids = [request["id"] for request in client.seen]
    assert len(set(ids)) == 2


def test_truncated_completion_stops_the_loop():
    """Рассуждать поверх обрубка бессмысленно."""
    client = canned_client(["cut off", "never asked"], ["maxTokens", "endTurn"])
    out = run_sampling_loop([["pick"], ["summarize"]], client)
    assert out["rounds"] == 1 and out["stopReason"] == "maxTokens"


def test_runaway_loop_hits_the_budget():
    client = canned_client(["x"] * 10)
    with pytest.raises(SamplingBudgetExceeded):
        run_sampling_loop([["go"]] * 10, client, limit=3)


def test_preferences_reach_the_client_untouched():
    prefs = model_preferences(0, 0, 1, hints=["opus"])
    client = canned_client(["a"])
    run_sampling_loop([["pick"]], client, preferences=prefs)
    assert client.seen[0]["params"]["modelPreferences"] == prefs


# ---------------------------------------------------------- review_request
def test_review_shows_what_the_server_asks_the_model():
    request = create_message_request(1, ["Summarize the repo"], system_prompt="be terse")
    assert review_request(request)["asks"] == "Summarize the repo"


def test_ordinary_request_is_low_risk():
    assert review_request(create_message_request(1, ["hi"]))["risk"] == "low"


def test_cross_server_context_is_high_risk():
    """includeContext="allServers" протаскивает переписку чужих серверов."""
    request = create_message_request(1, ["hi"], include_context="allServers")
    assert review_request(request)["risk"] == "high"


def test_tools_in_the_request_are_high_risk():
    tools = [{"name": "fetch_url", "description": "d", "inputSchema": {}}]
    review = review_request(create_message_request(1, ["hi"], tools=tools))
    assert review["risk"] == "high" and review["tools"] == ["fetch_url"]
