"""Тесты к уроку «Продакшн-рантаймы агентов». Правь exercise.py."""

import pytest

from exercise import (
    estimate_runtime_cost,
    handle_request,
    pick_runtime,
    route_model,
    run_workflow,
    split_records,
    stub_model,
    typed_tool_call,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ROUTER = {
    "models": {"anthropic/claude-x": "anthropic", "openai/gpt-x": "openai"},
    "aliases": {"fast": "anthropic/claude-x", "broken": "also-an-alias"},
}


def counting_factory():
    """Фабрика агентов + счётчик вызовов: сколько агентов реально создали."""
    created = []

    def factory():
        agent = {"instructions": "be brief", "tools": ["search"]}
        created.append(agent)
        return agent

    return factory, created


# ------------------------------------------------------------- stub_model
def test_stub_model_is_deterministic():
    assert stub_model("hi", "anthropic/claude-x") == stub_model("hi", "anthropic/claude-x")


def test_stub_model_answer_carries_the_resolved_model():
    assert stub_model("hi", "anthropic/claude-x") == "claude-x:2:209"


def test_stub_model_separates_providers_of_the_same_prompt():
    a = stub_model("hi", "anthropic/claude-x")
    b = stub_model("hi", "openai/gpt-x")
    assert a != b


# ------------------------------------------------------------- route_model
def test_route_model_resolves_a_full_id():
    assert route_model(ROUTER, "openai/gpt-x") == {
        "provider": "openai",
        "model": "openai/gpt-x",
    }


def test_route_model_resolves_an_alias():
    assert route_model(ROUTER, "fast") == {
        "provider": "anthropic",
        "model": "anthropic/claude-x",
    }


def test_route_model_refuses_an_alias_pointing_at_an_alias():
    """Цепочки алиасов однажды замкнутся в кольцо — второй прыжок запрещён."""
    with pytest.raises(ValueError):
        route_model(ROUTER, "broken")


def test_route_model_does_not_invent_a_default_model():
    with pytest.raises(KeyError):
        route_model(ROUTER, "mistral/large")


# --------------------------------------------------------- typed_tool_call
SEARCH_TOOL = {
    "name": "search",
    "schema": {"query": str, "limit": int},
    "handler": lambda p: f"{p['query'].upper()}x{p['limit']}",
}


def test_typed_tool_call_runs_handler_on_valid_payload():
    assert typed_tool_call(SEARCH_TOOL, {"query": "ai", "limit": 3}) == "AIx3"


def test_typed_tool_call_rejects_missing_field():
    with pytest.raises(ValueError):
        typed_tool_call(SEARCH_TOOL, {"query": "ai"})


def test_typed_tool_call_rejects_bool_where_int_is_expected():
    """isinstance(True, int) == True — вечная ловушка, схема не должна её пропустить."""
    with pytest.raises(TypeError):
        typed_tool_call(SEARCH_TOOL, {"query": "ai", "limit": True})


def test_typed_tool_call_rejects_extra_field():
    with pytest.raises(ValueError):
        typed_tool_call(SEARCH_TOOL, {"query": "ai", "limit": 3, "cursor": "z"})


def test_typed_tool_call_does_not_run_handler_on_bad_payload():
    """Проверка схемы идёт ДО обработчика, иначе он уже сходил в базу."""
    ran = []
    tool = {
        "name": "t",
        "schema": {"n": int},
        "handler": lambda p: ran.append(p) or "ok",
    }
    with pytest.raises(TypeError):
        typed_tool_call(tool, {"n": "1"})
    assert ran == []


# -------------------------------------------------------- handle_request
def test_handle_request_returns_the_resolved_provider():
    factory, _ = counting_factory()
    out = handle_request({}, "s1", "hi", factory, ROUTER, "fast")
    assert out["provider"] == "anthropic"
    assert out["model"] == "anthropic/claude-x"


def test_handle_request_builds_a_fresh_agent_every_time():
    factory, created = counting_factory()
    store = {}
    handle_request(store, "s1", "a", factory, ROUTER, "fast")
    handle_request(store, "s1", "b", factory, ROUTER, "fast")
    assert len(created) == 2


def test_agent_does_not_survive_the_request():
    """Stateless session-scoped: агент видит только свой запрос."""
    factory, _ = counting_factory()
    store = {}
    first = handle_request(store, "s1", "a", factory, ROUTER, "fast")
    second = handle_request(store, "s1", "b", factory, ROUTER, "fast")
    assert first["agent_seen"] == ["a"]
    assert second["agent_seen"] == ["b"]


def test_session_state_survives_in_the_store():
    """А вот история переживает запрос — потому что живёт снаружи агента."""
    factory, _ = counting_factory()
    store = {}
    first = handle_request(store, "s1", "a", factory, ROUTER, "fast")
    second = handle_request(store, "s1", "b", factory, ROUTER, "fast")
    assert (first["history_len"], second["history_len"]) == (2, 4)


def test_handle_request_answer_depends_on_the_history():
    """Второй ответ в сессии не равен первому — контекст восстановлен из store."""
    factory, _ = counting_factory()
    store = {}
    first = handle_request(store, "s1", "a", factory, ROUTER, "fast")
    second = handle_request(store, "s1", "a", factory, ROUTER, "fast")
    assert first["answer"] != second["answer"]


# ---------------------------------------------------------- run_workflow
def test_run_workflow_pipes_output_into_the_next_step():
    steps = [("inc", lambda x: x + 1), ("dbl", lambda x: x * 2)]
    assert run_workflow(steps, 3)["output"] == 8


def test_run_workflow_of_no_steps_returns_the_payload():
    assert run_workflow([], 3) == {"output": 3, "trace": [], "failed": None}


def test_failed_step_stops_the_workflow():
    """Шаг после сбоя не должен получить мусор и «починить» падение."""
    ran = []
    steps = [
        ("boom", lambda x: 1 / 0),
        ("after", lambda x: ran.append(x) or x),
    ]
    result = run_workflow(steps, 3)
    assert result["failed"] == "boom"
    assert ran == []


def test_failed_step_keeps_the_trace_of_what_did_run():
    steps = [("ok", lambda x: x + 1), ("boom", lambda x: x["nope"])]
    result = run_workflow(steps, 3)
    assert result["trace"] == [("ok", 4)]
    assert result["output"] is None


# --------------------------------------------------------- split_records
ROUTING = {"memory": "postgres", "workflows": "postgres", "observability": "clickhouse"}


def test_split_records_sends_each_kind_to_its_backend():
    records = [
        {"kind": "memory", "data": 1},
        {"kind": "observability", "data": 2},
    ]
    out = split_records(records, ROUTING)
    assert sorted(out) == ["clickhouse", "postgres"]


def test_split_records_can_share_one_backend_between_kinds():
    records = [{"kind": "memory", "data": 1}, {"kind": "workflows", "data": 2}]
    out = split_records(records, ROUTING)
    assert list(out) == ["postgres"] and len(out["postgres"]) == 2


def test_split_records_preserves_order_inside_a_backend():
    records = [{"kind": "memory", "data": i} for i in range(4)]
    out = split_records(records, ROUTING)
    assert [r["data"] for r in out["postgres"]] == [0, 1, 2, 3]


def test_split_records_refuses_a_kind_without_a_backend():
    """Иначе observability тихо уедет в базу памяти и раздует её."""
    with pytest.raises(ValueError):
        split_records([{"kind": "observability", "data": 1}], {"memory": "postgres"})


# --------------------------------------------------- estimate_runtime_cost
def test_instantiation_of_a_thousand_agents_is_two_milliseconds():
    assert estimate_runtime_cost(1000, 0.0)["instantiation_ms"] == APPROX(2.0)


def test_instantiation_share_is_negligible_when_the_model_call_dominates():
    """«2 μs» ничего не решает, если один вызов модели идёт 800 ms."""
    assert estimate_runtime_cost(1000, 800.0)["instantiation_share"] < 1e-3


def test_instantiation_share_is_everything_without_model_calls():
    assert estimate_runtime_cost(1000, 0.0)["instantiation_share"] == APPROX(1.0)


def test_zero_agents_do_not_divide_by_zero():
    assert estimate_runtime_cost(0, 800.0)["instantiation_share"] == APPROX(0.0)


def test_a_slow_runtime_flips_the_verdict():
    """Тот же workload, но инстанциация 5 ms вместо 2 μs — и доля уже заметна."""
    fast = estimate_runtime_cost(1000, 1.0)["instantiation_share"]
    slow = estimate_runtime_cost(1000, 1.0, instantiation_us=5000.0)["instantiation_share"]
    assert fast < 0.01 < 0.5 < slow


# ------------------------------------------------------------ pick_runtime
def test_python_without_graph_state_picks_agno():
    assert pick_runtime({"language": "python", "needs_durable_graph_state": False}) == "agno"


def test_typescript_picks_mastra():
    assert pick_runtime({"language": "typescript", "needs_durable_graph_state": False}) == "mastra"


def test_durable_graph_state_outranks_the_language():
    """Ни Agno, ни Mastra не про долгоживущее состояние графа."""
    assert pick_runtime({"language": "python", "needs_durable_graph_state": True}) == "langgraph"
    assert pick_runtime({"language": "typescript", "needs_durable_graph_state": True}) == "langgraph"


def test_unsupported_language_is_an_error():
    with pytest.raises(ValueError):
        pick_runtime({"language": "rust", "needs_durable_graph_state": False})
