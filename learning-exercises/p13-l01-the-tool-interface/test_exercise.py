"""Тесты к уроку «Интерфейс инструмента: цикл из четырёх шагов». Правь exercise.py."""

import pytest

from exercise import (
    MAX_TURNS,
    describe_registry,
    execute_call,
    make_tool,
    make_tool_call,
    needs_confirmation,
    run_loop,
    validate_arguments,
)

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city"],
    "additionalProperties": False,
}

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


def registry():
    """Свежий реестр на каждый тест: инструменты — изменяемые словари."""
    return [
        make_tool(
            "add",
            "Use when the user asks for a sum. Do not use for products.",
            ADD_SCHEMA,
            lambda args: args["a"] + args["b"],
        ),
        make_tool(
            "get_weather",
            "Use when the user asks about current conditions. Do not use for forecasts.",
            WEATHER_SCHEMA,
            lambda args: {"city": args["city"], "temp": 21},
        ),
        make_tool(
            "send_email",
            "Use when the user asks to send mail. Do not use for drafts.",
            {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]},
            lambda args: "sent",
            consequential=True,
        ),
    ]


def call(name, arguments, call_id="c1"):
    return {"id": call_id, "name": name, "arguments": arguments}


# ---------------------------------------------------------------- make_tool
def test_make_tool_keeps_all_four_declaration_parts():
    tool = make_tool("add", "Use when ...", ADD_SCHEMA, len)
    assert tool["name"] == "add"
    assert tool["description"] == "Use when ..."
    assert tool["input_schema"] == ADD_SCHEMA
    assert tool["executor"] is len


def test_make_tool_defaults_to_pure():
    assert make_tool("add", "d", ADD_SCHEMA, len)["consequential"] is False


def test_make_tool_copies_the_schema_deeply():
    """Правка исходного словаря не должна менять контракт в реестре."""
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    tool = make_tool("add", "d", schema, len)
    schema["properties"]["a"]["type"] = "string"
    schema["required"].append("b")
    assert tool["input_schema"]["properties"]["a"]["type"] == "integer"
    assert tool["input_schema"]["required"] == ["a"]


# -------------------------------------------------------- describe_registry
def test_describe_registry_exposes_exactly_three_fields():
    described = describe_registry(registry())
    assert all(set(d) == {"name", "description", "input_schema"} for d in described)


def test_describe_registry_hides_the_consequential_flag():
    """Флаг безопасности — решение хоста, модели он не показывается."""
    described = describe_registry(registry())
    assert not any("consequential" in d for d in described)


def test_describe_registry_preserves_order():
    assert [d["name"] for d in describe_registry(registry())] == [
        "add",
        "get_weather",
        "send_email",
    ]


def test_describe_registry_of_empty_registry_is_empty():
    assert describe_registry([]) == []


# ------------------------------------------------------- validate_arguments
def test_valid_arguments_produce_no_problems():
    assert validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo"}) == []


def test_missing_required_property_is_reported():
    assert validate_arguments(WEATHER_SCHEMA, {}) == ["missing required property: city"]


def test_wrong_type_is_reported_with_the_actual_type():
    problems = validate_arguments(WEATHER_SCHEMA, {"city": 5})
    assert problems == ["city: expected string, got int"]


def test_hallucinated_property_is_rejected():
    """Явно закрытая схема не пропускает выдуманное поле."""
    problems = validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo", "hemisphere": "N"})
    assert problems == ["unknown property: hemisphere"]


def test_additional_properties_are_allowed_when_keyword_is_absent():
    """JSON Schema по умолчанию открыта, это не синоним strict mode."""
    schema = dict(WEATHER_SCHEMA)
    schema.pop("additionalProperties")
    assert validate_arguments(schema, {"city": "Tokyo", "hemisphere": "N"}) == []


def test_value_outside_enum_is_rejected():
    problems = validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo", "units": "kelvin"})
    assert len(problems) == 1 and "enum" in problems[0]


def test_boolean_does_not_pass_as_integer():
    """bool — подкласс int, поэтому isinstance(True, int) лжёт."""
    assert validate_arguments(ADD_SCHEMA, {"a": True, "b": 2}) == [
        "a: expected integer, got bool"
    ]


def test_optional_property_may_be_omitted():
    assert validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo", "units": "celsius"}) == []


# ----------------------------------------------------------- make_tool_call
def test_make_tool_call_has_the_three_stable_fields():
    assert make_tool_call("call_1", "add", {"a": 2, "b": 3}) == {
        "id": "call_1",
        "name": "add",
        "arguments": {"a": 2, "b": 3},
    }


def test_make_tool_call_copies_arguments():
    args = {"a": 2, "b": 3}
    made = make_tool_call("call_1", "add", args)
    args["a"] = 99
    assert made["arguments"]["a"] == 2


def test_call_without_an_id_is_refused():
    """Без id результат параллельного вызова не с чем склеить."""
    with pytest.raises(ValueError):
        make_tool_call("", "add", {"a": 1, "b": 2})


# -------------------------------------------------------------- execute_call
def test_execute_call_returns_the_executor_output_as_text():
    msg = execute_call(registry(), call("add", {"a": 2, "b": 3}))
    assert msg["content"] == "5"
    assert msg["is_error"] is False


def test_execute_call_serializes_non_string_results_to_json():
    msg = execute_call(registry(), call("get_weather", {"city": "Tokyo"}))
    assert msg["content"] == '{"city": "Tokyo", "temp": 21}'


def test_unknown_tool_becomes_an_error_message_not_an_exception():
    """Ошибка обязана дойти до модели, а не уронить хост."""
    msg = execute_call(registry(), call("nope", {}))
    assert msg["is_error"] is True
    assert "nope" in msg["content"]


def test_invalid_arguments_are_caught_before_the_executor_runs():
    ran = []
    reg = [make_tool("add", "d", ADD_SCHEMA, lambda args: ran.append(1))]
    msg = execute_call(reg, call("add", {"a": 1}))
    assert msg["is_error"] is True
    assert ran == []


def test_executor_exception_is_wrapped_not_propagated():
    def boom(args):
        raise ZeroDivisionError("division by zero")

    reg = [make_tool("add", "d", {"type": "object", "properties": {}}, boom)]
    msg = execute_call(reg, call("add", {}))
    assert msg["is_error"] is True
    assert msg["content"] == "ZeroDivisionError: division by zero"


def test_every_result_echoes_the_call_id_even_on_failure():
    reg = registry()
    ok = execute_call(reg, call("add", {"a": 1, "b": 2}, call_id="c_ok"))
    bad = execute_call(reg, call("nope", {}, call_id="c_bad"))
    assert (ok["tool_call_id"], bad["tool_call_id"]) == ("c_ok", "c_bad")


# -------------------------------------------------------- needs_confirmation
def test_pure_tool_needs_no_confirmation():
    assert needs_confirmation(registry(), call("add", {"a": 1, "b": 2})) is False


def test_consequential_tool_needs_confirmation():
    assert needs_confirmation(registry(), call("send_email", {"to": "a@b.c"})) is True


def test_unknown_tool_defaults_to_closed_gate():
    """Не знаешь, что это — тем более не знаешь, что оно натворит."""
    assert needs_confirmation(registry(), call("rm_rf", {})) is True


# ------------------------------------------------------------------ run_loop
def test_loop_stops_when_the_model_answers_with_text():
    out = run_loop(registry(), "2+3?", lambda msgs: {"content": "5"})
    assert out["stop_reason"] == "final"
    assert out["turns"] == 1
    assert out["messages"][-1] == {"role": "assistant", "content": "5"}


def test_loop_feeds_the_tool_result_back_before_the_next_turn():
    """Шаг observe: на втором ходу модель уже видит результат инструмента."""
    seen = []

    def decide(messages):
        if messages[-1].get("role") == "tool":
            seen.append(messages[-1]["content"])
            return {"content": "answer: " + messages[-1]["content"]}
        return {"tool_calls": [call("add", {"a": 2, "b": 3})]}

    out = run_loop(registry(), "2+3?", decide)
    assert seen == ["5"]
    assert out["messages"][-1]["content"] == "answer: 5"
    assert out["turns"] == 2


def test_circuit_breaker_stops_a_model_that_never_finishes():
    out = run_loop(
        registry(),
        "loop forever",
        lambda msgs: {"tool_calls": [call("add", {"a": 1, "b": 1})]},
    )
    assert out["stop_reason"] == "max_turns"
    assert out["turns"] == MAX_TURNS


def test_circuit_breaker_bound_is_configurable():
    out = run_loop(
        registry(),
        "loop forever",
        lambda msgs: {"tool_calls": [call("add", {"a": 1, "b": 1})]},
        max_turns=2,
    )
    assert out["turns"] == 2
    assert sum(1 for m in out["messages"] if m.get("role") == "tool") == 2


def test_parallel_calls_in_one_turn_all_get_executed():
    calls = [
        call("add", {"a": 1, "b": 1}, call_id="c1"),
        call("get_weather", {"city": "Tokyo"}, call_id="c2"),
    ]
    steps = iter([{"tool_calls": calls}, {"content": "done"}])
    out = run_loop(registry(), "two things", lambda msgs: next(steps))
    results = [m for m in out["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]


def test_loop_starts_the_transcript_with_the_user_message():
    out = run_loop(registry(), "hello", lambda msgs: {"content": "hi"})
    assert out["messages"][0] == {"role": "user", "content": "hello"}
