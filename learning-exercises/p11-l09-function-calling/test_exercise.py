"""Тесты к уроку «Function calling и вызов инструментов». Правь exercise.py."""

import pytest

from exercise import (
    CALCULATOR_SCHEMA,
    WEATHER_SCHEMA,
    agent_loop,
    execute_tool_call,
    parse_tool_calls,
    register_tool,
    run_tool_calls,
    tool_selection_accuracy,
    validate_arguments,
)

WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy"},
    "london": {"temp_c": 12, "condition": "rainy"},
}

CALLS_MADE = []


def get_weather(city, units="celsius"):
    CALLS_MADE.append(("get_weather", city))
    key = city.lower().strip()
    if key not in WEATHER_DB:
        return {"error": True, "code": "CITY_NOT_FOUND", "message": f"City '{city}' not found."}
    data = dict(WEATHER_DB[key], city=city)
    if units == "fahrenheit":
        data["temp_f"] = round(data.pop("temp_c") * 9 / 5 + 32, 1)
    return data


def calculator(expression, precision=2):
    CALLS_MADE.append(("calculator", expression))
    if not set(expression) <= set("0123456789+-*/.() "):
        raise ValueError(f"Invalid characters in expression: {expression}")
    return {"result": round(eval(expression), precision)}  # noqa: S307 — учебный стенд


def definition(name, description, parameters):
    """Форма записи в реестре — та же, что обязан построить register_tool."""
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


@pytest.fixture
def registry():
    """Реестр собран вручную: он нужен тестам остальных функций даже тогда,
    когда register_tool ещё не написан."""
    CALLS_MADE.clear()
    return {
        "get_weather": {
            "definition": definition("get_weather", "Get current weather for a city.", WEATHER_SCHEMA),
            "function": get_weather,
        },
        "calculator": {
            "definition": definition("calculator", "Evaluate a math expression.", CALCULATOR_SCHEMA),
            "function": calculator,
        },
    }


# ----------------------------------------------------------- register_tool
def test_registered_definition_has_the_openai_shape():
    reg = register_tool({}, "get_weather", "Get weather.", WEATHER_SCHEMA, get_weather)
    built = reg["get_weather"]["definition"]
    assert built["type"] == "function"
    assert built["function"]["name"] == "get_weather"
    assert built["function"]["parameters"] is WEATHER_SCHEMA


def test_registered_description_is_kept_verbatim():
    """Описание — промпт для выбора инструмента, обрезать его нельзя."""
    text = "Get current weather for a city. Returns temperature in Celsius."
    reg = register_tool({}, "get_weather", text, WEATHER_SCHEMA, get_weather)
    assert reg["get_weather"]["definition"]["function"]["description"] == text


def test_registry_holds_the_callable_next_to_the_schema():
    reg = register_tool({}, "calculator", "Math.", CALCULATOR_SCHEMA, calculator)
    assert reg["calculator"]["function"] is calculator


def test_register_tool_returns_the_same_registry():
    reg = {}
    assert register_tool(reg, "a", "d", WEATHER_SCHEMA, get_weather) is reg


# ------------------------------------------------------- validate_arguments
def test_valid_arguments_produce_no_errors():
    assert validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo"}) == []


def test_missing_required_argument_is_reported():
    assert validate_arguments(WEATHER_SCHEMA, {}) == ["Missing required argument: city"]


def test_unknown_argument_is_reported():
    errors = validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo", "colour": "red"})
    assert errors == ["Unknown argument: colour"]


def test_wrong_type_is_reported():
    errors = validate_arguments(CALCULATOR_SCHEMA, {"expression": 123})
    assert len(errors) == 1
    assert "expected string" in errors[0]


def test_value_outside_the_enum_is_reported():
    errors = validate_arguments(WEATHER_SCHEMA, {"city": "Tokyo", "units": "kelvin"})
    assert len(errors) == 1
    assert "kelvin" in errors[0]


def test_boolean_is_not_accepted_as_an_integer():
    """isinstance(True, int) — True, и без отдельной проверки булево проедет."""
    errors = validate_arguments(CALCULATOR_SCHEMA, {"expression": "1+1", "precision": True})
    assert len(errors) == 1
    assert "expected integer" in errors[0]


def test_integer_is_accepted_where_a_number_is_expected():
    schema = {"type": "object", "properties": {"t": {"type": "number"}}, "required": []}
    assert validate_arguments(schema, {"t": 3}) == []


def test_every_problem_is_reported_not_just_the_first():
    errors = validate_arguments(WEATHER_SCHEMA, {"units": "kelvin", "colour": "red"})
    assert len(errors) == 3


def test_non_object_arguments_are_rejected():
    assert validate_arguments(WEATHER_SCHEMA, ["Tokyo"]) == [
        "Arguments must be an object, got list"
    ]


# --------------------------------------------------------- parse_tool_calls
def test_parse_wraps_a_single_object_into_a_list():
    assert parse_tool_calls('{"name": "get_weather", "arguments": {"city": "Tokyo"}}') == [
        {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    ]


def test_parse_accepts_parallel_calls_as_a_list():
    raw = '[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]'
    assert [c["name"] for c in parse_tool_calls(raw)] == ["a", "b"]


def test_missing_arguments_default_to_an_empty_object():
    assert parse_tool_calls('{"name": "ping"}') == [{"name": "ping", "arguments": {}}]


def test_parse_unwraps_openai_style_string_arguments():
    """OpenAI кладёт arguments строкой с JSON внутри — её надо распарсить."""
    raw = '{"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"}'
    assert parse_tool_calls(raw) == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def test_parse_preserves_the_call_identifier():
    raw = '{"id": "call_weather", "name": "get_weather", "arguments": {"city": "Tokyo"}}'
    assert parse_tool_calls(raw)[0]["call_id"] == "call_weather"


def test_broken_json_raises_value_error():
    with pytest.raises(ValueError):
        parse_tool_calls('{"name": "get_weather"')


def test_call_without_a_name_raises_value_error():
    with pytest.raises(ValueError):
        parse_tool_calls('{"arguments": {"city": "Tokyo"}}')


def test_scalar_instead_of_a_call_raises_value_error():
    with pytest.raises(ValueError):
        parse_tool_calls("[1, 2, 3]")


# ------------------------------------------------------- execute_tool_call
def test_execute_runs_the_registered_function(registry):
    out = execute_tool_call(registry, {"name": "get_weather", "arguments": {"city": "Tokyo"}})
    assert out["ok"] is True
    assert out["result"]["temp_c"] == 18


def test_execute_passes_optional_arguments_through(registry):
    out = execute_tool_call(
        registry, {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "fahrenheit"}}
    )
    assert out["result"]["temp_f"] == pytest.approx(64.4)


def test_execute_carries_the_call_identifier_to_the_result(registry):
    call = {
        "call_id": "call_tokyo",
        "name": "get_weather",
        "arguments": {"city": "Tokyo"},
    }
    assert execute_tool_call(registry, call)["call_id"] == "call_tokyo"


def test_unknown_tool_is_refused_by_the_allowlist(registry):
    """Модель может назвать что угодно — вызвать можно только то, что в реестре."""
    out = execute_tool_call(registry, {"name": "shell", "arguments": {"cmd": "rm -rf /"}})
    assert out["ok"] is False
    assert out["result"]["code"] == "UNKNOWN_TOOL"


def test_invalid_arguments_are_refused_before_the_call(registry):
    out = execute_tool_call(registry, {"name": "get_weather", "arguments": {}})
    assert out["result"]["code"] == "INVALID_ARGUMENTS"
    assert CALLS_MADE == []


def test_exception_inside_a_tool_becomes_a_structured_error(registry):
    """Модель не умеет читать traceback — она умеет читать error и исправляться."""
    out = execute_tool_call(
        registry, {"name": "calculator", "arguments": {"expression": "__import__('os')"}}
    )
    assert out["ok"] is False
    assert out["result"]["code"] == "TOOL_ERROR"
    assert "ValueError" in out["result"]["message"]


def test_tool_level_error_is_not_a_dispatcher_error(registry):
    """Город не найден — инструмент отработал штатно, ok остаётся True."""
    out = execute_tool_call(registry, {"name": "get_weather", "arguments": {"city": "Mars"}})
    assert out["ok"] is True
    assert out["result"]["code"] == "CITY_NOT_FOUND"


# ---------------------------------------------------------- run_tool_calls
def test_parallel_calls_all_run(registry):
    calls = [
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "London"}},
    ]
    results = run_tool_calls(registry, calls)
    assert [r["result"]["city"] for r in results] == ["Tokyo", "London"]


def test_calls_over_the_budget_are_not_executed(registry):
    """CALL_LIMIT существует, чтобы зациклившаяся модель не съела бюджет."""
    calls = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}] * 5
    results = run_tool_calls(registry, calls, max_calls=2)
    assert [r["ok"] for r in results] == [True, True, False, False, False]
    assert results[-1]["result"]["code"] == "CALL_LIMIT"
    assert len(CALLS_MADE) == 2


def test_one_failing_call_does_not_stop_the_others(registry):
    calls = [
        {"name": "nope", "arguments": {}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
    ]
    assert [r["ok"] for r in run_tool_calls(registry, calls)] == [False, True]


# -------------------------------------------------------------- agent_loop
def weather_once(message, conversation):
    """Заглушка модели: зовёт погоду, пока результата ещё нет в истории."""
    if any(m["role"] == "tool" for m in conversation):
        return []
    return [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def never_stops(message, conversation):
    return [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def no_tools(message, conversation):
    return []


def test_loop_without_tools_leaves_the_conversation_alone(registry):
    out = agent_loop(registry, "Tell me a joke", no_tools)
    assert out["iterations"] == 0
    assert out["conversation"] == [{"role": "user", "content": "Tell me a joke"}]


def test_loop_appends_the_assistant_turn_and_the_tool_result(registry):
    out = agent_loop(registry, "Weather in Tokyo?", weather_once)
    roles = [m["role"] for m in out["conversation"]]
    assert roles == ["user", "assistant", "tool"]
    assert "18" in out["conversation"][-1]["content"]


def test_loop_stops_after_the_tool_result_is_available(registry):
    assert agent_loop(registry, "Weather in Tokyo?", weather_once)["iterations"] == 1


def test_a_model_stuck_in_a_loop_is_cut_off(registry):
    """Без max_iterations это бесконечный цикл и бесконечный счёт за API."""
    out = agent_loop(registry, "Weather?", never_stops, max_iterations=3)
    assert out["iterations"] == 3
    assert len(out["results"]) == 3


def test_loop_collects_every_result(registry):
    def two_cities(message, conversation):
        if any(m["role"] == "tool" for m in conversation):
            return []
        return [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "get_weather", "arguments": {"city": "London"}},
        ]

    out = agent_loop(registry, "Weather in Tokyo and London?", two_cities)
    assert out["iterations"] == 1
    assert len(out["results"]) == 2


def test_parallel_tool_results_link_back_to_their_exact_calls(registry):
    def two_cities(message, conversation):
        if any(m["role"] == "tool" for m in conversation):
            return []
        return [
            {
                "call_id": "call_tokyo",
                "name": "get_weather",
                "arguments": {"city": "Tokyo"},
            },
            {
                "call_id": "call_london",
                "name": "get_weather",
                "arguments": {"city": "London"},
            },
        ]

    conversation = agent_loop(registry, "Weather in two cities?", two_cities)["conversation"]
    assistant_ids = [c["call_id"] for c in conversation[1]["tool_calls"]]
    result_ids = [m["tool_call_id"] for m in conversation[2:]]
    assert result_ids == assistant_ids == ["call_tokyo", "call_london"]


# ------------------------------------------------- tool_selection_accuracy
def keyword_decide(message, conversation):
    msg = message.lower()
    if "weather" in msg:
        return [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    if "calculate" in msg:
        return [{"name": "calculator", "arguments": {"expression": "1+1"}}]
    return []


def test_selection_accuracy_is_one_when_every_guess_is_right():
    cases = [
        ("What is the weather in Tokyo?", "get_weather"),
        ("Calculate 2 + 2", "calculator"),
        ("Tell me a joke", None),
    ]
    report = tool_selection_accuracy(keyword_decide, cases)
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["errors"] == []


def test_selection_accuracy_records_the_confusing_query():
    cases = [("How much is 2 + 2?", "calculator")]
    report = tool_selection_accuracy(keyword_decide, cases)
    assert report["correct"] == 0
    assert report["errors"][0]["actual"] is None


def test_selection_accuracy_of_an_empty_suite_is_zero():
    assert tool_selection_accuracy(keyword_decide, [])["accuracy"] == pytest.approx(0.0)
