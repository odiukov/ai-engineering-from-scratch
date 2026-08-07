"""Тесты к уроку «Function calling у трёх провайдеров». Правь exercise.py."""

import pytest

from exercise import (
    DEPTH_LIMITS,
    PROVIDERS,
    check_limits,
    declare,
    gemini_schema,
    make_tool_result,
    parse_tool_calls,
    schema_depth,
    tool_choice_for,
)

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city", "units"],
    "additionalProperties": False,
}

WEATHER = {
    "name": "get_weather",
    "description": "Use when the user asks about current conditions. "
    "Do not use for forecasts.",
    "input_schema": WEATHER_SCHEMA,
    "strict": True,
}


def nested(levels):
    """Схема заданной глубины: объект в объекте в объекте..."""
    node = {"type": "string"}
    for _ in range(levels - 1):
        node = {"type": "object", "properties": {"x": node}}
    return node


OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo", "units": "celsius"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
}

ANTHROPIC_RESPONSE = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Looking that up."},
        {
            "type": "tool_use",
            "id": "toolu_xyz",
            "name": "get_weather",
            "input": {"city": "Tokyo", "units": "celsius"},
        },
    ],
    "stop_reason": "tool_use",
}

GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "fc-1",
                            "name": "get_weather",
                            "args": {"city": "Tokyo", "units": "celsius"},
                        }
                    }
                ],
            }
        }
    ]
}

RESPONSES = {
    "openai": OPENAI_RESPONSE,
    "anthropic": ANTHROPIC_RESPONSE,
    "gemini": GEMINI_RESPONSE,
}


# ------------------------------------------------------------- schema_depth
def test_scalar_schema_is_one_level_deep():
    assert schema_depth({"type": "string"}) == 1


def test_empty_object_is_still_one_level_deep():
    assert schema_depth({"type": "object", "properties": {}}) == 1


def test_object_with_scalar_fields_is_two_levels_deep():
    assert schema_depth(WEATHER_SCHEMA) == 2


def test_array_of_objects_costs_two_levels():
    """Массив — уровень, объект внутри него — ещё один."""
    schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"a": {"type": "string"}}},
    }
    assert schema_depth(schema) == 3


def test_depth_counts_the_deepest_branch_not_the_first():
    schema = {
        "type": "object",
        "properties": {"shallow": {"type": "string"}, "deep": nested(4)},
    }
    assert schema_depth(schema) == 5


# ------------------------------------------------------------ gemini_schema
def test_gemini_uppercases_type_names():
    out = gemini_schema({"type": "object", "properties": {"a": {"type": "integer"}}})
    assert out["type"] == "OBJECT"
    assert out["properties"]["a"]["type"] == "INTEGER"


def test_gemini_drops_additional_properties():
    assert "additionalProperties" not in gemini_schema(WEATHER_SCHEMA)


def test_gemini_keeps_enum_values_untouched():
    """Заглавными пишутся ИМЕНА ТИПОВ, а не значения перечисления."""
    out = gemini_schema(WEATHER_SCHEMA)
    assert out["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]


def test_gemini_does_not_mutate_the_source_schema():
    """Та же схема уходит ещё двум провайдерам — портить её нельзя."""
    before = repr(WEATHER_SCHEMA)
    gemini_schema(WEATHER_SCHEMA)
    assert repr(WEATHER_SCHEMA) == before


# ------------------------------------------------------------------ declare
def test_openai_declaration_wraps_the_tool_in_a_function_envelope():
    out = declare("openai", WEATHER)
    assert out["type"] == "function"
    assert out["function"]["name"] == "get_weather"
    assert out["function"]["parameters"] is WEATHER_SCHEMA
    assert out["function"]["strict"] is True


def test_anthropic_declaration_is_flat():
    assert declare("anthropic", WEATHER) == {
        "name": WEATHER["name"],
        "description": WEATHER["description"],
        "input_schema": WEATHER_SCHEMA,
    }


def test_anthropic_declaration_carries_no_strict_flag():
    """У Anthropic такого поля нет: лишний ключ в теле запроса — 400."""
    assert "strict" not in declare("anthropic", WEATHER)


def test_gemini_declaration_nests_under_function_declarations():
    out = declare("gemini", WEATHER)
    assert list(out) == ["functionDeclarations"]
    assert out["functionDeclarations"][0]["parameters"]["type"] == "OBJECT"


def test_all_three_declarations_carry_the_same_name_and_description():
    names = {
        declare("openai", WEATHER)["function"]["name"],
        declare("anthropic", WEATHER)["name"],
        declare("gemini", WEATHER)["functionDeclarations"][0]["name"],
    }
    assert names == {"get_weather"}


def test_unknown_provider_is_refused():
    with pytest.raises(ValueError):
        declare("mistral", WEATHER)


# ----------------------------------------------------------- tool_choice_for
def test_openai_simple_modes_are_bare_strings():
    assert [tool_choice_for("openai", m) for m in ("auto", "none", "required")] == [
        "auto",
        "none",
        "required",
    ]


def test_anthropic_renames_required_to_any():
    assert tool_choice_for("anthropic", "required") == {"type": "any"}


def test_gemini_wraps_modes_in_function_calling_config():
    assert tool_choice_for("gemini", "none") == {
        "function_calling_config": {"mode": "NONE"}
    }


def test_forcing_one_tool_names_it_in_every_provider():
    assert tool_choice_for("openai", "force", "add") == {
        "type": "function",
        "function": {"name": "add"},
    }
    assert tool_choice_for("anthropic", "force", "add") == {"type": "tool", "name": "add"}
    assert tool_choice_for("gemini", "force", "add") == {
        "function_calling_config": {"mode": "ANY", "allowed_function_names": ["add"]}
    }


def test_force_without_a_tool_name_is_refused():
    with pytest.raises(ValueError):
        tool_choice_for("openai", "force")


def test_typo_in_the_mode_is_refused_not_silently_downgraded_to_auto():
    """Молчаливый auto означал бы «вызывай что хочешь» — худший из исходов."""
    with pytest.raises(ValueError):
        tool_choice_for("openai", "requred")


# ---------------------------------------------------------- parse_tool_calls
def test_all_three_shapes_parse_to_the_same_canonical_call():
    parsed = [parse_tool_calls(p, RESPONSES[p])[0] for p in PROVIDERS]
    assert {c["name"] for c in parsed} == {"get_weather"}
    assert all(c["arguments"] == {"city": "Tokyo", "units": "celsius"} for c in parsed)


def test_openai_arguments_string_is_decoded_into_a_dict():
    """OpenAI кладёт JSON строкой; без json.loads получишь str вместо dict."""
    call = parse_tool_calls("openai", OPENAI_RESPONSE)[0]
    assert isinstance(call["arguments"], dict)


def test_ids_keep_their_provider_specific_prefixes():
    assert [parse_tool_calls(p, RESPONSES[p])[0]["id"] for p in PROVIDERS] == [
        "call_abc",
        "toolu_xyz",
        "fc-1",
    ]


def test_anthropic_text_blocks_are_skipped():
    assert len(parse_tool_calls("anthropic", ANTHROPIC_RESPONSE)) == 1


def test_plain_text_answer_yields_no_calls():
    assert parse_tool_calls("openai", {"choices": [{"message": {"content": "hi"}}]}) == []
    assert parse_tool_calls("anthropic", {"content": [{"type": "text", "text": "hi"}]}) == []
    assert parse_tool_calls("gemini", {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}) == []


# ---------------------------------------------------------- make_tool_result
def test_openai_result_is_a_tool_role_message():
    assert make_tool_result("openai", "call_1", "add", "5") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "5",
    }


def test_anthropic_result_comes_from_the_user_role():
    """В Messages API роли tool нет вообще — результат шлёт user."""
    out = make_tool_result("anthropic", "toolu_1", "add", "5")
    assert out["role"] == "user"
    assert out["content"][0]["tool_use_id"] == "toolu_1"


def test_gemini_result_wraps_content_in_an_object():
    out = make_tool_result("gemini", "fc-1", "add", "5")
    assert out["functionResponse"]["response"] == {"result": "5"}
    assert out["functionResponse"]["name"] == "add"


def test_every_provider_echoes_the_call_id_somewhere():
    assert "call_1" in repr(make_tool_result("openai", "call_1", "add", "5"))
    assert "call_1" in repr(make_tool_result("anthropic", "call_1", "add", "5"))
    assert "call_1" in repr(make_tool_result("gemini", "call_1", "add", "5"))


# -------------------------------------------------------------- check_limits
def test_a_small_registry_passes_everywhere():
    assert all(check_limits(p, [WEATHER]) == [] for p in PROVIDERS)


def test_too_many_tools_is_reported_with_both_numbers():
    problems = check_limits("anthropic", [WEATHER] * 65)
    assert problems == ["too many tools: 65 > 64"]


def test_the_same_registry_can_pass_one_provider_and_fail_another():
    """Глубина 6 запрещена у OpenAI и разрешена у Anthropic."""
    deep = {"name": "deep", "description": "d", "input_schema": nested(6)}
    assert check_limits("openai", [deep]) == [
        f"deep: schema depth 6 > {DEPTH_LIMITS['openai']}"
    ]
    assert check_limits("anthropic", [deep]) == []


def test_problems_are_listed_registry_first_then_per_tool():
    deep = {"name": "deep", "description": "d", "input_schema": nested(6)}
    problems = check_limits("openai", [deep] * 129)
    assert problems[0].startswith("too many tools")
    assert all(p.startswith("deep:") for p in problems[1:])
