"""Тесты к уроку «Structured outputs: JSON Schema, валидация, повторные попытки». Правь exercise.py."""

import pytest

from exercise import (
    VALUE_TOKENS,
    extract_with_retry,
    model_to_schema,
    next_valid_tokens,
    parse_llm_json,
    python_type_to_schema,
    strip_code_fence,
    validate,
)

PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "price": {"type": "number", "minimum": 0},
        "in_stock": {"type": "boolean"},
        "status": {"type": "string", "enum": ["in_stock", "out_of_stock", "preorder"]},
        "categories": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["product", "price", "in_stock"],
    "additionalProperties": False,
}

VALID_PRODUCT = {"product": "Sony WH-1000XM5", "price": 348.0, "in_stock": True}


# ---------------------------------------------------------- strip_code_fence
def test_strip_code_fence_unwraps_a_tagged_block():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_unwraps_an_untagged_block():
    assert strip_code_fence("```\n[1, 2]\n```") == "[1, 2]"


def test_strip_code_fence_leaves_bare_json_alone():
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


# ------------------------------------------------------------ parse_llm_json
def test_parse_llm_json_reads_clean_json():
    assert parse_llm_json('{"a": 1}') == ({"a": 1}, None)


def test_parse_llm_json_survives_a_preamble():
    """«Here's the JSON:» — самая частая примесь в ответе модели."""
    data, error = parse_llm_json('Here is the JSON: {"a": 1}')
    assert (data, error) == ({"a": 1}, None)


def test_parse_llm_json_survives_a_code_fence():
    data, error = parse_llm_json('```json\n{"a": [1, 2]}\n```')
    assert (data, error) == ({"a": [1, 2]}, None)


def test_parse_llm_json_reports_an_error_instead_of_raising():
    data, error = parse_llm_json("not json at all")
    assert data is None
    assert isinstance(error, str) and error


# ------------------------------------------------------------------ validate
def test_validate_accepts_a_conforming_object():
    assert validate(VALID_PRODUCT, PRODUCT_SCHEMA) == []


def test_validate_catches_a_missing_required_field():
    errors = validate({"product": "x", "in_stock": True}, PRODUCT_SCHEMA)
    assert len(errors) == 1 and "price" in errors[0]


def test_validate_catches_an_extra_field():
    """Пропущенное поле ловят все. Лишнее — только при additionalProperties: False."""
    errors = validate({**VALID_PRODUCT, "colour": "black"}, PRODUCT_SCHEMA)
    assert len(errors) == 1 and "colour" in errors[0]


def test_validate_allows_extra_fields_when_the_schema_is_open():
    open_schema = {k: v for k, v in PRODUCT_SCHEMA.items() if k != "additionalProperties"}
    assert validate({**VALID_PRODUCT, "colour": "black"}, open_schema) == []


def test_validate_catches_a_value_below_minimum():
    errors = validate({**VALID_PRODUCT, "price": -5.0}, PRODUCT_SCHEMA)
    assert len(errors) == 1 and "minimum" in errors[0]


def test_validate_catches_a_value_outside_the_enum():
    """Модель отвечает 'available' вместо 'in_stock' — по смыслу верно, по схеме нет."""
    errors = validate({**VALID_PRODUCT, "status": "available"}, PRODUCT_SCHEMA)
    assert len(errors) == 1 and "available" in errors[0]


def test_validate_applies_enum_to_integer_values():
    schema = {"type": "integer", "enum": [1, 2, 3]}
    assert validate(4, schema) == ["$: 4 not in enum [1, 2, 3]"]


def test_validate_applies_enum_to_boolean_values():
    schema = {"type": "boolean", "enum": [True]}
    assert validate(False, schema) == ["$: False not in enum [True]"]


def test_validate_does_not_accept_a_boolean_as_a_number():
    """Ловушка: isinstance(True, int) истинно, значит True пролезет в integer."""
    errors = validate({**VALID_PRODUCT, "price": True}, PRODUCT_SCHEMA)
    assert len(errors) == 1


def test_validate_reports_the_path_inside_a_nested_array():
    errors = validate({**VALID_PRODUCT, "categories": ["audio", 7]}, PRODUCT_SCHEMA)
    assert len(errors) == 1 and "[1]" in errors[0]


def test_validate_collects_several_errors_at_once():
    """Отдавать все ошибки разом — иначе модель чинит их по одной за вызов."""
    errors = validate({"price": -1.0, "extra": 1}, PRODUCT_SCHEMA)
    assert len(errors) >= 3


# --------------------------------------------------- python_type_to_schema
def test_python_type_to_schema_maps_int_to_integer():
    assert python_type_to_schema(int) == {"type": "integer"}


def test_python_type_to_schema_maps_float_to_number():
    """integer и number в JSON Schema — разные типы, 1.5 не пройдёт как integer."""
    assert python_type_to_schema(float) == {"type": "number"}


def test_python_type_to_schema_rejects_an_unsupported_type():
    with pytest.raises(ValueError):
        python_type_to_schema(set)


# -------------------------------------------------------------- model_to_schema
def test_model_to_schema_marks_fields_required_by_default():
    schema = model_to_schema({"product": {"type": str}})
    assert schema["required"] == ["product"]


def test_model_to_schema_skips_optional_fields_in_required():
    schema = model_to_schema(
        {"product": {"type": str}, "categories": {"type": list, "required": False}}
    )
    assert schema["required"] == ["product"]
    assert "categories" in schema["properties"]


def test_model_to_schema_carries_constraints_into_the_property():
    schema = model_to_schema({"price": {"type": float, "minimum": 0}})
    assert schema["properties"]["price"] == {"type": "number", "minimum": 0}


def test_model_to_schema_output_is_accepted_by_the_validator():
    """Генератор схем и валидатор обязаны говорить на одном языке."""
    schema = model_to_schema({"product": {"type": str}, "price": {"type": float}})
    assert validate({"product": "x", "price": 1.5}, schema) == []
    assert validate({"product": "x"}, schema) != []


# ------------------------------------------------------------ next_valid_tokens
def test_next_valid_tokens_starts_an_object():
    assert next_valid_tokens("") == ["{"]


def test_next_valid_tokens_after_an_open_brace_forbids_a_digit():
    """Ключ объекта — только строка, поэтому цифра тут невозможна."""
    assert "0-9" not in next_valid_tokens("{")


def test_next_valid_tokens_after_a_colon_expects_a_value():
    assert next_valid_tokens('{"price":') == VALUE_TOKENS


def test_next_valid_tokens_after_a_comma_cannot_close_the_object():
    """Висячая запятая невалидна в JSON, хотя в Python она законна."""
    assert "}" not in next_valid_tokens('{"a": 1,')


def test_next_valid_tokens_on_a_complete_object_is_eos():
    assert next_valid_tokens('{"price": 348}') == ["<EOS>"]


def test_next_valid_tokens_lets_a_number_continue_or_end():
    tokens = next_valid_tokens('{"price": 348')
    assert "0-9" in tokens and "}" in tokens


# ---------------------------------------------------------- extract_with_retry
def test_extract_with_retry_returns_on_the_first_success():
    result = extract_with_retry("text", PRODUCT_SCHEMA, lambda *_: '{"product": "x", "price": 1, "in_stock": true}')
    assert result["attempts"] == 1
    assert result["errors"] == []
    assert result["data"]["product"] == "x"


def test_extract_with_retry_feeds_errors_back_to_the_model():
    """Ошибки валидации уходят в следующий вызов — это и есть дешёвая коррекция."""
    seen = []

    def call_model(text, attempt, errors):
        seen.append(list(errors))
        if attempt == 0:
            return '{"product": "x", "price": -1, "in_stock": true}'
        return '{"product": "x", "price": 1, "in_stock": true}'

    result = extract_with_retry("text", PRODUCT_SCHEMA, call_model)
    assert result["attempts"] == 2
    assert seen[0] == []
    assert seen[1] and "minimum" in seen[1][0]


def test_extract_with_retry_gives_up_after_max_retries():
    result = extract_with_retry("t", PRODUCT_SCHEMA, lambda *_: "{}", max_retries=2)
    assert result["data"] is None
    assert result["attempts"] == 2
    assert result["errors"]


def test_extract_with_retry_retries_on_unparseable_output_too():
    """Сломанный JSON и валидный-но-неподходящий JSON лечатся одинаково."""

    def call_model(text, attempt, errors):
        return "sorry, I cannot" if attempt == 0 else '{"product": "x", "price": 1, "in_stock": true}'

    assert extract_with_retry("t", PRODUCT_SCHEMA, call_model)["attempts"] == 2
