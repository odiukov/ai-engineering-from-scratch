"""Тесты к уроку «Структурированный вывод». Правь exercise.py."""

import json

import pytest

from exercise import (
    extract_with_retry,
    make_strict,
    parse_output,
    retry_prompt,
    schema_from_fields,
    strict_mode_problems,
    validate,
)

INVOICE = {
    "type": "object",
    "properties": {
        "customer": {"type": "string", "minLength": 1, "maxLength": 200},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "pattern": "^[A-Z0-9-]+$"},
                    "qty": {"type": "integer", "minimum": 1},
                },
                "required": ["sku", "qty"],
                "additionalProperties": False,
            },
        },
        "total_usd": {"type": "number", "minimum": 0},
        "currency": {"type": "string", "enum": ["USD", "EUR", "INR"]},
    },
    "required": ["customer", "line_items", "total_usd", "currency"],
    "additionalProperties": False,
}

GOOD_INVOICE = {
    "customer": "Acme Corp",
    "line_items": [{"sku": "ABC-123", "qty": 2}],
    "total_usd": 99.98,
    "currency": "USD",
}


def paths(errors):
    """Только пути: тексты сообщений в тестах не фиксируем."""
    return [path for path, _ in errors]


# ----------------------------------------------------------------- validate
def test_a_conforming_document_has_no_errors():
    assert validate(INVOICE, GOOD_INVOICE) == []


def test_missing_required_field_is_reported_at_its_path():
    doc = dict(GOOD_INVOICE)
    del doc["currency"]
    assert paths(validate(INVOICE, doc)) == ["$.currency"]


def test_error_path_points_into_the_array_element():
    """Путь важнее текста: по нему модель понимает, что именно чинить."""
    doc = json.loads(json.dumps(GOOD_INVOICE))
    doc["line_items"][0]["qty"] = 0
    assert paths(validate(INVOICE, doc)) == ["$.line_items[0].qty"]


def test_additional_property_is_rejected_when_the_door_is_shut():
    doc = dict(GOOD_INVOICE, discount=0.1)
    assert paths(validate(INVOICE, doc)) == ["$.discount"]


def test_enum_and_pattern_close_the_set_of_allowed_values():
    outside_enum = dict(GOOD_INVOICE, currency="RUB")
    assert paths(validate(INVOICE, outside_enum)) == ["$.currency"]

    bad_sku = json.loads(json.dumps(GOOD_INVOICE))
    bad_sku["line_items"][0]["sku"] = "abc_123"
    assert paths(validate(INVOICE, bad_sku)) == ["$.line_items[0].sku"]


def test_boolean_does_not_pass_as_a_number():
    """bool — подкласс int, поэтому isinstance(True, int) лжёт."""
    assert validate({"type": "number"}, True) != []


def test_wrong_type_stops_further_checks_on_that_node():
    """Строка вместо объекта даёт одну претензию, а не список про каждое поле."""
    assert len(validate(INVOICE, "not an invoice")) == 1


# ------------------------------------------------------- strict_mode_problems
def test_a_strict_ready_schema_has_no_problems():
    assert strict_mode_problems(INVOICE) == []


def test_optional_property_breaks_strict_mode():
    """В strict mode необязательных полей не бывает вовсе."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert paths(strict_mode_problems(schema)) == ["$.b"]


def test_open_object_breaks_strict_mode():
    schema = {"type": "object", "properties": {}, "required": []}
    assert paths(strict_mode_problems(schema)) == ["$"]


def test_ref_is_rejected():
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "required": ["a"],
        "additionalProperties": False,
    }
    assert paths(strict_mode_problems(schema)) == ["$.a"]


def test_problems_are_found_inside_array_items():
    schema = {
        "type": "object",
        "properties": {"xs": {"type": "array", "items": {"type": "object",
                                                         "properties": {"a": {"type": "string"}},
                                                         "required": []}}},
        "required": ["xs"],
        "additionalProperties": False,
    }
    assert "$.xs[]" in paths(strict_mode_problems(schema))


# --------------------------------------------------------------- make_strict
def test_make_strict_fixes_everything_it_can():
    loose = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "xs": {"type": "array", "items": {"type": "object",
                                              "properties": {"b": {"type": "integer"}}}},
        },
    }
    assert strict_mode_problems(make_strict(loose)) == []


def test_make_strict_lists_every_property_as_required():
    out = make_strict({"type": "object", "properties": {"a": {"type": "string"}}})
    assert out["required"] == ["a"]
    assert out["additionalProperties"] is False


def test_make_strict_does_not_mutate_the_source_schema():
    """Та же схема уходит в Anthropic, где strict mode не нужен."""
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    make_strict(loose)
    assert loose == {"type": "object", "properties": {"a": {"type": "string"}}}


def test_make_strict_keeps_the_data_valid():
    """Ужесточение схемы не должно ломать документ, который ей и так отвечал."""
    strict = make_strict(INVOICE)
    assert validate(strict, GOOD_INVOICE) == []


def test_make_strict_cannot_remove_a_ref():
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
    }
    assert strict_mode_problems(make_strict(schema)) != []


# --------------------------------------------------------- schema_from_fields
def test_python_types_map_to_json_schema_types():
    out = schema_from_fields({"customer": str, "total_usd": float, "qty": int})
    assert out["properties"] == {
        "customer": {"type": "string"},
        "total_usd": {"type": "number"},
        "qty": {"type": "integer"},
    }


def test_generated_schema_is_strict_ready():
    assert strict_mode_problems(schema_from_fields({"a": str, "b": int})) == []


def test_bool_maps_to_boolean_not_integer():
    """bool — подкласс int; проверка по порядку типов обязана ловить его первым."""
    assert schema_from_fields({"paid": bool})["properties"]["paid"] == {"type": "boolean"}


def test_field_order_is_preserved_in_required():
    assert schema_from_fields({"z": str, "a": str})["required"] == ["z", "a"]


def test_unsupported_type_is_refused():
    with pytest.raises(ValueError):
        schema_from_fields({"when": complex})


# -------------------------------------------------------------- parse_output
def test_valid_json_that_matches_the_schema_is_ok():
    out = parse_output({"content": json.dumps(GOOD_INVOICE)}, INVOICE)
    assert out["kind"] == "ok"
    assert out["value"] == GOOD_INVOICE


def test_broken_json_is_a_parse_error_not_a_violation():
    out = parse_output({"content": '{"customer": "Acme",}'}, INVOICE)
    assert out["kind"] == "parse_error"
    assert out["value"] is None


def test_parsed_but_wrong_is_a_violation():
    out = parse_output({"content": json.dumps({"customer": "Acme"})}, INVOICE)
    assert out["kind"] == "violation"
    assert out["value"] == {"customer": "Acme"}
    assert out["errors"] != []


def test_refusal_is_a_typed_outcome_not_a_failure():
    out = parse_output({"refusal": "This is a poem, not an invoice."}, INVOICE)
    assert out["kind"] == "refusal"
    assert out["reason"] == "This is a poem, not an invoice."
    assert out["errors"] == []


def test_refusal_wins_over_whatever_is_in_content():
    """Отказ проверяется первым: разбирать content при нём не надо."""
    out = parse_output({"refusal": "no", "content": "{{{"}, INVOICE)
    assert out["kind"] == "refusal"


# -------------------------------------------------------------- retry_prompt
def test_retry_prompt_lists_path_and_message():
    text = retry_prompt([("$.total_usd", "below minimum 0")])
    assert "$.total_usd" in text and "below minimum 0" in text


def test_retry_prompt_mentions_every_error():
    text = retry_prompt([("$.a", "missing required field"), ("$.b", "expected string")])
    assert text.count("\n- ") == 2


def test_no_errors_means_no_prompt():
    assert retry_prompt([]) == ""


# --------------------------------------------------------- extract_with_retry
def test_first_try_success_costs_one_attempt():
    out = extract_with_retry(lambda fb: {"content": json.dumps(GOOD_INVOICE)}, INVOICE)
    assert out["kind"] == "ok"
    assert out["attempts"] == 1


def test_refusal_is_not_retried():
    """Модель не ошиблась — она сказала, что задача не решается."""
    calls = []

    def model(feedback):
        calls.append(feedback)
        return {"refusal": "The email is a song lyric."}

    out = extract_with_retry(model, INVOICE)
    assert out["kind"] == "refusal"
    assert out["attempts"] == 1
    assert calls == [None]


def test_violation_is_retried_with_the_error_text_injected():
    seen = []

    def model(feedback):
        seen.append(feedback)
        if feedback is None:
            return {"content": json.dumps({"customer": "Acme"})}
        return {"content": json.dumps(GOOD_INVOICE)}

    out = extract_with_retry(model, INVOICE)
    assert out["kind"] == "ok"
    assert out["attempts"] == 2
    assert seen[0] is None
    assert "$.currency" in seen[1]


def test_hopeless_output_stops_at_max_attempts():
    calls = []

    def model(feedback):
        calls.append(feedback)
        return {"content": "not json at all"}

    out = extract_with_retry(model, INVOICE, max_attempts=3)
    assert out["kind"] == "parse_error"
    assert out["attempts"] == 3
    assert len(calls) == 3


def test_max_attempts_of_one_means_no_retry():
    calls = []

    def model(feedback):
        calls.append(feedback)
        return {"content": "{}"}

    out = extract_with_retry(model, INVOICE, max_attempts=1)
    assert out["attempts"] == 1
    assert len(calls) == 1
