"""Тесты к уроку «Tool use и function calling». Правь exercise.py."""

from itertools import permutations

import pytest

from exercise import (
    MIN_DESCRIPTION_WORDS,
    breaker_allows,
    build_registry,
    coerce_value,
    dispatch,
    dispatch_many,
    make_tool,
    tool_catalog,
    validate_args,
)

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["open", "closed", "pending"]}},
    "required": ["status"],
}

DIVIDE_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}

SCORE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number", "minimum": 0, "maximum": 100}},
    "required": ["score"],
}


def add(a, b):
    return a + b


def classify(status):
    return f"classified as {status}"


def divide(a, b):
    return a / b


def recorder():
    """Исполнитель, который запоминает каждый свой вызов, — детектор побочных эффектов."""
    seen = []

    def record(a, b):
        seen.append((a, b))
        return "recorded"

    return record, seen


def sample_registry():
    """Реестр из трёх инструментов. Строится ВНУТРИ теста, а не при импорте."""
    return build_registry(
        [
            make_tool("add", "Add two integers a and b together.", ADD_SCHEMA, add),
            make_tool(
                "classify",
                "Classify a ticket status into an allowed label.",
                CLASSIFY_SCHEMA,
                classify,
            ),
            make_tool(
                "divide", "Divide number a by number b and return the ratio.", DIVIDE_SCHEMA, divide
            ),
        ]
    )


# ------------------------------------------------------------- coerce_value
def test_coerce_value_repairs_int_written_as_string():
    assert coerce_value("5", {"type": "integer"}) == (5, None)


def test_coerce_value_rejects_bool_where_integer_is_required():
    """bool — подкласс int, поэтому True просачивается без явной проверки."""
    value, error = coerce_value(True, {"type": "integer"})
    assert error == "expected integer, got bool"


def test_coerce_value_widens_int_to_number():
    value, error = coerce_value(2, {"type": "number"})
    assert error is None
    assert isinstance(value, float)


def test_coerce_value_refuses_to_stringify_a_number():
    """Обратная коэрсия спрятала бы реальный баг в промпте."""
    value, error = coerce_value(5, {"type": "string"})
    assert (value, error) == (5, "expected string, got int")


def test_coerce_value_reports_unparsable_string():
    assert coerce_value("x", {"type": "integer"}) == (
        "x",
        "cannot coerce string 'x' to integer",
    )


# ------------------------------------------------------------ validate_args
def test_validate_args_returns_a_clean_coerced_dict():
    assert validate_args({"a": "5", "b": 2}, ADD_SCHEMA) == ({"a": 5, "b": 2}, [])


def test_validate_args_lists_every_missing_required_field():
    """Все ошибки сразу — модель починит вызов за одну попытку, а не за две."""
    validated, errors = validate_args({}, ADD_SCHEMA)
    assert errors == ["missing required: a", "missing required: b"]


def test_validate_args_rejects_unknown_field():
    validated, errors = validate_args({"a": 1, "b": 2, "c": 3}, ADD_SCHEMA)
    assert errors == ["unknown field: c"]


def test_validate_args_rejects_value_outside_enum():
    validated, errors = validate_args({"status": "in_progress"}, CLASSIFY_SCHEMA)
    assert len(errors) == 1
    assert "not in ['open', 'closed', 'pending']" in errors[0]


def test_validate_args_enforces_minimum_and_maximum():
    assert validate_args({"score": 50}, SCORE_SCHEMA)[1] == []
    assert validate_args({"score": 150}, SCORE_SCHEMA)[1] != []
    assert validate_args({"score": -1}, SCORE_SCHEMA)[1] != []


def test_validate_args_keeps_a_broken_field_out_of_validated():
    """Исполнитель не должен увидеть значение, которое не прошло проверку."""
    validated, errors = validate_args({"a": 1, "b": "oops"}, ADD_SCHEMA)
    assert errors
    assert "b" not in validated


# ---------------------------------------------------------------- make_tool
def test_make_tool_keeps_schema_and_executor():
    tool = make_tool("add", "Add two integers a and b together.", ADD_SCHEMA, add)
    assert tool["name"] == "add"
    assert tool["input_schema"] is ADD_SCHEMA
    assert tool["executor"] is add


def test_make_tool_rejects_empty_name():
    with pytest.raises(ValueError):
        make_tool("  ", "Add two integers a and b together.", ADD_SCHEMA, add)


def test_make_tool_rejects_a_description_the_model_cannot_choose_by():
    """«Add.» вместо «когда применять» — это wrong-tool-picked в BFCL."""
    short = " ".join(["word"] * (MIN_DESCRIPTION_WORDS - 1))
    with pytest.raises(ValueError):
        make_tool("add", short, ADD_SCHEMA, add)


def test_make_tool_rejects_schema_without_properties():
    with pytest.raises(ValueError):
        make_tool("add", "Add two integers a and b together.", {"type": "object"}, add)


def test_make_tool_rejects_non_positive_timeout():
    with pytest.raises(ValueError):
        make_tool("add", "Add two integers a and b together.", ADD_SCHEMA, add, timeout_s=0)


# ------------------------------------------------------------ build_registry
def test_build_registry_indexes_tools_by_name():
    registry = sample_registry()
    assert sorted(registry) == ["add", "classify", "divide"]


def test_build_registry_rejects_duplicate_names():
    """Второй инструмент с тем же именем молча стал бы недостижим."""
    tool = make_tool("add", "Add two integers a and b together.", ADD_SCHEMA, add)
    with pytest.raises(ValueError):
        build_registry([tool, tool])


# -------------------------------------------------------------- tool_catalog
def test_tool_catalog_is_sorted_by_name():
    """Стабильный порядок — условие попадания промпта в кэш."""
    registry = build_registry(
        [
            make_tool("divide", "Divide number a by number b.", DIVIDE_SCHEMA, divide),
            make_tool("add", "Add two integers a and b together.", ADD_SCHEMA, add),
        ]
    )
    assert [entry["name"] for entry in tool_catalog(registry)] == ["add", "divide"]


def test_tool_catalog_hides_the_executor():
    """Модели нужны имя, описание и схема; исполнитель — деталь рантайма."""
    for entry in tool_catalog(sample_registry()):
        assert set(entry) == {"name", "description", "input_schema"}


# ------------------------------------------------------------------ dispatch
def test_dispatch_returns_the_tool_output_as_string():
    result = dispatch(sample_registry(), {"tool_use_id": "u1", "name": "add", "args": {"a": 2, "b": 3}})
    assert result == {"tool_use_id": "u1", "ok": True, "content": "5"}


def test_dispatch_reports_unknown_tool_as_an_observation():
    """Придуманный инструмент не должен ронять цикл агента."""
    result = dispatch(
        sample_registry(), {"tool_use_id": "u9", "name": "subtract", "args": {"a": 1, "b": 2}}
    )
    assert result["ok"] is False
    assert result["content"] == "error: unknown tool 'subtract'"


def test_dispatch_rejects_an_extra_argument_before_executing():
    """Вызов с лишним аргументом отвергается ДО побочного эффекта."""
    record, seen = recorder()
    registry = build_registry(
        [make_tool("record", "Record the pair a and b somewhere.", ADD_SCHEMA, record)]
    )
    result = dispatch(
        registry, {"tool_use_id": "u1", "name": "record", "args": {"a": 1, "b": 2, "c": 3}}
    )
    assert result["ok"] is False
    assert "unknown field: c" in result["content"]
    assert seen == []


def test_dispatch_passes_coerced_arguments_to_the_executor():
    record, seen = recorder()
    registry = build_registry(
        [make_tool("record", "Record the pair a and b somewhere.", ADD_SCHEMA, record)]
    )
    dispatch(registry, {"tool_use_id": "u1", "name": "record", "args": {"a": "4", "b": 5}})
    assert seen == [(4, 5)]


def test_dispatch_captures_an_exception_from_the_executor():
    result = dispatch(
        sample_registry(), {"tool_use_id": "u1", "name": "divide", "args": {"a": 1, "b": 0}}
    )
    assert result["ok"] is False
    assert result["content"].startswith("execution error: ZeroDivisionError")


def test_dispatch_echoes_the_correlation_id():
    for tool_use_id in ("toolu_01", "toolu_02"):
        result = dispatch(
            sample_registry(), {"tool_use_id": tool_use_id, "name": "add", "args": {"a": 1, "b": 1}}
        )
        assert result["tool_use_id"] == tool_use_id


# ------------------------------------------------------------- dispatch_many
def _parallel_calls():
    return [
        {"tool_use_id": "u1", "name": "add", "args": {"a": 2, "b": 3}},
        {"tool_use_id": "u2", "name": "classify", "args": {"status": "in_progress"}},
        {"tool_use_id": "u3", "name": "add", "args": {"a": "4", "b": 5}},
    ]


def test_dispatch_many_gives_the_same_answer_for_any_completion_order():
    """Параллельный turn не имеет права зависеть от того, кто ответил первым."""
    registry = sample_registry()
    calls = _parallel_calls()
    baseline = dispatch_many(registry, calls)
    for order in permutations(range(len(calls))):
        assert dispatch_many(registry, calls, order) == baseline


def test_dispatch_many_matches_every_result_to_its_own_call():
    registry = sample_registry()
    calls = _parallel_calls()
    results = dispatch_many(registry, calls, completion_order=(2, 0, 1))
    assert [r["tool_use_id"] for r in results] == ["u1", "u2", "u3"]
    assert results[0]["content"] == "5"
    assert results[1]["ok"] is False
    assert results[2]["content"] == "9"


def test_dispatch_many_rejects_duplicate_correlation_ids():
    """Один tool_use_id на два вызова — прямой путь к ответу не от того инструмента."""
    registry = sample_registry()
    calls = [
        {"tool_use_id": "u1", "name": "add", "args": {"a": 1, "b": 1}},
        {"tool_use_id": "u1", "name": "add", "args": {"a": 2, "b": 2}},
    ]
    with pytest.raises(ValueError):
        dispatch_many(registry, calls)


def test_dispatch_many_rejects_a_completion_order_that_drops_a_call():
    with pytest.raises(ValueError):
        dispatch_many(sample_registry(), _parallel_calls(), completion_order=(0, 1))


# ------------------------------------------------------------ breaker_allows
def test_breaker_allows_while_under_threshold():
    assert breaker_allows([(1.0, False), (2.0, False)], 3.0) is True


def test_breaker_opens_after_consecutive_failures():
    assert breaker_allows([(1.0, False), (2.0, False), (3.0, False)], 4.0) is False


def test_breaker_counter_is_reset_by_a_success():
    """Инструмент, падающий раз в сотню вызовов, не должен закрыться навсегда."""
    outcomes = [(1.0, False), (2.0, False), (3.0, True), (4.0, False), (5.0, False)]
    assert breaker_allows(outcomes, 6.0) is True


def test_breaker_closes_again_after_cooldown():
    outcomes = [(1.0, False), (2.0, False), (3.0, False)]
    assert breaker_allows(outcomes, 62.9) is False
    assert breaker_allows(outcomes, 63.1) is True


def test_breaker_opens_after_three_failing_dispatches():
    """Сквозной сценарий: наблюдения от dispatch кормят breaker."""
    registry = sample_registry()
    outcomes = []
    for at in (10.0, 20.0, 30.0):
        result = dispatch(
            registry, {"tool_use_id": f"u{at}", "name": "divide", "args": {"a": 1, "b": 0}}
        )
        outcomes.append((at, result["ok"]))
    assert breaker_allows(outcomes, now=31.0) is False
    assert breaker_allows(outcomes, now=95.0) is True
