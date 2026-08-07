"""Тесты к уроку «Память репозитория и durable state». Правь exercise.py."""

import json

import pytest

from exercise import (
    SCHEMA_VERSION,
    STATE_SCHEMA,
    TEMP_SUFFIX,
    SchemaError,
    atomic_write,
    commit_state,
    forget_stale,
    load_state,
    migrate_state,
    remember,
    validate,
)

STATE_PATH = "agent_state.json"


def good_state():
    return {
        "schema_version": SCHEMA_VERSION,
        "active_task_id": "T-001",
        "touched_files": ["app.py"],
        "risks": [],
        "next_action": "прочитать обработчик /signup",
    }


# ----------------------------------------------------------------- validate
def test_a_valid_state_passes():
    assert validate(good_state(), STATE_SCHEMA) is None


def test_missing_required_field_is_refused():
    state = good_state()
    del state["next_action"]
    with pytest.raises(SchemaError):
        validate(state, STATE_SCHEMA)


def test_invented_field_is_refused():
    """Каждый агент, придумавший своё поле, ломает всех читателей файла."""
    state = good_state()
    state["mood"] = "confident"
    with pytest.raises(SchemaError):
        validate(state, STATE_SCHEMA)


def test_task_id_must_match_the_pattern():
    state = good_state()
    state["active_task_id"] = "T-bogus"
    with pytest.raises(SchemaError):
        validate(state, STATE_SCHEMA)


def test_null_task_id_is_allowed_because_the_type_list_permits_it():
    state = good_state()
    state["active_task_id"] = None
    assert validate(state, STATE_SCHEMA) is None


def test_bool_does_not_sneak_through_as_integer():
    """isinstance(True, int) — правда, поэтому проверку типа пишут руками."""
    with pytest.raises(SchemaError):
        validate(True, {"type": "integer"})


def test_wrong_item_type_inside_an_array_is_refused():
    state = good_state()
    state["touched_files"] = ["app.py", 42]
    with pytest.raises(SchemaError):
        validate(state, STATE_SCHEMA)


# ------------------------------------------------------------- atomic_write
def test_atomic_write_puts_the_content_at_the_path():
    assert atomic_write({}, STATE_PATH, "{}") == {STATE_PATH: "{}"}


def test_atomic_write_leaves_no_temp_file_behind():
    result = atomic_write({}, STATE_PATH, "{}")
    assert not any(name.endswith(TEMP_SUFFIX) for name in result)


def test_a_crash_mid_write_leaves_the_old_content_intact():
    """Наполовину записанный файл состояния хуже, чем отсутствие файла."""
    fs = {STATE_PATH: "старое содержимое"}
    with pytest.raises(OSError):
        atomic_write(fs, STATE_PATH, "новое", crash_after_temp=True)
    assert fs == {STATE_PATH: "старое содержимое"}


def test_atomic_write_does_not_mutate_the_input_filesystem():
    fs = {STATE_PATH: "old"}
    atomic_write(fs, STATE_PATH, "new")
    assert fs == {STATE_PATH: "old"}


# ----------------------------------------------------------------- remember
def test_a_new_fact_is_appended():
    memory = remember([], "python", "3.12", 100)
    assert memory == [
        {"key": "python", "value": "3.12", "first_seen": 100, "last_seen": 100}
    ]


def test_writing_a_known_fact_does_not_create_a_duplicate():
    """Из дублей вырастает память, которой агент перестаёт доверять."""
    memory = remember([], "python", "3.12", 100)
    memory = remember(memory, "python", "3.12", 200)
    assert len(memory) == 1


def test_a_repeated_fact_keeps_first_seen_and_bumps_last_seen():
    memory = remember(remember([], "python", "3.12", 100), "python", "3.12", 200)
    assert (memory[0]["first_seen"], memory[0]["last_seen"]) == (100, 200)


def test_a_changed_fact_overwrites_the_value_in_place():
    memory = remember(remember([], "python", "3.11", 100), "python", "3.12", 200)
    assert len(memory) == 1
    assert memory[0]["value"] == "3.12"


def test_remember_does_not_mutate_the_input_memory():
    memory = remember([], "python", "3.12", 100)
    remember(memory, "python", "3.13", 200)
    assert memory[0]["value"] == "3.12"


def test_different_keys_live_side_by_side_in_write_order():
    memory = remember(remember([], "python", "3.12", 1), "node", "22", 2)
    assert [item["key"] for item in memory] == ["python", "node"]


# -------------------------------------------------------------- forget_stale
def test_a_fresh_fact_survives():
    memory = remember([], "python", "3.12", 100)
    assert forget_stale(memory, 150, ttl=100) == memory


def test_a_stale_fact_is_dropped():
    """Старый факт про версию Python переживает апгрейд и уводит не туда."""
    memory = remember([], "python", "3.11", 0)
    assert forget_stale(memory, 1000, ttl=100) == []


def test_the_ttl_boundary_is_inclusive():
    memory = remember([], "python", "3.12", 0)
    assert len(forget_stale(memory, 100, ttl=100)) == 1
    assert len(forget_stale(memory, 101, ttl=100)) == 0


def test_confirming_a_fact_saves_it_from_the_sweep():
    """Дедупликация и устаревание работают в паре: подтверждение продлевает жизнь."""
    memory = remember([], "python", "3.12", 0)
    memory = remember(memory, "python", "3.12", 900)
    assert len(forget_stale(memory, 1000, ttl=200)) == 1


# ------------------------------------------------- commit_state / load_state
def test_state_survives_a_commit_and_load_round_trip():
    fs = commit_state({}, STATE_PATH, good_state())
    assert load_state(fs, STATE_PATH) == good_state()


def test_a_bad_write_is_a_refused_write():
    state = good_state()
    state["active_task_id"] = "nope"
    with pytest.raises(SchemaError):
        commit_state({}, STATE_PATH, state)


def test_a_refused_write_leaves_the_filesystem_alone():
    fs = commit_state({}, STATE_PATH, good_state())
    broken = good_state()
    broken["schema_version"] = 99
    with pytest.raises(SchemaError):
        commit_state(fs, STATE_PATH, broken)
    assert load_state(fs, STATE_PATH) == good_state()


def test_loading_a_file_from_an_older_schema_version_is_refused():
    fs = {STATE_PATH: json.dumps({"schema_version": 1, "blockers": []})}
    with pytest.raises(SchemaError):
        load_state(fs, STATE_PATH)


def test_a_missing_state_file_is_not_silently_an_empty_state():
    """Тихая подстановка пустышки стирает работу предыдущей сессии."""
    with pytest.raises(KeyError):
        load_state({}, STATE_PATH)


# --------------------------------------------------------------- migrate_state
def test_migration_renames_blockers_to_risks():
    old = {
        "schema_version": 1,
        "active_task_id": "T-001",
        "touched_files": [],
        "blockers": ["ждём ключ от staging"],
        "next_action": "ждать",
    }
    new = migrate_state(old)
    assert new["risks"] == ["ждём ключ от staging"]
    assert "blockers" not in new


def test_migrated_state_passes_the_current_schema():
    old = {
        "schema_version": 1,
        "active_task_id": None,
        "touched_files": [],
        "blockers": [],
        "next_action": "",
    }
    assert validate(migrate_state(old), STATE_SCHEMA) is None


def test_migration_is_idempotent():
    """Её вешают на старт каждой сессии, поэтому второй прогон обязан быть no-op."""
    once = migrate_state({"schema_version": 1, "blockers": ["x"]})
    assert migrate_state(once) == once


def test_migration_does_not_mutate_the_input():
    old = {"schema_version": 1, "blockers": ["x"]}
    migrate_state(old)
    assert old == {"schema_version": 1, "blockers": ["x"]}


def test_unknown_schema_version_refuses_to_migrate():
    with pytest.raises(SchemaError):
        migrate_state({"schema_version": 99})
