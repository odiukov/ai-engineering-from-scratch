"""Тесты к уроку «Блоки памяти и sleep-time compute». Правь exercise.py."""

import pytest

from exercise import (
    block_append,
    block_replace,
    dedup_archival,
    invalidate_contradicted,
    make_block,
    near_limit,
    sleep_time_pass,
    summarize_block,
)


def rec(rid, text, valid=True):
    """Запись archival для теста."""
    return {"rid": rid, "text": text, "valid": valid}


# --------------------------------------------------------------- make_block
def test_make_block_starts_at_version_one_with_empty_history():
    block = make_block("human", "facts about the user", limit=180)
    assert (block["version"], block["history"], block["value"]) == (1, (), "")


def test_make_block_keeps_the_description_the_model_routes_by():
    """По description модель решает, в какой блок писать факт."""
    block = make_block("persona", "the agent's self-concept", limit=160)
    assert block["description"] == "the agent's self-concept"
    assert block["limit"] == 160


def test_make_block_refuses_a_value_longer_than_the_limit():
    with pytest.raises(ValueError):
        make_block("human", "facts", limit=4, value="name=ava")


# ------------------------------------------------------------- block_append
def test_block_append_writes_and_bumps_the_version():
    block = block_append(make_block("human", "facts", limit=50), "name=ava")
    assert (block["value"], block["version"]) == ("name=ava", 2)


def test_block_append_joins_with_a_single_space():
    block = make_block("human", "facts", limit=50, value="name=ava")
    assert block_append(block, "city=Berlin")["value"] == "name=ava city=Berlin"


def test_block_append_refuses_to_overflow_instead_of_truncating():
    """Block bloat: сначала summarize, потом запись. Молча обрезать нельзя."""
    block = make_block("human", "facts", limit=20)
    with pytest.raises(ValueError):
        block_append(block, "x" * 50)


def test_block_append_keeps_the_previous_value_in_history():
    """Диффы блоков — единственный способ отладить «почему агент забыл X»."""
    block = block_append(make_block("human", "facts", limit=50), "name=ava")
    block = block_append(block, "city=Berlin")
    assert block["history"] == ("", "name=ava")


def test_block_append_leaves_the_input_block_alone():
    block = make_block("human", "facts", limit=50)
    block_append(block, "name=ava")
    assert (block["value"], block["version"]) == ("", 1)


# ------------------------------------------------------------ block_replace
def test_block_replace_updates_the_fact():
    block = make_block("human", "facts", limit=50, value="city=Berlin")
    assert block_replace(block, "Berlin", "Lisbon")["value"] == "city=Lisbon"


def test_block_replace_refuses_when_the_old_text_is_absent():
    block = make_block("human", "facts", limit=50, value="city=Berlin")
    with pytest.raises(ValueError):
        block_replace(block, "Paris", "Lisbon")


def test_block_replace_refuses_when_the_new_text_overflows():
    block = make_block("human", "facts", limit=12, value="city=Berlin")
    with pytest.raises(ValueError):
        block_replace(block, "Berlin", "Berlin-Kreuzberg-Mitte")


def test_block_replace_records_history_and_version():
    block = make_block("human", "facts", limit=50, value="city=Berlin")
    updated = block_replace(block, "Berlin", "Lisbon")
    assert (updated["version"], updated["history"]) == (2, ("city=Berlin",))


# ---------------------------------------------------------------- near_limit
def test_near_limit_is_true_exactly_on_the_threshold():
    """На пороге ужимать уже надо — следующая запись переполнит блок."""
    block = make_block("human", "facts", limit=10, value="12345678")
    assert near_limit(block) is True


def test_near_limit_respects_a_stricter_threshold():
    block = make_block("human", "facts", limit=10, value="12345678")
    assert near_limit(block, 0.9) is False


def test_near_limit_is_false_for_an_empty_block():
    assert near_limit(make_block("human", "facts", limit=10)) is False


# ----------------------------------------------------------- summarize_block
def test_summarize_keeps_whole_sentences():
    """Обрезок «Audience seni» модель прочитает как факт и сошлётся на него."""
    block = make_block("task", "current task", limit=100,
                       value="Plan curriculum. Audience senior. Cite arXiv.")
    assert summarize_block(block, 35)["value"] == "Plan curriculum. Audience senior."


def test_summarize_drops_the_tail_when_the_budget_is_smaller():
    block = make_block("task", "current task", limit=100,
                       value="Plan curriculum. Audience senior. Cite arXiv.")
    assert summarize_block(block, 20)["value"] == "Plan curriculum."


def test_summarize_shrinks_the_block():
    block = make_block("task", "current task", limit=100,
                       value="Plan curriculum. Audience senior. Cite arXiv.")
    assert len(summarize_block(block, 35)["value"]) < len(block["value"])


def test_summarize_stores_the_full_value_in_history():
    block = make_block("task", "current task", limit=100,
                       value="Plan curriculum. Audience senior. Cite arXiv.")
    summarized = summarize_block(block, 35)
    assert summarized["history"][-1] == block["value"]
    assert summarized["version"] == 2


def test_summarize_of_an_empty_block_stays_empty():
    assert summarize_block(make_block("t", "task", limit=50), 10)["value"] == ""


# ----------------------------------------------------------- dedup_archival
def test_dedup_keeps_the_earlier_record():
    """На старую запись уже могли сослаться — выживает она."""
    records = [rec("a001", "ava lives in berlin"),
               rec("a002", "ava lives in berlin now")]
    assert [r["rid"] for r in dedup_archival(records, 0.7)] == ["a001"]


def test_dedup_leaves_distinct_records_alone():
    records = [rec("a001", "ava lives in berlin"),
               rec("a002", "bob requested a refund for invoice 4711")]
    assert len(dedup_archival(records, 0.7)) == 2


def test_dedup_is_idempotent():
    records = [rec("a001", "ava lives in berlin"),
               rec("a002", "ava lives in berlin now"),
               rec("a003", "ava lives in berlin already")]
    once = dedup_archival(records, 0.7)
    assert dedup_archival(once, 0.7) == once


def test_dedup_never_compares_against_invalidated_records():
    """Инвалидированное — история, а не мусор: остаётся и не съедает свежее."""
    records = [rec("a001", "ava lives in berlin", valid=False),
               rec("a002", "ava lives in berlin now")]
    assert [r["rid"] for r in dedup_archival(records, 0.7)] == ["a001", "a002"]


# --------------------------------------------------- invalidate_contradicted
def test_invalidate_marks_but_never_deletes():
    """Soft delete: удалённый факт не предъявить аудиту и не откатить."""
    records = [rec("a001", "ava lives in Berlin")]
    updated, touched = invalidate_contradicted(records, "ava lives in berlin")
    assert len(updated) == 1
    assert updated[0]["valid"] is False
    assert touched == ["a001"]


def test_invalidate_leaves_unrelated_records_valid():
    records = [rec("a001", "ava lives in Berlin"), rec("a002", "ava ships agents")]
    updated, _ = invalidate_contradicted(records, "ava lives in berlin")
    assert [r["valid"] for r in updated] == [False, True]


def test_invalidate_does_not_touch_already_invalid_records():
    records = [rec("a001", "ava lives in Berlin", valid=False)]
    _, touched = invalidate_contradicted(records, "ava lives in berlin")
    assert touched == []


def test_invalidate_leaves_the_input_list_alone():
    records = [rec("a001", "ava lives in Berlin")]
    invalidate_contradicted(records, "ava lives in berlin")
    assert records[0]["valid"] is True


# ---------------------------------------------------------- sleep_time_pass
def test_sleep_pass_does_not_touch_a_block_far_from_its_limit():
    """Каждая правка блока стоит вызов модели. «На всякий случай» — нельзя."""
    blocks = {"persona": make_block("persona", "self-concept", limit=200,
                                    value="terse and citation-heavy")}
    new_blocks, _, trace = sleep_time_pass(blocks, [])
    assert new_blocks["persona"]["version"] == 1
    assert new_blocks["persona"]["value"] == "terse and citation-heavy"
    assert trace == []


def test_sleep_pass_consolidates_a_block_at_its_limit():
    blocks = {"task": make_block("task", "current task", limit=40,
                                 value="Plan curriculum. Audience senior.")}
    new_blocks, _, trace = sleep_time_pass(blocks, [])
    assert len(new_blocks["task"]["value"]) < 32
    assert trace == ["consolidate task v2"]


def test_sleep_pass_invalidates_a_contradicted_fact_without_deleting_it():
    records = [rec("a001", "ava lives in Berlin"), rec("a002", "ava ships agents")]
    _, new_records, trace = sleep_time_pass({}, records, ("ava lives in berlin",))
    assert len(new_records) == 2
    assert [r["valid"] for r in new_records] == [False, True]
    assert "invalidate a001" in trace


def test_sleep_pass_invalidates_before_it_dedups():
    """Иначе дедуп схлопнет свежий факт в тот, который мы признаём протухшим."""
    records = [rec("a001", "ava lives in Berlin"),
               rec("a002", "ava lives in Berlin still")]
    _, new_records, _ = sleep_time_pass(
        {}, records, ("ava lives in berlin still",), dedup_threshold=0.7)
    rids = {r["rid"]: r["valid"] for r in new_records}
    assert rids == {"a001": True, "a002": False}


def test_sleep_pass_on_an_empty_agent_does_nothing():
    assert sleep_time_pass({}, []) == ({}, [], [])
