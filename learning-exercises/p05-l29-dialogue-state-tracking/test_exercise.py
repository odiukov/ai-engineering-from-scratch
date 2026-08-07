"""Тесты к уроку «Dialogue state tracking». Правь exercise.py."""

import json

import pytest

from exercise import (
    extract_slots,
    is_negated,
    joint_goal_accuracy,
    llm_dst,
    slot_accuracy,
    track_dialogue,
    update_state,
    validate_state,
)

APPROX = lambda x: pytest.approx(x, abs=1e-9)

ONT = {
    "cuisine": {
        "italian": ["italian", "pasta", "pizza"],
        "chinese": ["chinese", "dim sum"],
        "any": ["any food", "any cuisine"],
    },
    "area": {"north": ["north"], "center": ["center", "centre"]},
    "price": {
        "cheap": ["cheap", "budget"],
        "moderate": ["moderate", "mid-range"],
        "expensive": ["expensive", "fancy"],
    },
}
CUES = ["never mind", "forget about"]
SCHEMA = {
    "cuisine": ["italian", "chinese", "any"],
    "area": ["north", "center"],
    "price": ["cheap", "moderate", "expensive"],
    "name": None,
}


# ------------------------------------------------------------- extract_slots
def test_extract_slots_finds_several_slots_in_one_utterance():
    assert extract_slots("a cheap italian place in the north", ONT) == {
        "cuisine": "italian",
        "area": "north",
        "price": "cheap",
    }


def test_extract_slots_leaves_unmentioned_slots_out_of_the_dict():
    """Отсутствие ключа и ключ со значением None — разные вещи."""
    assert extract_slots("hello there", ONT) == {}


def test_extract_slots_respects_word_boundaries():
    """north внутри northern — не район, а совпадение подстроки."""
    assert "area" not in extract_slots("northern lights over the city", ONT)


def test_extract_slots_matches_a_multi_word_synonym():
    assert extract_slots("some dim sum please", ONT) == {"cuisine": "chinese"}


# ---------------------------------------------------------------- is_negated
def test_is_negated_needs_both_a_cue_and_the_slot_name():
    assert is_negated("never mind the cuisine", "cuisine", CUES) is True


def test_is_negated_does_not_touch_other_slots():
    """Отказ от кухни не должен стирать район."""
    assert is_negated("never mind the cuisine", "area", CUES) is False


def test_is_negated_ignores_a_slot_mention_without_a_cue():
    assert is_negated("the cuisine matters to me", "cuisine", CUES) is False


# -------------------------------------------------------------- update_state
def test_update_state_fills_a_new_slot():
    assert update_state({}, "a cheap place", ONT, CUES) == {"price": "cheap"}


def test_update_state_keeps_slots_the_user_did_not_touch():
    """Пользователь уточнил цену — район обязан остаться."""
    state = {"area": "north", "price": "cheap"}
    assert update_state(state, "make it moderate", ONT, CUES) == {
        "area": "north",
        "price": "moderate",
    }


def test_update_state_overwrites_on_correction_instead_of_appending():
    updated = update_state({"price": "cheap"}, "actually moderate", ONT, CUES)
    assert updated["price"] == "moderate"


def test_update_state_clears_an_explicitly_negated_slot():
    updated = update_state({"cuisine": "italian"}, "never mind the cuisine", ONT, CUES)
    assert updated["cuisine"] is None


def test_update_state_prefers_an_extracted_value_over_a_negation():
    """«never mind the cuisine, any food is fine» — это значение any, а не None."""
    updated = update_state(
        {"cuisine": "italian"}, "never mind the cuisine, any food is fine", ONT, CUES
    )
    assert updated["cuisine"] == "any"


def test_update_state_does_not_mutate_the_state_it_was_given():
    state = {"price": "cheap"}
    update_state(state, "make it expensive", ONT, CUES)
    assert state == {"price": "cheap"}


# ------------------------------------------------------------ track_dialogue
def test_track_dialogue_returns_one_state_per_turn():
    turns = ["a cheap place", "italian food", "in the north"]
    assert len(track_dialogue(turns, ONT, CUES)) == 3


def test_track_dialogue_accumulates_slots_across_turns():
    turns = ["a cheap place", "italian food", "in the north"]
    assert track_dialogue(turns, ONT, CUES)[-1] == {
        "price": "cheap",
        "cuisine": "italian",
        "area": "north",
    }


def test_track_dialogue_snapshots_are_independent():
    """Ранний снимок не должен задним числом получить поздние слоты."""
    history = track_dialogue(["a cheap place", "italian food"], ONT, CUES)
    assert history[0] == {"price": "cheap"}


def test_track_dialogue_starts_from_the_initial_state_without_mutating_it():
    initial = {"area": "north"}
    history = track_dialogue(["a cheap place"], ONT, CUES, initial)
    assert history[0] == {"area": "north", "price": "cheap"}
    assert initial == {"area": "north"}


# ------------------------------------------------------------ validate_state
def test_validate_state_drops_a_slot_that_is_not_in_the_schema():
    assert validate_state({"colour": "red"}, SCHEMA) == {}


def test_validate_state_normalises_case_and_whitespace():
    assert validate_state({"cuisine": " ITALIAN "}, SCHEMA) == {"cuisine": "italian"}


def test_validate_state_drops_a_value_outside_the_closed_set():
    assert validate_state({"price": "gratis"}, SCHEMA) == {}


def test_validate_state_accepts_anything_in_a_free_form_slot_and_keeps_cleared_slots():
    assert validate_state({"name": "The Copper Kettle", "area": None}, SCHEMA) == {
        "name": "the copper kettle",
        "area": None,
    }


# ------------------------------------------------------------------- llm_dst
def test_llm_dst_shows_the_model_the_whole_history():
    """Регенерация состояния имеет смысл только по всем ходам сразу."""
    seen = []

    def spy(prompt):
        seen.append(prompt)
        return '{"price": "cheap"}'

    llm_dst(["a cheap place", "italian food"], spy, SCHEMA)
    assert "a cheap place" in seen[0]
    assert "italian food" in seen[0]


def test_llm_dst_validates_what_the_model_returned():
    """Придуманный моделью слот не должен доехать до бэкенда."""
    raw = json.dumps({"cuisine": "Italian", "vibe": "cosy"})
    assert llm_dst(["italian food"], lambda p: raw, SCHEMA) == {"cuisine": "italian"}


def test_llm_dst_returns_none_when_the_model_answers_with_prose():
    """None — «модель не ответила», а пустое состояние — «пользователь ничего не просил»."""
    assert llm_dst(["italian food"], lambda p: "sorry, I cannot", SCHEMA) is None


def test_llm_dst_survives_json_wrapped_in_prose():
    raw = 'Here is the state:\n```json\n{"area": "north"}\n```'
    assert llm_dst(["north please"], lambda p: raw, SCHEMA) == {"area": "north"}


# ------------------------------------------------------- joint_goal_accuracy
def test_joint_goal_accuracy_counts_fully_correct_turns():
    pred = [{"a": 1}, {"a": 1, "b": 2}]
    gold = [{"a": 1}, {"a": 1, "b": 3}]
    assert joint_goal_accuracy(pred, gold) == APPROX(0.5)


def test_joint_goal_accuracy_punishes_an_extra_slot():
    """Лишний слот — тоже неверное состояние: бэкенд получит не тот запрос."""
    assert joint_goal_accuracy([{"a": 1, "b": 2}], [{"a": 1}]) == APPROX(0.0)


def test_joint_goal_accuracy_of_an_empty_run_is_zero():
    assert joint_goal_accuracy([], []) == APPROX(0.0)


def test_joint_goal_accuracy_rejects_lists_of_different_length():
    with pytest.raises(ValueError):
        joint_goal_accuracy([{"a": 1}], [])


# -------------------------------------------------------------- slot_accuracy
def test_slot_accuracy_counts_slots_not_turns():
    assert slot_accuracy([{"a": 1, "b": 2}], [{"a": 1, "b": 9}]) == APPROX(0.5)


def test_slot_accuracy_counts_a_forgotten_slot_as_an_error():
    """Не предсказать слот — ошибка, а не повод убрать его из знаменателя."""
    assert slot_accuracy([{"a": 1}], [{"a": 1, "b": 2}]) == APPROX(0.5)


def test_joint_goal_accuracy_is_stricter_than_slot_accuracy():
    """Один неверный слот из четырёх: по слотам 0.75, по ходам ноль."""
    pred = [{"a": 1, "b": 2, "c": 3, "d": 9}]
    gold = [{"a": 1, "b": 2, "c": 3, "d": 4}]
    assert slot_accuracy(pred, gold) == APPROX(0.75)
    assert joint_goal_accuracy(pred, gold) == APPROX(0.0)


def test_slot_accuracy_rejects_lists_of_different_length():
    with pytest.raises(ValueError):
        slot_accuracy([{"a": 1}], [])
