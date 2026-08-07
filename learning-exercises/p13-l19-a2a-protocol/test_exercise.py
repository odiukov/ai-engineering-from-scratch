"""Тесты к уроку «A2A — протокол общения агентов». Правь exercise.py."""

import pytest

from exercise import (
    AGENT_CARD_PATH,
    TASK_EVENTS,
    TERMINAL_STATES,
    build_agent_card,
    canonical_json,
    make_artifact,
    next_task_state,
    run_task,
    select_skill,
    sign_agent_card,
    verify_agent_card,
)

QUICK_NOTE = {
    "id": "quick_note",
    "name": "Quick note",
    "inputModes": ["text"],
    "outputModes": ["text"],
}

DRAFT_REPORT = {
    "id": "draft_report",
    "name": "Draft report",
    "inputModes": ["text", "file", "data"],
    "outputModes": ["text", "artifact"],
    "requiredData": ["targetLength"],
}

FREE_REPORT = {
    "id": "free_report",
    "name": "Report without extra data",
    "inputModes": ["text", "file", "data"],
    "outputModes": ["text"],
}


def card_with(*skills):
    return build_agent_card(
        "writer-agent",
        "Drafts technical summaries.",
        "https://writer.example.com/a2a",
        "1.0.0",
        list(skills),
    )


def text_message(text, role="user"):
    return {"role": role, "parts": [{"kind": "text", "text": text}]}


def data_message(payload, role="user"):
    return {"role": role, "parts": [{"kind": "data", "data": dict(payload)}]}


# --------------------------------------------------------- build_agent_card
def test_agent_card_carries_schema_version_and_endpoint():
    card = card_with(QUICK_NOTE)
    assert card["schemaVersion"] == "1.0"
    assert card["url"] == "https://writer.example.com/a2a"
    assert AGENT_CARD_PATH == "/.well-known/agent.json"


def test_agent_card_fills_missing_capabilities_with_false():
    """Потребитель не должен угадывать ключ, которого нет в карточке."""
    card = build_agent_card("a", "d", "u", "1", [QUICK_NOTE], {"streaming": True})
    assert card["capabilities"] == {"streaming": True, "pushNotifications": False}


def test_agent_card_snapshots_skills_instead_of_aliasing_them():
    """Правка исходного skill после публикации не имеет права менять карточку."""
    mutable = dict(QUICK_NOTE)
    card = card_with(mutable)
    mutable["id"] = "hijacked"
    assert card["skills"][0]["id"] == "quick_note"


def test_agent_card_rejects_duplicate_skill_ids():
    with pytest.raises(ValueError):
        card_with(QUICK_NOTE, dict(QUICK_NOTE, name="Copy"))


# ------------------------------------------------------------- select_skill
def test_select_skill_picks_the_narrowest_capable_skill():
    card = card_with(QUICK_NOTE, DRAFT_REPORT)
    assert select_skill(card, ["text"], "text") == "quick_note"


def test_select_skill_needs_every_requested_part_kind():
    card = card_with(QUICK_NOTE, DRAFT_REPORT)
    assert select_skill(card, ["text", "file"], "text") == "draft_report"


def test_select_skill_returns_none_when_output_mode_is_unsupported():
    card = card_with(QUICK_NOTE, DRAFT_REPORT)
    assert select_skill(card, ["text"], "audio") is None


def test_select_skill_does_not_depend_on_skill_order_in_the_card():
    """Два подходящих skill обязаны разрешаться одинаково при любом порядке."""
    forward = card_with(QUICK_NOTE, DRAFT_REPORT)
    backward = card_with(DRAFT_REPORT, QUICK_NOTE)
    assert select_skill(forward, ["text"], "text") == select_skill(
        backward, ["text"], "text"
    )


def test_select_skill_rejects_an_unknown_part_kind():
    """Опечатка в виде Part — ошибка, а не тихое None."""
    with pytest.raises(ValueError):
        select_skill(card_with(QUICK_NOTE), ["txt"], "text")


# ------------------------------------------------------------ canonical_json
def test_canonical_json_ignores_key_insertion_order():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_has_no_spaces_after_separators():
    assert canonical_json({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_canonical_json_keeps_non_ascii_readable():
    """ensure_ascii=True раздул бы строку в \\uXXXX и сломал бы длину подписи."""
    assert canonical_json({"name": "агент"}) == '{"name":"агент"}'


# ---------------------------------------------------------- sign_agent_card
def test_signature_is_stable_for_the_same_card_and_secret():
    card = card_with(QUICK_NOTE)
    assert (
        sign_agent_card(card, "s3cret")["signature"]
        == sign_agent_card(card, "s3cret")["signature"]
    )


def test_signature_changes_with_the_secret():
    card = card_with(QUICK_NOTE)
    assert (
        sign_agent_card(card, "s3cret")["signature"]
        != sign_agent_card(card, "other")["signature"]
    )


def test_signing_snapshots_the_card_at_signing_time():
    """Подписали — значит зафиксировали; правка оригинала снимок не трогает."""
    card = card_with(QUICK_NOTE)
    signed = sign_agent_card(card, "s3cret")
    card["url"] = "https://evil.example.com/a2a"
    assert signed["card"]["url"] == "https://writer.example.com/a2a"


# -------------------------------------------------------- verify_agent_card
def test_verify_accepts_a_freshly_signed_card():
    signed = sign_agent_card(card_with(QUICK_NOTE), "s3cret")
    assert verify_agent_card(signed, "s3cret") is True


def test_verify_rejects_a_mutated_card():
    """Ради этого подпись и нужна: подменённый url обязан не пройти."""
    signed = sign_agent_card(card_with(QUICK_NOTE), "s3cret")
    signed["card"]["url"] = "https://evil.example.com/a2a"
    assert verify_agent_card(signed, "s3cret") is False


def test_verify_rejects_the_wrong_secret():
    signed = sign_agent_card(card_with(QUICK_NOTE), "s3cret")
    assert verify_agent_card(signed, "guess") is False


def test_verify_rejects_an_unknown_algorithm():
    """alg "none" — классическая дыра: проверка обязана отказать, а не пропустить."""
    signed = sign_agent_card(card_with(QUICK_NOTE), "s3cret")
    signed["alg"] = "none"
    assert verify_agent_card(signed, "s3cret") is False


# --------------------------------------------------------- next_task_state
def test_submitted_task_starts_working_on_accept():
    assert next_task_state("submitted", "accept") == "working"


def test_input_required_returns_to_working_when_input_arrives():
    assert next_task_state("input_required", "provide_input") == "working"


def test_terminal_states_accept_no_event_at_all():
    """Завершённую задачу нельзя ни отменить, ни продолжить."""
    for state in TERMINAL_STATES:
        for event in TASK_EVENTS:
            with pytest.raises(ValueError):
                next_task_state(state, event)


def test_illegal_transition_between_known_states_is_rejected():
    with pytest.raises(ValueError):
        next_task_state("submitted", "finish")


def test_unknown_state_name_is_rejected():
    with pytest.raises(ValueError):
        next_task_state("in_progress", "accept")


# ------------------------------------------------------------ make_artifact
def test_artifact_joins_chunks_without_separators():
    art = make_artifact("summary", "text/markdown", ["Hel", "lo"])
    assert art["parts"][0]["text"] == "Hello"


def test_artifact_does_not_depend_on_how_the_stream_was_cut():
    """Границы чанков — свойство сети, а не артефакта."""
    a = make_artifact("summary", "text/markdown", ["ab", "c", "de"])
    b = make_artifact("summary", "text/markdown", ["abcde"])
    assert a == b


def test_artifact_from_an_empty_stream_is_empty_not_missing():
    art = make_artifact("summary", "text/markdown", [])
    assert art["parts"] == [{"kind": "text", "text": ""}]


def test_artifact_refuses_a_non_string_chunk():
    with pytest.raises(TypeError):
        make_artifact("summary", "text/markdown", ["ok", 42])


# ----------------------------------------------------------------- run_task
def test_unknown_skill_is_rejected_not_failed():
    """rejected — «не брались», failed — «взялись и не смогли». Это разное."""
    task = run_task("t1", card_with(QUICK_NOTE), "no_such_skill", [])
    assert (task["state"], task["artifact"]) == ("rejected", None)


def test_task_pauses_when_required_data_is_missing():
    task = run_task("t1", card_with(DRAFT_REPORT), "draft_report", [text_message("go")])
    assert task["state"] == "input_required"
    assert task["artifact"] is None
    assert task["messages"][-1]["role"] == "agent"


def test_task_resumes_and_completes_when_the_data_arrives():
    task = run_task(
        "t1",
        card_with(DRAFT_REPORT),
        "draft_report",
        [text_message("summarize this"), data_message({"targetLength": "3 paragraphs"})],
    )
    assert task["state"] == "completed"
    assert task["artifact"]["parts"][0]["text"] == "summarize this"


def test_task_without_required_data_completes_on_the_first_message():
    task = run_task(
        "t1", card_with(FREE_REPORT), "free_report", [text_message("just do it")]
    )
    assert task["state"] == "completed"


def test_task_exposes_only_the_public_a2a_surface():
    """Непрозрачность: наружу — состояние, переписка, артефакт. Больше ничего."""
    task = run_task(
        "t1", card_with(FREE_REPORT), "free_report", [text_message("hello")]
    )
    assert set(task) == {"id", "skillId", "state", "messages", "artifact"}


def test_task_does_not_mutate_the_caller_messages():
    messages = [text_message("hello")]
    run_task("t1", card_with(FREE_REPORT), "free_report", messages)
    assert messages == [{"role": "user", "parts": [{"kind": "text", "text": "hello"}]}]
