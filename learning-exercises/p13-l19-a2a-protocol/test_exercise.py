"""Тесты к уроку «A2A v1.0». Правь exercise.py."""

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
    "id": "quick_note", "name": "Quick note", "description": "Write a note.",
    "tags": ["writing"], "inputModes": ["text/plain"], "outputModes": ["text/plain"],
}
DRAFT_REPORT = {
    "id": "draft_report", "name": "Draft report", "description": "Draft a report.",
    "tags": ["writing"],
    "inputModes": ["text/plain", "application/pdf", "application/json"],
    "outputModes": ["text/markdown"], "requiredData": ["targetLength"],
}
FREE_REPORT = dict(DRAFT_REPORT, id="free_report", requiredData=[])


def card_with(*skills):
    return build_agent_card("writer-agent", "Drafts summaries.", "https://writer.example/a2a", "1.0.0", list(skills))


def text_message(text, message_id="msg-1"):
    return {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": text}]}


def data_message(data, message_id="msg-2"):
    return {"messageId": message_id, "role": "ROLE_USER", "parts": [{"data": dict(data), "mediaType": "application/json"}]}


def test_agent_card_uses_current_path_and_supported_interfaces():
    card = card_with(QUICK_NOTE)
    assert AGENT_CARD_PATH == "/.well-known/agent-card.json"
    assert "schemaVersion" not in card and "url" not in card
    assert card["supportedInterfaces"] == [{"url": "https://writer.example/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]


def test_agent_card_carries_required_default_modes_and_capabilities():
    card = build_agent_card("a", "d", "u", "1", [QUICK_NOTE], {"streaming": True})
    assert card["defaultInputModes"] == ["text/plain"]
    assert card["defaultOutputModes"] == ["text/plain"]
    assert card["capabilities"] == {"streaming": True, "pushNotifications": False}


def test_agent_card_snapshots_skills_and_rejects_duplicate_ids():
    mutable = dict(QUICK_NOTE)
    card = card_with(mutable)
    mutable["id"] = "changed"
    assert card["skills"][0]["id"] == "quick_note"
    with pytest.raises(ValueError):
        card_with(QUICK_NOTE, dict(QUICK_NOTE))


def test_select_skill_uses_mime_modes_and_is_order_independent():
    forward = card_with(QUICK_NOTE, DRAFT_REPORT)
    backward = card_with(DRAFT_REPORT, QUICK_NOTE)
    assert select_skill(forward, ["text/plain"], "text/plain") == "quick_note"
    assert select_skill(backward, ["text/plain"], "text/plain") == "quick_note"
    assert select_skill(forward, ["application/pdf"], "text/markdown") == "draft_report"
    assert select_skill(forward, ["audio/wav"], "text/plain") is None


def test_canonical_json_is_stable_compact_and_unicode_readable():
    assert canonical_json({"b": 1, "a": "агент"}) == '{"a":"агент","b":1}'


def test_signatures_live_inside_the_agent_card_in_current_shape():
    signed = sign_agent_card(card_with(QUICK_NOTE), "s3cret")
    assert set(signed["signatures"][0]) == {"protected", "signature"}
    assert verify_agent_card(signed, "s3cret") is True


def test_signature_rejects_mutation_and_wrong_secret():
    signed = sign_agent_card(card_with(QUICK_NOTE), "s3cret")
    signed["description"] = "tampered"
    assert verify_agent_card(signed, "s3cret") is False
    assert verify_agent_card(sign_agent_card(card_with(QUICK_NOTE), "s3cret"), "wrong") is False


def test_task_state_uses_v1_enum_values_and_has_no_terminal_exit():
    assert next_task_state("TASK_STATE_SUBMITTED", "accept") == "TASK_STATE_WORKING"
    assert next_task_state("TASK_STATE_INPUT_REQUIRED", "provide_input") == "TASK_STATE_WORKING"
    for state in TERMINAL_STATES:
        for event in TASK_EVENTS:
            with pytest.raises(ValueError):
                next_task_state(state, event)


def test_artifact_has_artifact_id_plural_container_part_without_kind():
    artifact = make_artifact("art-1", "summary", "text/markdown", ["Hel", "lo"])
    assert artifact["artifactId"] == "art-1"
    assert artifact["parts"] == [{"text": "Hello", "mediaType": "text/markdown"}]
    assert "kind" not in artifact["parts"][0]


def test_artifact_rejects_non_string_chunk():
    with pytest.raises(TypeError):
        make_artifact("art-1", "summary", "text/plain", ["ok", 42])


def test_part_requires_exactly_one_content_field():
    bad = {"messageId": "m", "role": "ROLE_USER", "parts": [{"text": "x", "data": {}}]}
    with pytest.raises(ValueError):
        run_task("t1", card_with(FREE_REPORT), "free_report", [bad])


def test_task_uses_status_history_and_artifacts_shapes():
    task = run_task("t1", card_with(FREE_REPORT), "free_report", [text_message("hello")])
    assert set(task) == {"id", "contextId", "status", "history", "artifacts"}
    assert task["status"] == {"state": "TASK_STATE_COMPLETED"}
    assert task["history"][0]["messageId"] == "msg-1"
    assert task["artifacts"][0]["artifactId"] == "art_t1"


def test_task_pauses_and_resumes_with_status_message():
    waiting = run_task("t1", card_with(DRAFT_REPORT), "draft_report", [text_message("go")])
    assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert waiting["status"]["message"]["role"] == "ROLE_AGENT"
    done = run_task("t2", card_with(DRAFT_REPORT), "draft_report", [text_message("summarize"), data_message({"targetLength": "short"})])
    assert done["status"] == {"state": "TASK_STATE_COMPLETED"}
    assert done["artifacts"][0]["parts"][0]["text"] == "summarize"


def test_unknown_skill_is_rejected_and_messages_are_not_mutated():
    messages = [text_message("hello")]
    rejected = run_task("t1", card_with(QUICK_NOTE), "missing", messages)
    assert rejected["status"]["state"] == "TASK_STATE_REJECTED"
    run_task("t2", card_with(FREE_REPORT), "free_report", messages)
    assert messages == [text_message("hello")]
